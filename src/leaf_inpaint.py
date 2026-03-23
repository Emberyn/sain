import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# 自定义模块：数据集加载、模型定义、工具函数、评估指标
from .dataset import Dataset
from .models import InpaintingModel, EdgeModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
# 图像质量评估指标：结构相似性(SSIM)、峰值信噪比(PSNR)
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# 实验跟踪工具
import wandb
# os.environ['WANDB_DISABLED'] = 'true'  # 关闭wandb
# wandb.init(mode="offline")  # 离线模式运行wandb
# 感知损失计算库
import lpips
import torchvision

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''


class LEAFINPAINT():
    """
    叶片图像修复(LEAFINPAINT)主类
    核心功能：
    1. 初始化模型、数据集、损失函数、评估指标
    2. 训练边缘生成模型/图像修复模型/联合模型
    3. 测试模型并生成修复结果，计算评估指标
    """

    def __init__(self, config):
        """
        初始化函数
        Args:
            config: 配置类对象，包含所有训练/测试参数（如设备、路径、模型类型、批次大小等）
        """
        self.config = config
        # self.best_val_loss = float('inf')  # 初始化为无穷大  用作保存最佳参数

        # 根据配置选择模型类型
        if config.MODEL == 1:
            model_name = 'edge'  # 仅训练边缘生成模型
        elif config.MODEL == 2:
            model_name = 'inpaint'  # 仅训练图像修复模型
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'  # 先生成边缘再修复图像（联合模型）
        # elif config.MODEL == 4:
        #     model_name = 'joint'
        print('model_name == ', model_name)

        self.debug = False  # 调试模式标记
        self.model_name = model_name  # 当前模型名称

        # 初始化模型并加载到指定设备（CPU/GPU）
        self.edge_model = EdgeModel(config).to(config.DEVICE)  # 边缘生成模型
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)  # 图像修复模型

        # 图像归一化变换：将像素值从[0,1]转换为[-1,1]（适配LPIPS损失计算）
        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # 初始化LPIPS损失函数（基于VGG网络的感知损失）
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        # 初始化PSNR评估指标（峰值信噪比，最大值255）
        self.psnr = PSNR(255.0).to(config.DEVICE)
        # 初始化MAE损失（L1损失，求和模式）
        self.cal_mae = nn.L1Loss(reduction='sum')

        # 训练模式：加载训练数据集
        if self.config.MODE == 1:
            self.train_dataset = Dataset(
                config,
                config.TRAIN_INPAINT_IMAGE_FLIST,  # 训练图像列表路径
                config.TRAIN_MASK_FLIST,  # 训练掩码列表路径
                config.TRAIN_INPAINT_EDGE_FLIST,  # 训练边缘图列表路径
                augment=True,  # 数据增强
                training=True
            )
            # 验证集暂时注释，如需启用需确保配置文件中有对应路径
            # self.val_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
            #                            config.TEST_INPAINT_SYM_FLIST, augment=False,
            #                            training=False)  # 目前验证集是按照测试集写的

        # 测试模式：加载测试数据集
        if self.config.MODE == 2:
            print('MODE == 2')
            self.test_dataset = Dataset(
                config,
                config.TEST_INPAINT_IMAGE_FLIST,  # 测试图像列表路径
                config.TEST_MASK_FLIST,  # 测试掩码列表路径
                config.TEST_INPAINT_EDGE_FLIST,  # 测试边缘图列表路径
                augment=False,  # 关闭数据增强
                training=False
            )


        # 定义结果保存路径
        self.samples_path = os.path.join(config.PATH, 'samples')  # 采样结果路径
        self.results_path = os.path.join(config.PATH, 'results')  # 测试/训练可视化结果路径


        # 如果配置中指定了自定义结果路径，则覆盖默认路径
        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        # 启用调试模式
        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        # 日志文件路径：记录训练过程中的损失、指标等
        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        """
        加载预训练模型权重
        根据模型类型选择加载边缘模型/修复模型/两者都加载
        """
        if self.config.MODEL == 1:
            self.edge_model.load()  # 仅加载边缘生成模型

        elif self.config.MODEL == 2:
            self.inpaint_model.load()  # 仅加载图像修复模型

        else:
            self.edge_model.load()  # 加载边缘生成模型
            self.inpaint_model.load()  # 加载图像修复模型

    def save(self):
        """
        保存当前模型权重
        根据模型类型选择保存边缘模型/修复模型/两者都保存
        """
        if self.config.MODEL == 1:
            self.edge_model.save()  # 仅保存边缘生成模型

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()  # 保存图像修复模型（联合模型仅保存修复模型）

        else:
            self.edge_model.save()  # 保存边缘生成模型
            self.inpaint_model.save()  # 保存图像修复模型

    # def save_best(self):  # 保存最佳模型（按验证集损失）
    #     if self.config.MODEL == 2:
    #         self.inpaint_model.save_best()

    def train(self):
        """
        模型训练主函数
        支持三种训练模式：边缘生成、图像修复、边缘引导的图像修复
        """
        # wandb.watch(self.inpaint_model, self.psnr, log='all', log_freq=10)  # wandb监控模型

        # 打印训练数据集长度
        print("len(self.train_dataset)=", (len(self.train_dataset)))

        # 创建训练数据加载器
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,  # 批次大小
            num_workers=8,  # 数据加载线程数（Windows系统需改为0，否则会报多线程错误）
            drop_last=True,  # 丢弃最后一个不完整批次
            shuffle=True  # 打乱数据顺序
        )

        # 验证集加载器（暂时注释）
        # val_loader = DataLoader(
        #     dataset=self.val_dataset,
        #     batch_size=self.config.BATCH_SIZE,
        #     shuffle=False
        # )

        epoch = 0  # 初始化训练轮数
        print("epoch=", epoch)

        keep_training = True  # 训练循环标记
        model = self.config.MODEL  # 当前训练模型类型
        max_iteration = int(float((self.config.MAX_ITERS)))  # 最大训练迭代次数
        total = len(self.train_dataset)  # 训练数据集总样本数

        # 训练主循环
        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            # 进度条初始化：显示训练进度、损失、指标等
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            # 遍历训练批次
            for items in train_loader:
                # 设置模型为训练模式（启用dropout、BN等训练层）
                self.edge_model.train()
                self.inpaint_model.train()

                # 将数据加载到指定设备（CPU/GPU）
                # items包含：原始图像、掩码、边缘图、灰度图
                images, masks, edge, gray_img = self.cuda(*items)

                # 将单通道掩码转换为三通道（匹配图像通道数）
                if masks.shape[1] == 1:
                    masks = torch.cat([masks, masks, masks], dim=1)

                # 将单通道边缘图转换为三通道（匹配图像通道数）
                if edge.shape[1] == 1:
                    edge = torch.cat([edge, edge, edge], dim=1)

                # -------------------------- 1. 仅训练边缘生成模型 --------------------------
                if model == 1:
                    # 前向传播：获取边缘预测结果和各类损失
                    outputs_edge, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, ssim_loss, gen_fm_loss = self.edge_model.process(
                        images, masks, edge, gray_img
                    )

                    # 融合预测边缘与原始边缘：仅修复掩码区域（孔洞），保留非掩码区域原始边缘
                    outputs_merged = (outputs_edge * masks) + (edge * (1 - masks))

                    # 计算PSNR（峰值信噪比）：评估边缘修复质量
                    # psnr = self.psnr(self.postprocess(edge), self.postprocess(outputs_merged))
                    psnr = self.psnr(edge * 255.0, outputs_merged * 255.0)
                    # 计算MAE（平均绝对误差）：评估边缘修复误差
                    mae = (torch.sum(torch.abs(edge - outputs_merged)) / torch.sum(edge)).float()

                    # 将PSNR和MAE加入日志
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # 打印各类损失值（调试用）
                    # print("gen_loss=", gen_loss.item(), " dis_loss=", dis_loss.item(), " gen_gan_loss=",
                    #       gen_gan_loss.item(), " gen_l1_loss=", gen_l1_loss.item())
                    # print("gen_content_loss=", gen_content_loss.item(), " gen_style_loss=", gen_style_loss.item(),
                    #       " ssim_loss=", ssim_loss.item(), " gen_fm_loss=", gen_fm_loss)

                    # 反向传播：更新边缘模型参数
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration  # 当前迭代次数

                # -------------------------- 2. 仅训练图像修复模型 --------------------------
                elif model == 2:
                    # 前向传播：获取图像修复结果和各类损失
                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss = self.inpaint_model.process(
                        images, masks, edge, gray_img
                    )

                    # 融合修复图像与原始图像：仅修复掩码区域（孔洞），保留非掩码区域原始图像
                    outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                    # 计算PSNR：评估图像修复质量
                    # psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    psnr = self.psnr(images * 255.0, outputs_merged * 255.0)
                    # 计算MAE：评估图像修复误差
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    # 将PSNR和MAE加入日志
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # 反向传播：更新图像修复模型参数
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration  # 当前迭代次数

                # -------------------------- 3. 边缘引导的图像修复（联合模型） --------------------------
                elif model == 3:
                    # 将边缘模型设为评估模式（禁用dropout、BN等训练层）
                    self.edge_model.eval()
                    with torch.no_grad():  # 禁用梯度计算（节省显存，加速推理）
                        # 生成边缘图
                        outputs_edge = self.edge_model(images, masks, edge, gray_img)
                    # 融合预测边缘与原始边缘：仅修复掩码区域
                    outputs_edge = (outputs_edge * masks) + (edge * (1 - masks))

                    # # 前5萬次，用真的edge訓練（注释的实验策略）
                    # outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss = self.inpaint_model.process(
                    #     images, masks, edge, gray_img)

                    # 后5w萬次，用生成的边缘图训练图像修复模型
                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss = self.inpaint_model.process(
                        images, masks, outputs_edge, gray_img
                    )

                    # 融合修复图像与原始图像：仅修复掩码区域
                    outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                    # 计算PSNR：评估图像修复质量
                    # psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    psnr = self.psnr(images * 255.0, outputs_merged * 255.0)
                    # 计算MAE：评估图像修复误差
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    # 将PSNR和MAE加入日志
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # 反向传播：仅更新图像修复模型参数（边缘模型已冻结）
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration  # 当前迭代次数

                # 达到最大迭代次数，终止训练
                if iteration >= max_iteration:
                    keep_training = False
                    break




                # 整理日志：轮数、迭代次数 + 各类损失/指标
                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs


                # 更新进度条：仅在详细模式下显示所有日志，否则隐藏以'l_'开头的损失
                progbar.add(len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])




                # wandb日志记录（注释，如需启用需取消注释）
                # if iteration % 10 == 0:
                #     if model == 1:   # edge 记录loss个数与
                #         wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                #                    'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss, 'ssim_loss': ssim_loss,
                #                    'dis_loss': dis_loss}, step=iteration)
                #
                #     elif model == 2 or model == 3:
                #         wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                #                    'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                #                    'dis_loss': dis_loss}, step=iteration)

                ###################### 训练过程可视化 ######################
                if iteration % 40 == 0:
                    # 创建结果保存目录
                    create_dir(self.results_path)

                    # 边缘模型：拼接原始边缘、掩码边缘、预测边缘、融合边缘
                    if model == 1:
                        inputs = (edge * (1 - masks))  # 掩码后的边缘图（孔洞区域为0）
                        images_joint = stitch_images(
                            self.postprocess(edge),  # 原始边缘图
                            self.postprocess(inputs),  # 掩码后的边缘图
                            self.postprocess(outputs_edge),  # 预测的完整边缘图
                            self.postprocess(outputs_merged),  # 融合后的边缘图
                            img_per_row=1  # 每行显示1张图
                        )
                    # 图像修复模型：拼接原始图像、掩码图像、修复图像、融合图像
                    elif model == 2:
                        inputs = (images * (1 - masks))  # 掩码后的图像（孔洞区域为0）
                        images_joint = stitch_images(
                            self.postprocess(images),  # 原始图像
                            self.postprocess(inputs),  # 掩码后的图像
                            self.postprocess(outputs_img),  # 预测的完整图像
                            self.postprocess(outputs_merged),  # 融合后的图像
                            img_per_row=1
                        )
                    # 联合模型：拼接原始图像、掩码图像、预测边缘、修复图像、融合图像
                    elif model == 3:
                        inputs = (images * (1 - masks))  # 掩码后的图像
                        images_joint = stitch_images(
                            self.postprocess(images),  # 原始图像
                            self.postprocess(inputs),  # 掩码后的图像
                            self.postprocess(outputs_edge),  # 预测的边缘图
                            self.postprocess(outputs_img),  # 修复的图像
                            self.postprocess(outputs_merged),  # 融合后的图像
                            img_per_row=1
                        )

                    # 定义可视化结果保存路径
                    path_masked = os.path.join(self.results_path, self.model_name, 'masked')  # 掩码图像路径
                    path_result = os.path.join(self.results_path, self.model_name, 'result')  # 修复结果路径
                    path_joint = os.path.join(self.results_path, self.model_name, 'joint')  # 拼接对比图路径
                    # 获取当前样本名称（去掉后缀，改为png）
                    name = f"{str(iteration).zfill(6)}.png"

                    # 创建保存目录
                    create_dir(path_masked)
                    create_dir(path_result)
                    create_dir(path_joint)

                    # 处理掩码图像和修复结果图像（转换为PIL格式）
                    masked_images = self.postprocess(images * (1 - masks) + masks)[0]  # 掩码图像（孔洞区域为白色）
                    images_result = self.postprocess(outputs_merged)[0]  # 修复结果图像

                    # 打印保存路径（调试用）
                    print(os.path.join(path_joint, name[:-4] + '.png'))

                    # 保存拼接对比图、掩码图像、修复结果图像
                    images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
                    imsave(masked_images, os.path.join(path_masked, name))
                    imsave(images_result, os.path.join(path_result, name))

                    print(name + ' complete!')
                ###################### 可视化结束 ######################

                # 日志记录：每隔LOG_INTERVAL次迭代保存一次日志
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # 模型保存：每隔SAVE_INTERVAL次迭代保存一次模型权重
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')



    def test(self):
        """
        模型测试主函数
        功能：
        1. 加载测试数据集，用预训练模型推理
        2. 生成修复结果并保存可视化对比图
        3. 计算PSNR、SSIM、L1、LPIPS等评估指标（注释，需启用可取消注释）
        """
        # 设置模型为评估模式（禁用dropout、BN等训练层）
        self.edge_model.eval()
        self.inpaint_model.eval()
        model = self.config.MODEL  # 当前测试模型类型
        create_dir(self.results_path)  # 创建结果保存目录
        cal_mean_nme = self.cal_mean_nme()  # 初始化NME（归一化均方误差）计算器

        # 创建测试数据加载器（批次大小=1，便于逐张处理）
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        # 初始化评估指标列表
        psnr_list = []  # PSNR值列表
        ssim_list = []  # SSIM值列表
        l1_list = []  # L1损失列表
        lpips_list = []  # LPIPS损失列表

        print('here')
        index = 0  # 测试样本索引
        # 遍历测试批次
        for items in test_loader:
            # 加载数据到指定设备
            images, masks, edge, gray_img = self.cuda(*items)
            index += 1  # 索引自增

            # 将单通道掩码转换为三通道
            if masks.shape[1] == 1:
                masks = torch.cat([masks, masks, masks], dim=1)

            # 将单通道边缘图转换为三通道
            if edge.shape[1] == 1:
                edge = torch.cat([edge, edge, edge], dim=1)

            # -------------------------- 1. 测试边缘生成模型 --------------------------
            if model == 1:
                # 掩码后的边缘图（孔洞区域为0）
                inputs = (edge * (1 - masks))
                with torch.no_grad():  # 禁用梯度计算
                    # 推理生成边缘图
                    outputs_edge = self.edge_model(images, masks, edge, gray_img)
                # 融合预测边缘与原始边缘：仅修复掩码区域
                outputs_merged = (outputs_edge * masks) + (edge * (1 - masks))

                # 计算评估指标（注释，需启用可取消注释）
                # psnr, ssim = self.metric(edge, outputs_merged)
                # psnr_list.append(psnr)
                # ssim_list.append(ssim)
                #
                # if torch.cuda.is_available():
                #     pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(),
                #                           self.transf(edge[0].cpu()).cuda()).item()
                #     lpips_list.append(pl)
                # else:
                #     pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(edge[0].cpu())).item()
                #     lpips_list.append(pl)
                #
                # l1_loss = torch.nn.functional.l1_loss(outputs_merged, edge, reduction='mean').item()
                # l1_list.append(l1_loss)
                #
                # print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                #                                                                 ssim, np.average(ssim_list),
                #                                                                 l1_loss, np.average(l1_list),
                #                                                                 pl, np.average(lpips_list),
                #                                                                 len(ssim_list)))

                # 拼接可视化对比图：原始边缘、掩码边缘、预测边缘、融合边缘
                images_joint = stitch_images(
                    self.postprocess(edge),
                    self.postprocess(inputs),
                    self.postprocess(outputs_edge),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                # 定义结果保存路径（区分4060显卡结果）
                path_masked = os.path.join(self.results_path, self.model_name, 'masked4060')
                path_result = os.path.join(self.results_path, self.model_name, 'result4060')
                path_joint = os.path.join(self.results_path, self.model_name, 'joint4060')

                # 获取当前测试样本名称
                name = self.test_dataset.load_name(index - 1)[:-4] + '.png'

                # 创建保存目录
                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)

                # 处理掩码图像和修复结果图像
                masked_images = self.postprocess(images * (1 - masks) + masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                # 打印保存路径
                print(os.path.join(path_joint, name[:-4] + '.png'))

                # 保存可视化结果
                images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
                imsave(masked_images, os.path.join(path_masked, name))
                imsave(images_result, os.path.join(path_result, name))

                print(name + ' complete!')

            # -------------------------- 2. 测试图像修复模型 --------------------------
            elif model == 2:
                # 掩码后的图像（孔洞区域为0）
                inputs = (images * (1 - masks))

                with torch.no_grad():  # 禁用梯度计算
                    # 推理生成修复图像
                    outputs_img = self.inpaint_model(images, masks, edge, gray_img)
                # 融合修复图像与原始图像：仅修复掩码区域
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                # 计算评估指标（注释）
                psnr, ssim = self.metric(images, outputs_merged)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                if torch.cuda.is_available():
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(),
                                          self.transf(images[0].cpu()).cuda()).item()
                    lpips_list.append(pl)
                else:
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
                    lpips_list.append(pl)

                # 每修完一张图打印一次分数
                print(f"--> 当前图片 PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {pl:.4f}")

                # l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                # l1_list.append(l1_loss)
                #
                # print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                #                                                                 ssim, np.average(ssim_list),
                #                                                                 l1_loss, np.average(l1_list),
                #                                                                 pl, np.average(lpips_list),
                #                                                                 len(ssim_list)))

                # 拼接可视化对比图：原始图像、掩码图像、修复图像、融合图像
                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                # 定义结果保存路径
                path_masked = os.path.join(self.results_path, self.model_name, 'masked4060')
                path_result = os.path.join(self.results_path, self.model_name, 'result4060')
                path_joint = os.path.join(self.results_path, self.model_name, 'joint4060')

                # 获取当前测试样本名称
                name = self.test_dataset.load_name(index - 1)[:-4] + '.png'

                # 创建保存目录
                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)

                # 处理并保存可视化结果
                masked_images = self.postprocess(images * (1 - masks) + masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint, name[:-4] + '.png'))

                images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
                imsave(masked_images, os.path.join(path_masked, name))
                imsave(images_result, os.path.join(path_result, name))

                print(name + ' complete!')

            # -------------------------- 3. 测试边缘引导的图像修复模型 --------------------------
            elif model == 3:
                # 掩码后的图像（孔洞区域为0）
                inputs = (images * (1 - masks))

                with torch.no_grad():  # 禁用梯度计算
                    # 第一步：生成边缘图
                    outputs_edge = self.edge_model(images, masks, edge, gray_img)
                # 融合预测边缘与原始边缘
                outputs_edge = (outputs_edge * masks) + (edge * (1 - masks))

                with torch.no_grad():  # 禁用梯度计算
                    # 第二步：用生成的边缘图修复图像
                    outputs_img = self.inpaint_model(images, masks, outputs_edge, gray_img)
                # 融合修复图像与原始图像
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                # --- 【开启 PSNR, SSIM, 和 LPIPS 算分】 ---
                psnr, ssim = self.metric(images, outputs_merged)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                # 计算 LPIPS
                if torch.cuda.is_available():
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(),
                                          self.transf(images[0].cpu()).cuda()).item()
                    lpips_list.append(pl)
                else:
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()),
                                          self.transf(images[0].cpu())).item()
                    lpips_list.append(pl)

                # 每修完一张图打印一次分数，让您实时看到进度
                print(f"--> 当前图片 PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {pl:.4f}")



                # if torch.cuda.is_available():
                #     pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(),
                #                           self.transf(images[0].cpu()).cuda()).item()
                #     lpips_list.append(pl)
                # else:
                #     pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
                #     lpips_list.append(pl)
                #
                # l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                # l1_list.append(l1_loss)
                #
                # print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                #                                                                 ssim, np.average(ssim_list),
                #                                                                 l1_loss, np.average(l1_list),
                #                                                                 pl, np.average(lpips_list),
                #                                                                 len(ssim_list)))

                # 拼接可视化对比图：原始图像、掩码图像、预测边缘、修复图像、融合图像
                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_edge),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                # 定义结果保存路径
                path_masked = os.path.join(self.results_path, self.model_name, 'masked4060')
                path_result = os.path.join(self.results_path, self.model_name, 'result4060')
                path_joint = os.path.join(self.results_path, self.model_name, 'joint4060')

                # 获取当前测试样本名称
                name = self.test_dataset.load_name(index - 1)[:-4] + '.png'

                # 创建保存目录
                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)

                # 处理并保存可视化结果
                masked_images = self.postprocess(images * (1 - masks) + masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint, name[:-4] + '.png'))

                images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
                imsave(masked_images, os.path.join(path_masked, name))
                imsave(images_result, os.path.join(path_result, name))

                print(name + ' complete!')

        # 导出模型为ONNX格式（注意：此处model变量是整数，会报错，需改为实际模型对象）
        # 正确写法：torch.onnx.export(self.inpaint_model, (images, masks, outputs_edge, gray_img), 'model.onnx')
        # torch.onnx.export(model, images_joint, 'model.onnx')
        # wandb.save('model.onnx')  # 保存ONNX模型到wandb
        print('\nEnd Testing')

        # 打印最终的平均评估指标！(写进比赛 PPT 里的核心数据)
        print('\n=======================================')
        print('【大豆叶片修复系统 - 最终成绩单】')
        print(f'Average PSNR:  {np.average(psnr_list):.4f}  (越大越好)')
        print(f'Average SSIM:  {np.average(ssim_list):.4f}  (越大越好)')
        print(f'Average LPIPS: {np.average(lpips_list):.4f}  (越小越好)')
        print('=======================================\n')

        # 打印平均评估指标（注释的指标计算启用后需取消注释）
        # print('edge_psnr_ave:{} edge_ssim_ave:{} l1_ave:{} lpips:{}'.format(np.average(psnr_list),
        #                                                                      np.average(ssim_list),
        #                                                                      np.average(l1_list),
        #                                                                      np.average(lpips_list)))

    def log(self, logs):
        """
        日志保存函数
        Args:
            logs: 日志列表，格式为[(key1, value1), (key2, value2), ...]
        """
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            # 将日志值拼接为字符串并写入文件
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        """
        批量将数据加载到指定设备（CPU/GPU）
        Args:
            *args: 任意数量的张量
        Returns:
            加载到指定设备的张量生成器
        """
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        """
        图像后处理函数：将模型输出转换为可保存的PIL图像
        步骤：
        1. 将像素值从[0,1]缩放至[0,255]
        2. 将张量维度从[B, C, H, W]转换为[B, H, W, C]（适配PIL格式）
        3. 转换为整数类型
        Args:
            img: 模型输出张量，shape=[B, C, H, W]，取值范围[0,1]
        Returns:
            处理后的张量列表（每个元素为单张图像的PIL格式数据）
        """
        # [0, 1] => [0, 255]
        img = img * 255.0
        # 维度转换：BCHW -> BHWC
        img = img.permute(0, 2, 3, 1)
        # 转换为整数（像素值为整数）
        img = img.int()
        # 转换为PIL图像列表
        img_list = []
        for i in range(img.shape[0]):
            # 转换为numpy数组并调整数据类型
            img_np = img[i].cpu().numpy().astype(np.uint8)
            # 转换为PIL图像
            img_pil = Image.fromarray(img_np)
            img_list.append(img_pil)
        return img_list

    def metric(self, gt, pre):
        """
        计算图像质量评估指标（PSNR和SSIM）
        Args:
            gt: 真实图像张量，shape=[B, C, H, W]，取值范围[0,1]
            pre: 预测图像张量，shape=[B, C, H, W]，取值范围[0,1]
        Returns:
            psnr: 峰值信噪比
            ssim: 结构相似性
        """
        # 处理预测图像：缩放至[0,255]，维度转换，转numpy数组
        pre = pre.clamp_(0, 1) * 255.0  # 限制取值范围并缩放
        pre = pre.permute(0, 2, 3, 1)  # BCHW -> BHWC
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]  # 取第一张图，转numpy

        # 处理真实图像：同上
        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        # 计算PSNR（最大值限制为100）
        psnr = min(100, compare_psnr(gt, pre))
        # 计算SSIM（多通道，数据范围255，通道轴为最后一维）
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255, channel_axis=2)

        return psnr, ssim

    class cal_mean_nme():
        """
        计算平均NME（归一化均方误差）的内部类
        """
        sum = 0  # 累计NME值
        amount = 0  # 样本数量
        mean_nme = 0  # 平均NME

        def __call__(self, nme):
            """
            累加NME值并计算平均值
            Args:
                nme: 当前样本的NME值
            Returns:
                累计平均NME值
            """
            self.sum += nme
            self.amount += 1
            self.mean_nme = self.sum / self.amount
            return self.mean_nme

        def get_mean_nme(self):
            """
            获取平均NME值
            Returns:
                平均NME值
            """
            return self.mean_nme