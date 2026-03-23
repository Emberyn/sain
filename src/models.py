import decimal
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 自定义网络模块：边缘生成器、图像修复生成器、判别器
from .networks import EdgeGenerator, InpaintGenerator, Discriminator
# 自定义损失函数：对抗损失、感知损失、风格损失、SSIM损失
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, SSIM
# 学习率调度器：指数衰减
from torch.optim.lr_scheduler import ExponentialLR


class BaseModel(nn.Module):
    """
    基础模型类（抽象类）
    提供所有模型通用的功能：
    1. 模型权重的加载/保存（普通保存、最佳模型保存）
    2. 迭代次数、训练轮数的管理
    3. 权重保存路径的初始化
    """
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name  # 模型名称（EdgeModel/InpaintingModel）
        self.config = config  # 配置类对象（包含训练参数、路径、设备等）
        self.iteration = 0  # 训练迭代次数
        self.epoch = 0  # 训练轮数

        # 定义模型权重保存路径
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')  # 生成器权重路径
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')  # 判别器权重路径

        # 定义最佳模型权重保存路径（按验证集性能）
        self.best_gen_weights_path = os.path.join(config.PATH, name + '_gen_best.pth')
        self.best_dis_weights_path = os.path.join(config.PATH, name + '_dis_best.pth')

    def load(self):
        """
        加载预训练模型权重
        1. 优先加载生成器权重（训练/测试都需要）
        2. 训练模式下额外加载判别器权重
        """
        # 加载生成器权重
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            # 根据设备选择加载方式（GPU/CPU）
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                # CPU加载GPU保存的模型
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            # 加载生成器参数（strict=False：允许参数不匹配，适配不同版本模型）
            self.generator.load_state_dict(data['generator'], strict=False)
            # 恢复迭代次数和训练轮数
            self.iteration = data['iteration']
            self.epoch = data['epoch']

        # 训练模式下加载判别器权重
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            # 加载判别器参数
            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        """
        保存当前模型权重
        1. 保存生成器（包含迭代次数、训练轮数）
        2. 保存判别器
        """
        print(r'\nsaving %s...\n' % self.name)
        # 保存生成器
        torch.save({
            'iteration': self.iteration,
            'epoch': self.epoch,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        # 保存判别器
        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

    def save_best(self):  # 保存最佳模型参数（按验证集性能）
        """
        保存最佳模型权重（通常用于验证集性能最优时）
        """
        print(r'\nsaving best %s...\n' % self.name)
        # 保存最佳生成器
        torch.save({
            'iteration': self.iteration,
            'epoch': self.epoch,
            'generator': self.generator.state_dict()
        }, self.best_gen_weights_path)

        # 保存最佳判别器
        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.best_dis_weights_path)


class EdgeModel(BaseModel):
    """
    边缘生成模型类
    继承BaseModel，实现边缘生成的前向传播、损失计算、反向传播
    """
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # 初始化生成器和判别器
        generator = EdgeGenerator()  # 边缘生成器网络
        # 判别器：输入通道数=3，use_sigmoid根据GAN损失类型决定（hinge loss不需要sigmoid）
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        # 多GPU训练：DataParallel包装模型
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        # 初始化损失函数
        l1_loss = nn.L1Loss()  # L1损失（像素级损失）
        perceptual_loss = PerceptualLoss()  # 感知损失（基于VGG特征）
        style_loss = StyleLoss()  # 风格损失（Gram矩阵匹配）
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)  # 对抗损失（GAN Loss）
        ssim_loss = SSIM()  # SSIM损失（结构相似性）

        # 将模型和损失函数注册为模块（便于参数管理、设备迁移）
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('ssim_loss', ssim_loss)

        # 生成器优化器（Adam）
        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),  # 学习率
            betas=(config.BETA1, config.BETA2)  # Adam的beta参数
        )
        # self.gen_scheduler = ExponentialLR(self.gen_optimizer, gamma=0.98)  # 学习率指数衰减（注释）

        # 判别器优化器（学习率通常是生成器的D2G_LR倍）
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),  # 判别器学习率 = 生成器学习率 * D2G_LR
            betas=(config.BETA1, config.BETA2)
        )
        # self.dis_scheduler = ExponentialLR(self.dis_optimizer, gamma=0.98)  # 学习率指数衰减（注释）

    def process(self, images, masks, edge, gray_img):
        """
        单次迭代的训练过程（核心函数）
        Args:
            images: 原始RGB图像，shape=[B, 3, H, W]
            masks: 掩码图像（1=孔洞，0=背景），shape=[B, 3, H, W]
            edge: 真实边缘图，shape=[B, 3, H, W]
            gray_img: 灰度图像，shape=[B, 1, H, W]
        Returns:
            outputs_edge: 生成的边缘图，shape=[B, 3, H, W]
            gen_loss: 生成器总损失
            dis_loss: 判别器总损失
            logs: 损失日志（用于打印/保存）
            各类细分损失（用于监控）
        """
        self.iteration += 1  # 迭代次数+1

        # 清空优化器梯度
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # 前向传播：生成边缘图
        outputs_edge = self(images, masks, edge, gray_img)

        # 初始化损失
        gen_loss = 0
        dis_loss = 0

        # -------------------------- 判别器损失计算 --------------------------
        dis_input_real = edge  # 真实边缘图（判别器正样本）
        dis_input_fake = outputs_edge.detach()  # 生成的边缘图（detach：切断梯度，避免更新生成器）

        # 判别器前向传播：输出预测值和中间特征
        dis_real, dis_real_feat = self.discriminator(dis_input_real)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)

        # 计算判别器损失：区分真实/生成边缘图
        dis_real_loss = self.adversarial_loss(dis_real, True, True)  # 真实样本：期望判别为真
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)  # 生成样本：期望判别为假
        dis_loss += (dis_real_loss + dis_fake_loss) / 2  # 判别器总损失

        # -------------------------- 生成器损失计算 --------------------------
        # 生成器对抗损失：期望生成的边缘图被判别为真
        gen_input_fake = outputs_edge
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # L1损失：像素级匹配（除以掩码均值，平衡不同掩码大小的损失）
        gen_l1_loss = self.l1_loss(outputs_edge, edge) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # 感知损失：基于VGG特征的高层语义匹配
        gen_content_loss = self.perceptual_loss(outputs_edge, edge)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # SSIM损失：结构相似性损失（1-SSIM，越小表示结构越相似）
        ssim_loss = 1 - self.ssim_loss(outputs_edge, edge)
        ssim_loss = ssim_loss * self.config.SSIM_LOSS_WEIGHT
        gen_loss += ssim_loss

        # 特征匹配损失（注释，如需启用需取消注释）
        gen_fm_loss = 0
        # for i in range(len(dis_real_feat)):
        #     gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        # gen_fm_loss = gen_fm_loss * self.config.FEATURE_MATCHING_LOSS_WEIGHT
        # gen_loss += gen_fm_loss

        # 风格损失：仅计算掩码区域（孔洞）的风格匹配
        gen_style_loss = self.style_loss(outputs_edge * masks, edge * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #############################

        # 构建损失日志（用于打印/保存）
        logs = [
            ("gLoss", gen_loss.item()),  # 生成器总损失
            ("dLoss", dis_loss.item())   # 判别器总损失
        ]

        # 返回生成结果和各类损失
        return outputs_edge, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, ssim_loss, gen_fm_loss

    def forward(self, images, masks, edge, gray_img):
        """
        前向传播：生成边缘图
        Args:
            images: 原始RGB图像，shape=[B, 3, H, W]
            masks: 掩码图像，shape=[B, 3, H, W]
            edge: 真实边缘图，shape=[B, 3, H, W]
            gray_img: 灰度图像，shape=[B, 1, H, W]
        Returns:
            outputs_edge: 生成的边缘图，shape=[B, 3, H, W]
        """
        # 对RGB图像进行掩码操作：孔洞区域设为1（白色），保留非孔洞区域
        images_masked = (images * (1 - masks).float()) + masks
        # 掩码转换为单通道（灰度掩码）
        masks_gray = masks.mean(dim=1, keepdim=True)
        # 对灰度图像进行掩码操作
        gray_img_mask = (gray_img * (1 - masks_gray).float()) + masks_gray
        # 对边缘图进行掩码操作：孔洞区域设为1，保留非孔洞区域
        edge_masked = (edge * (1 - masks).float()) + masks

        # 生成器输入：掩码后的RGB图像
        inputs = images_masked
        # 生成不同尺度的掩码（用于多尺度特征融合）
        scaled_masks_tiny = F.interpolate(masks, size=[int(masks.shape[2] / 8), int(masks.shape[3] / 8)],
                                          mode='nearest')  # 1/8尺度掩码
        scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                             mode='nearest')  # 1/4尺度掩码
        scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                          mode='nearest')  # 1/2尺度掩码

        # 生成器前向传播：生成边缘图
        outputs_edge = self.generator(inputs, masks, edge_masked, gray_img_mask, scaled_masks_half, scaled_masks_quarter,
                                     scaled_masks_tiny)  # 输出shape=[B,3,256,256]
        return outputs_edge

    def backward(self, gen_loss=None, dis_loss=None):
        """
        反向传播：更新生成器和判别器参数
        Args:
            gen_loss: 生成器总损失
            dis_loss: 判别器总损失
        """
        # 判别器反向传播（retain_graph=True：保留计算图，用于后续生成器反向传播）
        dis_loss.backward(retain_graph=True)
        # 生成器反向传播
        gen_loss.backward()
        # 更新判别器参数
        self.dis_optimizer.step()

        # 打印当前生成器学习率（便于监控学习率变化）
        print("gen 学习率：", decimal.Decimal(self.gen_optimizer.state_dict()['param_groups'][0]['lr']))

    # def updataLr(self):
    #     self.dis_scheduler.step()  # 更新判别器学习率
    #     self.gen_scheduler.step()  # 更新生成器学习率

    def backward_joint(self, gen_loss=None, dis_loss=None):
        """
        联合反向传播（备用函数，与backward的区别：不保留计算图）
        """
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModel(BaseModel):
    """
    图像修复模型类
    继承BaseModel，实现图像修复的前向传播、损失计算、反向传播
    """
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # 初始化生成器和判别器
        generator = InpaintGenerator()  # 图像修复生成器网络
        # 判别器：输入通道数=3，use_sigmoid根据GAN损失类型决定
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        # 多GPU训练
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        # 初始化损失函数（无SSIM损失）
        l1_loss = nn.L1Loss()  # L1损失
        perceptual_loss = PerceptualLoss()  # 感知损失
        style_loss = StyleLoss()  # 风格损失
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)  # 对抗损失

        # 注册模块
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # 生成器优化器
        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        #self.gen_scheduler = ExponentialLR(self.gen_optimizer, gamma=0.98)  # 学习率调度器（注释）

        # 判别器优化器
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
        #self.dis_scheduler = ExponentialLR(self.dis_optimizer, gamma=0.98)  # 学习率调度器（注释）

    def process(self, images, masks, edge, gray_img):
        """
        单次迭代的训练过程（核心函数）
        Args:
            images: 原始RGB图像，shape=[B, 3, H, W]
            masks: 掩码图像（1=孔洞，0=背景），shape=[B, 3, H, W]
            edge: 边缘图（真实/生成），shape=[B, 3, H, W]
            gray_img: 灰度图像，shape=[B, 1, H, W]
        Returns:
            outputs_img: 修复后的图像，shape=[B, 3, H, W]
            gen_loss: 生成器总损失
            dis_loss: 判别器总损失
            logs: 损失日志
            各类细分损失
        """
        self.iteration += 1  # 迭代次数+1

        # 清空优化器梯度
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # 前向传播：生成修复图像
        outputs_img = self(images, masks, edge, gray_img)

        # 初始化损失
        gen_loss = 0
        dis_loss = 0

        # -------------------------- 判别器损失计算 --------------------------
        dis_input_real = images  # 真实图像（正样本）
        dis_input_fake = outputs_img.detach()  # 修复图像（负样本，detach切断梯度）

        # 判别器前向传播（仅保留预测值，忽略中间特征）
        dis_real, _ = self.discriminator(dis_input_real)
        dis_fake, _ = self.discriminator(dis_input_fake)

        # 计算判别器损失
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # -------------------------- 生成器损失计算 --------------------------
        # 生成器对抗损失
        gen_input_fake = outputs_img
        gen_fake, _ = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # L1损失：像素级匹配（除以掩码均值）
        gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # 感知损失：高层语义匹配
        gen_content_loss = self.perceptual_loss(outputs_img, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # 风格损失：仅计算掩码区域的风格匹配
        gen_style_loss = self.style_loss(outputs_img * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #############################

        # 构建损失日志
        logs = [
            ("gLoss",gen_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        # 返回修复结果和各类损失
        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss

    def forward(self, images, masks, edge, gray_img):
        """
        前向传播：生成修复图像
        Args:
            images: 原始RGB图像，shape=[B, 3, H, W]
            masks: 掩码图像，shape=[B, 3, H, W]
            edge: 边缘图（真实/生成），shape=[B, 3, H, W]
            gray_img: 灰度图像，shape=[B, 1, H, W]
        Returns:
            outputs_img: 修复后的图像，shape=[B, 3, H, W]
        """
        # 对RGB图像进行掩码操作：孔洞区域设为1
        images_masked = (images * (1 - masks).float()) + masks
        # 掩码转换为单通道
        masks_gray = masks.mean(dim=1, keepdim=True)
        # 对灰度图像进行掩码操作
        gray_img_mask = (gray_img * (1 - masks_gray).float()) + masks_gray

        # 生成器输入：掩码后的RGB图像
        inputs = images_masked
        # 生成不同尺度的掩码
        scaled_masks_tiny = F.interpolate(masks, size=[int(masks.shape[2] / 8), int(masks.shape[3] / 8)],
                                     mode='nearest')  # 1/8尺度
        scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                     mode='nearest')  # 1/4尺度
        scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                          mode='nearest')  # 1/2尺度

        # 生成器前向传播：生成修复图像
        outputs_img = self.generator(inputs,masks, edge, gray_img_mask,scaled_masks_half,scaled_masks_quarter,scaled_masks_tiny)
        return outputs_img

    def backward(self, gen_loss = None, dis_loss = None):
        """
        反向传播：更新生成器和判别器参数
        """
        # 判别器反向传播（保留计算图）
        dis_loss.backward(retain_graph= True)
        # 生成器反向传播
        gen_loss.backward()
        # 更新判别器参数
        self.dis_optimizer.step()
        # 更新生成器参数
        self.gen_optimizer.step()

        # 打印生成器学习率
        print("gen 学习率：", decimal.Decimal(self.gen_optimizer.state_dict()['param_groups'][0]['lr']))

    # def updataLr(self):
    #     self.dis_scheduler.step()  # 更新判别器学习率
    #     self.gen_scheduler.step()  # 更新生成器学习率

    def backward_joint(self, gen_loss = None, dis_loss = None):
        """
        联合反向传播（备用函数）
        """
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


def abs_smooth(x):
    """
    平滑绝对值函数（备用函数，未在主流程中使用）
    用于缓解梯度消失问题
    Args:
        x: 输入张量
    Returns:
        平滑后的绝对值张量
    """
    absx = torch.abs(x)
    minx = torch.min(absx, other=torch.ones(absx.shape).cuda())  # 限制最小值为1
    r = 0.5 * ((absx - 1) * minx + absx)  # 平滑计算
    return r