import torch
import torch.nn as nn
import torchvision.models as models


class AdversarialLoss(nn.Module):
    """
    对抗损失（Adversarial Loss）类
    支持三种常见的GAN损失类型：
    1. nsgan: 标准GAN（Binary Cross Entropy Loss）
    2. lsgan: 最小二乘GAN（Mean Squared Error Loss）
    3. hinge: 铰链损失GAN（ReLU-based Hinge Loss）
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        初始化对抗损失
        Args:
            type: 损失类型 ['nsgan', 'lsgan', 'hinge']
            target_real_label: 真实样本的目标标签值
            target_fake_label: 伪造样本的目标标签值
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        # 注册缓冲区（不参与梯度更新）存储真实/伪造标签
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        # 根据损失类型选择对应的损失函数
        if type == 'nsgan':
            self.criterion = nn.BCELoss()  # 二分类交叉熵损失

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()  # 均方误差损失

        elif type == 'hinge':
            self.criterion = nn.ReLU()  # ReLU激活（hinge损失核心）

    def __call__(self, outputs, is_real, is_disc=None):
        """
        计算对抗损失
        Args:
            outputs: 判别器的输出值
            is_real: 是否为真实样本（True=真实，False=伪造）
            is_disc: 是否为判别器损失计算（仅hinge损失需要）
        Returns:
            loss: 计算得到的对抗损失值
        """
        # Hinge损失计算逻辑（GAN训练更稳定）
        if self.type == 'hinge':
            if is_disc:  # 计算判别器损失
                if is_real:  # 真实样本：判别器应输出大值，损失为max(0, 1 - output)
                    outputs = -outputs
                # hinge损失公式：E[max(0, 1 ± D(x))]
                return self.criterion(1 + outputs).mean()
            else:  # 计算生成器损失：E[-D(G(z))]
                return (-outputs).mean()

        # NSGAN/LSGAN损失计算逻辑
        else:
            # 扩展标签维度以匹配输出维度
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            # 计算分类/回归损失
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    """
    风格损失（Style Loss）类
    基于VGG19提取的特征图计算Gram矩阵，衡量生成图像与真实图像的风格相似度
    Gram矩阵：捕获特征图之间的相关性，表征图像的风格特征
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        # 添加VGG19特征提取器（冻结参数，仅用于特征提取）
        self.add_module('vgg', VGG19())
        # L1损失作为Gram矩阵的距离度量（比L2更鲁棒）
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        """
        计算特征图的Gram矩阵
        Args:
            x: 特征图 [B, C, H, W]
        Returns:
            G: Gram矩阵 [B, C, C]
        """
        b, ch, h, w = x.size()
        # 展平特征图：[B, C, H*W]
        f = x.view(b, ch, w * h)
        # 转置：[B, H*W, C]
        f_T = f.transpose(1, 2)
        # 计算Gram矩阵并归一化（除以特征图元素总数）
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        """
        计算风格损失
        Args:
            x: 生成图像 [B, 3, H, W]
            y: 真实图像 [B, 3, H, W]
        Returns:
            style_loss: 风格损失值（多尺度Gram矩阵的L1距离之和）
        """
        # 提取VGG19特征
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # 计算多尺度风格损失（选择关键relu层的特征）
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Module):
    """
    感知损失（Perceptual Loss）类
    基于VGG19提取的特征图直接计算L1距离，衡量生成图像与真实图像的内容相似度
    相比像素级损失（L1/L2），更关注高层语义特征的一致性
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        """
        初始化感知损失
        Args:
            weights: 各层特征损失的权重 [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
        """
        super(PerceptualLoss, self).__init__()
        # 添加VGG19特征提取器
        self.add_module('vgg', VGG19())
        # L1损失作为特征距离度量
        self.criterion = torch.nn.L1Loss()
        # 各层权重（可调整以侧重不同尺度的特征）
        self.weights = weights

    def __call__(self, x, y):
        """
        计算感知损失
        Args:
            x: 生成图像 [B, 3, H, W]
            y: 真实图像 [B, 3, H, W]
        Returns:
            content_loss: 感知损失值（多尺度特征的加权L1距离之和）
        """
        # 提取VGG19特征
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # 计算多尺度感知损失（选择各层第一个relu的特征）
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])  # 浅层特征（细节）
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])  # 深层特征（语义）

        return content_loss


class VGG19(torch.nn.Module):
    """
    VGG19特征提取器
    截取预训练VGG19的特征提取部分，输出各关键relu层的特征图
    冻结所有参数，仅用于前向特征提取
    """

    def __init__(self):
        super(VGG19, self).__init__()
        # 加载预训练VGG19的特征提取部分
        features = models.vgg19(pretrained=True).features
        # 定义各层特征提取器（对应VGG19的relu层）
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        # 逐层构建各relu特征提取器（对应VGG19的层结构）
        # relu1_1: 前2层（conv1_1 + relu）
        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        # relu1_2: 2-4层（conv1_2 + relu）
        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        # relu2_1: 4-7层（pool1 + conv2_1 + relu）
        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        # relu2_2: 7-9层（conv2_2 + relu）
        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        # relu3_1: 9-12层（pool2 + conv3_1 + relu）
        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        # relu3_2: 12-14层（conv3_2 + relu）
        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        # relu3_3: 14-16层（conv3_3 + relu）
        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])  # 注：原代码笔误，应为self.relu3_3

        # relu3_4: 16-18层（conv3_4 + relu）
        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        # relu4_1: 18-21层（pool3 + conv4_1 + relu）
        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        # relu4_2: 21-23层（conv4_2 + relu）
        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        # relu4_3: 23-25层（conv4_3 + relu）
        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        # relu4_4: 25-27层（conv4_4 + relu）
        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        # relu5_1: 27-30层（pool4 + conv5_1 + relu）
        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        # relu5_2: 30-32层（conv5_2 + relu）
        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        # relu5_3: 32-34层（conv5_3 + relu）
        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        # relu5_4: 34-36层（conv5_4 + relu）
        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # 冻结所有参数（仅用于特征提取，不参与训练）
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        前向传播提取各层特征
        Args:
            x: 输入图像 [B, 3, H, W]（需归一化到ImageNet均值/方差）
        Returns:
            out: 字典，包含各relu层的特征图
        """
        # 逐层前向传播，提取各层特征
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)  # 注：原代码未定义relu3_3，需确认
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        # 整理特征输出
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


# ==================== SSIM（结构相似性指数）评估指标 ====================
import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    """
    生成一维高斯分布向量
    Args:
        window_size: 高斯核尺寸（奇数）
        sigma: 高斯核标准差
    Returns:
        gauss: 归一化的一维高斯向量 [window_size]
    """
    # 高斯函数：G(x) = exp(-(x - μ)² / (2σ²))，μ=window_size//2
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()  # 归一化


def create_window(window_size, channel=1):
    """
    创建二维高斯核（用于SSIM的局部加权平均）
    Args:
        window_size: 高斯核尺寸
        channel: 通道数（扩展为多通道高斯核）
    Returns:
        window: 归一化的二维高斯核 [C, 1, window_size, window_size]
    """
    # 生成一维高斯核并扩展为二维
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 扩展到指定通道数
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    """
    计算结构相似性指数（SSIM）
    SSIM公式：SSIM(x,y) = [(2μxμy + C1)(2σxy + C2)] / [(μx² + μy² + C1)(σx² + σy² + C2)]
    其中：
    - μx/μy: 局部均值（高斯加权）
    - σx²/σy²: 局部方差
    - σxy: 局部协方差
    - C1/C2: 稳定常数，避免分母为0
    Args:
        img1: 图像1 [B, C, H, W]
        img2: 图像2 [B, C, H, W]
        window_size: 高斯核尺寸
        window: 预生成的高斯核（避免重复计算）
        size_average: 是否对所有像素的SSIM取平均
        full: 是否返回完整结果（SSIM + 对比度灵敏度）
        val_range: 像素值范围（None则自动推断）
    Returns:
        ret: SSIM值（均值或逐像素值）
        cs: 对比度灵敏度（仅full=True时返回）
    """
    # 自动推断像素值范围
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val  # 像素值动态范围
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()

    # 生成/加载高斯核
    if window is None:
        real_size = min(window_size, height, width)  # 适配小尺寸图像
        window = create_window(real_size, channel=channel).to(img1.device)

    # 计算局部均值（高斯卷积）
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    # 计算均值平方和均值乘积
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差（利用公式：Var(X)=E[X²]-E[X]², Cov(X,Y)=E[XY]-E[X]E[Y]）
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # 稳定常数（经验值）
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 计算对比度灵敏度（CS）和SSIM
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # 对比度灵敏度

    # SSIM核心计算
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # 计算最终SSIM值
    if size_average:
        ret = ssim_map.mean()  # 全局均值
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)  # 逐样本均值

    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    """
    SSIM评估类（封装为nn.Module，便于集成到训练流程）
    可复用高斯核，避免每次计算都重新生成
    """

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # 初始化单通道高斯核
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        """
        前向计算SSIM
        Args:
            img1: 图像1 [B, C, H, W]
            img2: 图像2 [B, C, H, W]
        Returns:
            ssim_val: SSIM值
        """
        (_, channel, _, _) = img1.size()

        # 适配通道数和数据类型
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            # 重新生成适配当前通道/类型的高斯核
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        # 调用SSIM计算函数
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)