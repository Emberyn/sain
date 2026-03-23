import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx  # 调试用，打断点
import numbers  # 用于判断数据类型是否为数字
from einops import rearrange  # 灵活的张量维度重排工具
import os
import cv2
import numpy as np
from sympy import false  # 未使用，可删除


class BaseNetwork(nn.Module):
    """
    所有网络的基类，提供权重初始化的通用方法
    继承自nn.Module，是PyTorch中所有神经网络模块的基类
    """

    def __init__(self):
        super(BaseNetwork, self).__init__()

    # 初始化权重函数
    def init_weights(self, init_type='normal', gain=0.02):
        """
        对网络中的卷积层、线性层、BatchNorm层进行权重初始化
        Args:
            init_type: 初始化方式，可选 'normal'/'xavier'/'kaiming'/'orthogonal'
            gain: 初始化的增益系数，控制权重的标准差
        """

        def init_func(m):
            # 获取模块的类名（如Conv2d、Linear、BatchNorm2d）
            classname = m.__class__.__name__
            # 对卷积层/线性层的权重进行初始化
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    # 正态分布初始化: N(0, gain^2)
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    # Xavier初始化: 适用于tanh/sigmoid激活，保持前向/反向传播的方差一致
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    # Kaiming初始化: 适用于ReLU激活，解决梯度消失问题
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    # 正交初始化: 保持特征空间的正交性，提升训练稳定性
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                # 偏置初始化为0（如果存在偏置）
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            # 对BatchNorm2d层的权重和偏置初始化
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)  # 权重初始化为1附近
                nn.init.constant_(m.bias.data, 0.0)  # 偏置初始化为0

        # 将初始化函数应用到当前网络的所有子模块
        self.apply(init_func)


# 进行谱归一化操作
def spectral_norm(module, mode=True):
    """
    对模块进行谱归一化（Spectral Normalization）
    作用：限制权重矩阵的谱范数，防止梯度爆炸，提升GAN训练稳定性
    Args:
        module: 需要归一化的模块（如Conv2d）
        mode: 是否启用谱归一化
    Returns:
        归一化后的模块（或原模块，如果mode=False）
    """
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Discriminator(BaseNetwork):
    """
    基于PatchGAN的判别器网络（用于GAN任务）
    输出为每个patch的真假概率，同时返回各层特征用于特征匹配损失
    继承自BaseNetwork，使用其权重初始化方法
    """

    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        """
        Args:
            in_channels: 输入图像的通道数（如RGB图为3）
            use_sigmoid: 是否对输出应用sigmoid激活（将输出映射到[0,1]）
            use_spectral_norm: 是否对卷积层使用谱归一化
            init_weights: 是否初始化权重
        """
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        # 卷积层1: 下采样，通道数->64，步长2，特征图尺寸减半
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64,
                                    kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm),  # 谱归一化时通常禁用偏置
                          use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),  # 负斜率0.2，inplace=True节省内存
        )

        # 卷积层2: 下采样，通道数->128，步长2，特征图尺寸减半
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128,
                                    kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 卷积层3: 下采样，通道数->256，步长2，特征图尺寸减半
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256,
                                    kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 卷积层4: 不改变尺寸，通道数->512，步长1
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512,
                                    kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 卷积层5: 输出层，通道数->1（真假概率），步长1
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1,
                                    kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm),
                          use_spectral_norm),
        )

        # 初始化权重
        if init_weights:
            self.init_weights()

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，shape=[B, C, H, W]
        Returns:
            outputs: 每个patch的真假概率，shape=[B, 1, H', W']
            各层特征列表: [conv1, conv2, conv3, conv4, conv5]
        """
        conv1 = self.conv1(x)  # [B,64,H/2,W/2]
        conv2 = self.conv2(conv1)  # [B,128,H/4,W/4]
        conv3 = self.conv3(conv2)  # [B,256,H/8,W/8]
        conv4 = self.conv4(conv3)  # [B,512,H/8,W/8]
        conv5 = self.conv5(conv4)  # [B,1,H/8,W/8]

        outputs = conv5
        # 应用sigmoid将输出映射到[0,1]（判别器输出概率）
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        # 返回最终输出和各层特征（用于特征匹配损失）
        return outputs, [conv1, conv2, conv3, conv4, conv5]


############# 辅助函数：张量维度转换 #############
def to_3d(x):
    """
    将4维张量 (B, C, H, W) 转换为3维张量 (B, H*W, C)
    作用：适配LayerNorm的维度要求（对最后一维归一化）
    Args:
        x: 4维张量 [B, C, H, W]
    Returns:
        3维张量 [B, H*W, C]
    """
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """
    将3维张量 (B, H*W, C) 还原为4维张量 (B, C, H, W)
    Args:
        x: 3维张量 [B, H*W, C]
        h: 特征图高度
        w: 特征图宽度
    Returns:
        4维张量 [B, C, H, W]
    """
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# 定义无偏置的Layer Normalization（层归一化）
class BiasFree_LayerNorm(nn.Module):
    """
    无偏置的LayerNorm实现
    与PyTorch原生LayerNorm的区别：仅包含权重缩放，无偏置项
    适用于对归一化稳定性要求高的场景
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # 处理输入类型：如果是整数，转换为元组（如48 -> (48,)）
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 确保归一化维度为1（适配3维张量 [B, H*W, C]）
        assert len(normalized_shape) == 1

        # 可学习的权重参数（初始化为1）
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        前向传播：计算方差→归一化→权重缩放
        Args:
            x: 3维张量 [B, H*W, C]
        Returns:
            归一化后的张量 [B, H*W, C]
        """
        # 计算最后一维的方差（unbiased=False: 使用样本方差，不除以N-1）
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化：x / sqrt(方差 + 小常数)，防止除0；然后乘以权重缩放
        return x / torch.sqrt(sigma + 1e-5) * self.weight


# 定义带偏置的Layer Normalization（层归一化）
class WithBias_LayerNorm(nn.Module):
    """
    带偏置的LayerNorm实现（与PyTorch原生LayerNorm一致）
    包含权重缩放和偏置偏移，适配大多数场景
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        # 可学习参数：权重（初始1）、偏置（初始0）
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        前向传播：计算均值/方差→归一化→权重缩放+偏置偏移
        Args:
            x: 3维张量 [B, H*W, C]
        Returns:
            归一化后的张量 [B, H*W, C]
        """
        # 计算最后一维的均值
        mu = x.mean(-1, keepdim=True)
        # 计算最后一维的方差
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化公式：(x - 均值) / sqrt(方差 + 小常数) * 权重 + 偏置
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# 层归一化封装类（适配4维特征图）
class LayerNorm(nn.Module):
    """
    适配4维张量 [B, C, H, W] 的LayerNorm封装
    先将4维转3维，应用LayerNorm，再转回4维
    """

    def __init__(self, dim, LayerNorm_type):
        """
        Args:
            dim: 归一化的维度（特征通道数C）
            LayerNorm_type: 归一化类型，'BiasFree' 或 'WithBias'
        """
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 4维张量 [B, C, H, W]
        Returns:
            归一化后的4维张量 [B, C, H, W]
        """
        h, w = x.shape[-2:]
        # 4维→3维→归一化→3维→4维
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Feed-Forward Network (FFN)  ----来源于Restormer的Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    """
    门控深度卷积前馈网络（GDFN）
    结构：1x1卷积升维 → 3x3深度卷积 → 门控机制 → 1x1卷积降维
    作用：在Transformer块中对特征进行非线性变换
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        """
        Args:
            dim: 输入特征通道数
            ffn_expansion_factor: 通道扩展因子（控制隐藏层维度）
            bias: 卷积层是否使用偏置
        """
        super(FeedForward, self).__init__()

        # 计算隐藏层特征维度 = 输入维度 × 扩展因子
        hidden_features = int(dim * ffn_expansion_factor)

        # 1x1卷积：升维到2×hidden_features（为后续分块做准备）
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 3x3深度可分离卷积：在每个通道内部做卷积，不跨通道交互
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)  # groups=通道数 → 深度卷积

        # 1x1卷积：降维回原始维度dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        前向传播：升维 → 深度卷积 → 分块门控 → 降维
        Args:
            x: 输入张量 [B, dim, H, W]
        Returns:
            输出张量 [B, dim, H, W]
        """
        # 升维: [B, dim, H, W] → [B, 2*hidden, H, W]
        x = self.project_in(x)

        # 深度卷积后按通道分块：x1, x2 → 各 [B, hidden, H, W]
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        # 门控机制：GELU激活x1 × x2（增强特征表达的非线性）
        x = F.gelu(x1) * x2
        # 降维回原始维度: [B, hidden, H, W] → [B, dim, H, W]
        x = self.project_out(x)
        return x


##########################################################################
## Squeeze-and-Channel Attention Layer (SCAL)
class Attention(nn.Module):
    """
    挤压-通道注意力层（SCAL）
    结合通道注意力（Channel branch）和空间注意力（Spatial branch）
    通道分支：多头自注意力；空间分支：下采样+卷积+上采样；最终逐元素相乘融合
    """

    def __init__(self, dim, num_heads, bias):
        """
        Args:
            dim: 输入特征通道数
            num_heads: 注意力头数（多头注意力）
            bias: 卷积层是否使用偏置
        """
        #### Channel branch（通道分支：多头自注意力）
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 温度系数（可学习）：控制注意力分布的锐度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1卷积：生成Q/K/V（查询/键/值），通道数→3×dim
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 3x3深度卷积：增强Q/K/V的空间特征
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 1x1卷积：注意力输出投影回原维度
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        #### Spatial branch（空间分支：下采样+卷积+上采样）
        # 平均池化下采样：尺寸减半
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积块：提取空间特征
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),  # 层归一化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        # 上采样：尺寸还原
        self.upsample = nn.Upsample(scale_factor=2)
        ##########

    def forward(self, x):
        """
        前向传播：通道注意力 + 空间注意力 逐元素融合
        Args:
            x: 输入张量 [B, dim, H, W]
        Returns:
            融合后的特征张量 [B, dim, H, W]
        """
        #### Channel branch（通道分支）
        b, c, h, w = x.shape

        # 生成Q/K/V：1x1卷积→3x3深度卷积
        qkv = self.qkv_dwconv(self.qkv(x))
        # 按通道分块：Q/K/V → 各 [B, dim, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # 维度重排：适配多头注意力 → [B, num_heads, dim/num_heads, H*W]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化Q/K：L2归一化（dim=-1表示对最后一维归一化）
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力分数：Q @ K^T × 温度系数 → [B, num_heads, H*W, H*W]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 注意力分数softmax归一化（每行和为1）
        attn = attn.softmax(dim=-1)

        # 注意力加权求和：attn @ V → [B, num_heads, dim/num_heads, H*W]
        out = (attn @ v)

        # 维度重排还原：→ [B, dim, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        #### Spatial branch（空间分支）
        # 下采样→卷积提取空间特征→上采样还原尺寸
        y = self.avg_pool(x)  # [B, dim, H/2, W/2]
        y = self.conv(y)  # [B, dim, H/2, W/2]
        y = self.upsample(y)  # [B, dim, H, W]
        # 特征融合：通道注意力输出 × 空间注意力输出（逐元素相乘）
        out = y * out
        ###########

        # 投影回原维度并输出
        out = self.project_out(out)
        return out


########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
# 来源于Restormer，增加门控机制和mask感知（Mask-aware）
class Attention_MDTA(nn.Module):
    """
    多深度卷积头转置自注意力（MDTA）
    改进点：
    1. 增加门控机制（Gated）：增强特征选择能力
    2. 增加Mask感知：将mask信息融入注意力计算
    """

    def __init__(self, dim, num_heads, bias):
        super(Attention_MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 生成Q/K/V
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 深度卷积增强Q/K/V的空间特征
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 注意力输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 门控机制：1x1卷积 + GELU激活
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.GELU()
        )

        # [New] Mask 投影层：将 1通道mask映射到特征维度dim
        self.mask_proj = nn.Conv2d(1, dim, kernel_size=1, bias=False)

    def forward(self, x, mask=None):
        """
        前向传播（支持mask输入）
        Args:
            x: 输入特征 [B, dim, H, W]
            mask: 掩码张量 [B, C_mask, H_mask, W_mask]（可选）
        Returns:
            注意力输出 [B, dim, H, W]
        """
        b, c, h, w = x.shape

        # 生成Q/K/V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # 计算门控权重
        g = self.gate(x)

        # [New] 处理Mask（如果提供）
        if mask is not None:
            # 确保mask是单通道（多通道则取均值）
            if mask.shape[1] > 1:
                mask_in = mask.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            else:
                mask_in = mask

            # 确保mask尺寸与特征图一致（不一致则插值）
            if mask_in.shape[2:] != (h, w):
                mask_in = F.interpolate(mask_in, size=(h, w), mode='nearest')

            # Mask特征投影：1通道→dim通道
            mask_feat = self.mask_proj(mask_in)  # (B, dim, H, W)
            # 将mask信息融入Key：K = K + Mask特征（引导注意力关注非掩码区域）
            k = k + mask_feat

        # 维度重排适配多头注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Q/K归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力分数（标准自注意力公式）
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 注意力加权求和
        out = (attn @ v)
        # 维度重排还原
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 应用门控：注意力输出 × 门控权重
        out = out * g
        # 投影输出
        out = self.project_out(out)
        return out


##########################################################################
########### Sandwich Block（三明治块）
class SandwichBlock(nn.Module):
    """
    三明治结构的Transformer块：FFN → Attention → FFN
    每个子模块前都有LayerNorm，且使用残差连接
    输入输出通道数均为dim，保持维度一致
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        """
        Args:
            dim: 输入/输出通道数
            num_heads: 注意力头数
            ffn_expansion_factor: FFN通道扩展因子
            bias: 卷积层是否使用偏置
            LayerNorm_type: 层归一化类型（'BiasFree'/'WithBias'）
        """
        super(SandwichBlock, self).__init__()

        # 第一层归一化 + FFN（残差连接）
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        # 注意力前的归一化
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)

        # 第二层FFN前的归一化
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        """
        前向传播：残差连接 + 归一化 + 子模块
        Args:
            x: 输入张量 [B, dim, H, W]
        Returns:
            输出张量 [B, dim, H, W]
        """
        # 残差1：x + FFN1(LayerNorm(x))
        x = x + self.ffn1(self.norm1_1(x))
        # 残差2：x + Attention(LayerNorm(x))
        x = x + self.attn(self.norm1(x))
        # 残差3：x + FFN(LayerNorm(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Transformer Block  用于替代Sandwich Block
class TransformerBlock(nn.Module):
    """
    标准Transformer块（支持Mask输入）：Attention → FFN
    适配MDTA注意力，支持mask感知，用于图像修复/生成任务
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        # 注意力前的归一化
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_MDTA(dim, num_heads, bias)
        # FFN前的归一化
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        """
        前向传播（支持mask输入）
        Args:
            x: 输入特征 [B, dim, H, W]
            mask: 掩码张量（可选）
        Returns:
            输出特征 [B, dim, H, W]
        """
        # 残差1：x + 注意力输出（带mask）
        x = x + self.attn(self.norm1(x), mask=mask)
        # 残差2：x + FFN输出
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    """
    重叠图像块嵌入（Overlap Patch Embedding）
    使用3x3卷积实现，步长1，无重叠（padding=1）
    作用：将输入图像的通道数映射到模型的嵌入维度
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        """
        Args:
            in_c: 输入通道数（如RGB图=3，mask+图像=4）
            embed_dim: 嵌入维度（模型特征通道数）
            bias: 卷积层是否使用偏置
        """
        super(OverlapPatchEmbed, self).__init__()

        # 3x3卷积：通道数从in_c→embed_dim，尺寸不变（stride=1, padding=1）
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, in_c, H, W]
        Returns:
            嵌入特征 [B, embed_dim, H, W]
        """
        x = self.proj(x)
        return x


##########################################################################
## Gated Embedding layer（门控嵌入层）
class GatedEmb(nn.Module):
    """
    门控嵌入层：将输入通道数映射到目标维度，带门控机制
    常用于图像+mask的联合嵌入
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        """
        Args:
            in_c: 输入通道数（如图像+mask=4）
            embed_dim: 输出嵌入维度
            bias: 卷积层是否使用偏置
        """
        super(GatedEmb, self).__init__()

        # 1x1卷积：通道数→2×embed_dim（为分块门控做准备）
        self.gproj1 = nn.Conv2d(in_c, embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        """
        前向传播：卷积→分块→门控
        Args:
            x: 输入张量 [B, in_c, H, W]
        Returns:
            门控嵌入特征 [B, embed_dim, H, W]
        """
        # 卷积升维：[B, in_c, H, W] → [B, 2*embed_dim, H, W]
        x = self.gproj1(x)
        # 按通道分块：x1, x2 → 各 [B, embed_dim, H, W]
        x1, x2 = x.chunk(2, dim=1)

        # 门控机制：GELU(x1) × x2 → 输出 [B, embed_dim, H, W]
        x = F.gelu(x1) * x2
        return x


##########################################################################
## Mask-aware Pixel-Shuffle Down-Sampling (MPD)
class Downsample(nn.Module):
    """
    掩码感知的像素重排下采样（MPD）
    作用：特征图尺寸减半，通道数翻倍；同时融入mask信息
    输入：特征图x + mask，输出：下采样后的特征（通道数×2，尺寸÷2）
    """

    def __init__(self, n_feat):
        """
        Args:
            n_feat: 输入特征通道数
        """
        super(Downsample, self).__init__()

        # 特征下采样分支：1x1卷积（通道÷2） + PixelUnshuffle（尺寸÷2，通道×4）
        # 最终：n_feat → n_feat/2 ×4 = 2×n_feat（通道翻倍，尺寸减半）
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

        # Mask下采样分支：仅PixelUnshuffle（尺寸÷2，通道×4）
        self.body2 = nn.Sequential(nn.PixelUnshuffle(2))

        # 融合投影：将特征+mask的拼接特征投影回2×n_feat通道
        self.proj = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=1, padding=1, groups=n_feat * 2, bias=False)

    def forward(self, x, mask):
        """
        前向传播：特征下采样 + mask下采样 + 融合投影
        Args:
            x: 输入特征 [B, n_feat, H, W]
            mask: 掩码张量 [B, C_mask, H, W]
        Returns:
            下采样特征 [B, 2*n_feat, H/2, W/2]
        """
        # 特征下采样：[B, n_feat, H, W] → [B, 2*n_feat, H/2, W/2]
        out = self.body(x)
        # Mask下采样：[B, C_mask, H, W] → [B, 4*C_mask, H/2, W/2]
        out_mask = self.body2(mask)

        # 获取下采样后的维度
        b, n, h, w = out.shape
        # 创建融合张量（初始化为0）：[B, 2*n, h, w]
        t = torch.zeros((b, 2 * n, h, w)).cuda()

        # 填充特征部分：偶数通道填充特征
        for i in range(n):
            t[:, 2 * i, :, :] = out[:, i, :, :]
        # 填充Mask部分：奇数通道填充mask（mask通道不足则循环使用）
        for i in range(n):
            if i <= 3:
                t[:, 2 * i + 1, :, :] = out_mask[:, i, :, :]
            else:
                t[:, 2 * i + 1, :, :] = out_mask[:, (i % 4), :, :]

        # 分组卷积投影：融合特征+mask，输出维度还原为2*n_feat
        return self.proj(t)


class Upsample(nn.Module):
    """
    像素重排上采样（Upsample）
    作用：特征图尺寸翻倍，通道数减半
    """

    def __init__(self, n_feat):
        """
        Args:
            n_feat: 输入特征通道数
        """
        super(Upsample, self).__init__()

        # 上采样分支：1x1卷积（通道×2） + PixelShuffle（尺寸×2，通道÷4）
        # 最终：n_feat → n_feat×2 ÷4 = n_feat/2（通道减半，尺寸翻倍）
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, mask):
        """
        前向传播（mask参数保留但未使用，保持接口统一）
        Args:
            x: 输入特征 [B, n_feat, H, W]
            mask: 掩码张量（未使用）
        Returns:
            上采样特征 [B, n_feat/2, 2H, 2W]
        """
        return self.body(x)


# 对特征进行轴对称翻转
def flip_feature(feature, x1, y1, x2, y2):
    """
    沿任意直线（由两点(x1,y1)和(x2,y2)定义）对特征图进行轴对称翻转
    适用于图像修复/生成中的对称补全任务
    Args:
        feature: 输入特征张量 [B, C, H, W]（默认256×256，非256需缩放坐标）
        x1, y1: 对称轴第一个点的坐标
        x2, y2: 对称轴第二个点的坐标
    Returns:
        flipped_image: 翻转后的特征张量 [B, C, H, W]
    """
    # 获取特征图维度
    batch_size, channels, height, width = feature.shape
    # 选择计算设备（GPU优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 计算直线方程 Ax + By + C = 0 的系数
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    # 移到GPU（如果可用）
    A = A.cuda()
    B = B.cuda()
    C = C.cuda()

    # 创建坐标网格（H×W）
    y0, x0 = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    y0 = y0.cuda()
    x0 = x0.cuda()

    # 计算点(x0,y0)到直线的投影系数k
    k = -2 * (A * x0 + B * y0 + C) / (A ** 2 + B ** 2)

    # 计算投影点坐标（翻转后的坐标）
    x_prime = (x0 + k * A).float()
    y_prime = (y0 + k * B).float()

    # 裁剪坐标到特征图范围内（防止越界）
    x_prime = torch.clamp(x_prime, 0, width - 1)
    y_prime = torch.clamp(y_prime, 0, height - 1)

    # 初始化翻转后的特征图
    flipped_image = torch.zeros_like(feature)

    # 逐批次、逐通道进行双线性插值（反向映射）
    for b in range(batch_size):
        for c in range(channels):
            flipped_image[b, c] = bilinear_interpolate(feature[b, c], x_prime, y_prime)

    return flipped_image


def bilinear_interpolate(img, x, y):
    """
    双线性插值函数（用于翻转特征的像素值插值）
    Args:
        img: 单通道特征图 [H, W]
        x: 目标x坐标网格 [H, W]
        y: 目标y坐标网格 [H, W]
    Returns:
        插值后的特征图 [H, W]
    """
    # 计算四个邻近点的整数坐标
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # 裁剪坐标到图像范围内
    x0 = torch.clamp(x0, 0, img.shape[1] - 1)
    x1 = torch.clamp(x1, 0, img.shape[1] - 1)
    y0 = torch.clamp(y0, 0, img.shape[0] - 1)
    y1 = torch.clamp(y1, 0, img.shape[0] - 1)

    # 获取四个邻近点的像素值
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    # 计算双线性插值权重
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # 加权求和得到插值结果
    return wa * Ia + wb * Ib + wc * Ic + wd * Id




class GateConv(nn.Module):
    """
    门控卷积层（Gated Convolution）
    核心思想：将卷积输出分为特征分支和门控分支，通过sigmoid门控权重对特征进行调制
    作用：增强特征表达能力，自适应地选择重要特征，常用于图像生成/修复任务
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        """
        Args:
            in_channels: 输入特征通道数
            out_channels: 输出特征通道数（门控分支不占用最终输出通道）
            kernel_size: 卷积核尺寸
            stride: 卷积步长
            padding: 卷积填充（transpose=True时为输出填充）
            transpose: 是否使用转置卷积（用于上采样）
        """
        super(GateConv, self).__init__()
        self.out_channels = out_channels

        # 根据是否转置选择卷积类型：转置卷积用于上采样，普通卷积用于下采样/特征提取
        if transpose:
            # 转置卷积：输出通道数为2*out_channels（特征+门控）
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        else:
            # 普通卷积：输出通道数为2*out_channels（特征+门控）
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding)

    def forward(self, x):
        """
        前向传播：卷积 → 分块 → 门控调制
        Args:
            x: 输入张量 [B, in_channels, H, W]
        Returns:
            门控调制后的特征 [B, out_channels, H', W']
        """
        # 卷积输出：[B, 2*out_channels, H', W']
        x = self.gate_conv(x)
        # 按通道维度分块：特征分支x + 门控分支g → 各 [B, out_channels, H', W']
        (x, g) = torch.split(x, self.out_channels, dim=1)
        # 门控机制：特征 × sigmoid(门控) → 门控值∈[0,1]，对特征进行加权
        return x * torch.sigmoid(g)


class ResnetBlock(nn.Module):
    """
    改进版ResNet残差块（无最后ReLU激活）
    结构：投影（可选）→ 卷积1 → BN → ReLU → 卷积2 → BN → 残差连接
    特点：使用反射填充（ReflectionPad）避免边界失真，支持谱归一化和空洞卷积
    """

    def __init__(self, input_dim, out_dim=None, dilation=1, use_spectral_norm=False):
        """
        Args:
            input_dim: 输入特征通道数
            out_dim: 输出特征通道数（None则等于input_dim）
            dilation: 空洞卷积率（扩大感受野）
            use_spectral_norm: 是否使用谱归一化（提升GAN训练稳定性）
        """
        super(ResnetBlock, self).__init__()

        # 投影层：用于调整输入通道数（当input_dim≠out_dim时）
        if out_dim is not None:
            self.proj = nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1, bias=False)
        else:
            self.proj = None
            out_dim = input_dim  # 无投影时，输出通道=输入通道

        # 第一卷积层：反射填充 + 谱归一化卷积（空洞卷积）
        self.conv1 = nn.Sequential(
            # 反射填充：根据空洞率计算填充大小，保证输出尺寸不变
            nn.ReflectionPad2d(dilation),
            # 谱归一化卷积：3x3空洞卷积，padding=0（已通过反射填充处理）
            spectral_norm(
                nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=0, dilation=dilation,
                          bias=not use_spectral_norm), use_spectral_norm)
        )
        self.bn1 = nn.BatchNorm2d(out_dim)  # 批归一化
        self.act = nn.ReLU(True)  # ReLU激活（inplace=True节省内存）

        # 第二卷积层：普通3x3卷积（无空洞）
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),  # 普通填充（dilation=1）
                                   spectral_norm(
                                       nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=0,
                                                 dilation=1,
                                                 bias=not use_spectral_norm), use_spectral_norm))
        self.bn2 = nn.BatchNorm2d(out_dim)  # 批归一化

    def forward(self, x):
        """
        前向传播：残差连接 + 卷积块
        Args:
            x: 输入张量 [B, input_dim, H, W]
        Returns:
            残差输出 [B, out_dim, H, W]
        """
        # 可选投影：调整通道数以匹配残差连接
        if self.proj is not None:
            x = self.proj(x)

        # 主分支：卷积1 → BN → ReLU → 卷积2 → BN
        y = self.conv1(x)
        y = self.bn1(y.to(torch.float32))  # 转float32避免精度问题
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y.to(torch.float32))

        # 残差连接：输入 + 主分支输出（无最后ReLU，参考ResNet原版设计）
        out = x + y

        # 移除残差块末尾的ReLU，提升梯度流动
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class StructureEncoder(nn.Module):
    """
    结构编码器（Encoder-Decoder架构）
    作用：提取图像的多尺度结构特征，支持两种输出模式（编码器输出/解码器输出）
    结构：下采样卷积（GateConv）→ 残差块 → 上采样转置卷积（GateConv）
    """

    def __init__(self):
        super().__init__()

        # 控制是否使用rezero for mpe（位置编码相关，当前禁用）
        self.rezero_for_mpe = False
        # 输出模式控制：
        # True: 使用解码器输出（多尺度上采样特征）
        # False: 使用编码器输出（多尺度下采样特征）
        self.use_decoder_output = True

        # 第一层：反射填充 + 门控卷积（7x7大核提取全局特征）
        self.pad1 = nn.ReflectionPad2d(3)  # 7x7卷积需要padding=3，保持尺寸不变
        self.conv1 = GateConv(in_channels=3, out_channels=48, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(48)
        self.act = nn.ReLU(True)

        # 下采样卷积2：4x4步长2，通道48→96，尺寸减半
        self.conv2 = GateConv(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(96)

        # 下采样卷积3：4x4步长2，通道96→192，尺寸减半
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(192)

        # 下采样卷积4：4x4步长2，通道192→384，尺寸减半
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(384)

        # 中间残差块：3个空洞卷积残差块（dilation=2），增强特征表达
        blocks = []
        for i in range(3):
            blocks.append(ResnetBlock(input_dim=384, out_dim=None, dilation=2))
        self.middle = nn.Sequential(*blocks)

        # 可学习的缩放系数（alpha）：对各层特征进行自适应加权
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # 上采样转置卷积1：通道384→192，尺寸翻倍
        self.convt1 = GateConv(384, 192, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt1 = nn.BatchNorm2d(192)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # 上采样转置卷积2：通道192→96，尺寸翻倍
        self.convt2 = GateConv(192, 96, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt2 = nn.BatchNorm2d(96)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # 上采样转置卷积3：通道96→48，尺寸翻倍
        self.convt3 = GateConv(96, 48, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt3 = nn.BatchNorm2d(48)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # 以下为rezero_for_mpe相关代码（当前禁用）
        # if self.rezero_for_mpe:
        #     self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=config.rel_pos_num,
        #                                                            embedding_dim=64)
        #     self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
        #     self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        #     self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, x, rel_pos=None, direct=None):
        """
        前向传播：支持编码器/解码器两种输出模式
        Args:
            x: 输入图像 [B, 3, H, W]（默认256x256）
            rel_pos: 相对位置编码（rezero_for_mpe=True时使用）
            direct: 方向编码（rezero_for_mpe=True时使用）
        Returns:
            return_feats: 多尺度特征列表（4层）
            （可选）rel_pos_emb/direct_emb: 位置/方向编码特征
        """
        # 初始处理：反射填充 → 门控卷积 → BN → ReLU
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        if not self.use_decoder_output:  # 模式1：使用编码器输出（下采样特征）
            # 第1层特征：48通道，256x256（alpha1加权）
            feat1 = x * self.alpha1

            # 下采样到96通道，128x128
            x = self.conv2(x)
            x = self.bn2(x.to(torch.float32))
            x = self.act(x)
            feat2 = x * self.alpha2

            # 下采样到192通道，64x64
            x = self.conv3(x)
            x = self.bn3(x.to(torch.float32))
            x = self.act(x)
            feat3 = x * self.alpha3

            # 下采样到384通道，32x32
            x = self.conv4(x)
            x = self.bn4(x.to(torch.float32))
            x = self.act(x)
            feat4 = x * self.alpha4

            # 特征列表：[48,96,192,384]通道，尺寸依次减半
            return_feats = [feat1, feat2, feat3, feat4]

        else:  # 模式2：使用解码器输出（上采样特征，默认启用）
            # 编码器下采样流程（无特征保存）
            x = self.conv2(x)
            x = self.bn2(x.to(torch.float32))
            x = self.act(x)

            x = self.conv3(x)
            x = self.bn3(x.to(torch.float32))
            x = self.act(x)

            x = self.conv4(x)
            x = self.bn4(x.to(torch.float32))
            x = self.act(x)

            return_feats = []
            # 中间残差块处理：384通道，32x32
            x = self.middle(x)
            return_feats.append(x * self.alpha1)  # 384通道特征

            # 上采样1：384→192通道，32→64x64
            x = self.convt1(x)
            x = self.bnt1(x.to(torch.float32))
            x = self.act(x)
            return_feats.append(x * self.alpha2)  # 192通道特征

            # 上采样2：192→96通道，64→128x128
            x = self.convt2(x)
            x = self.bnt2(x.to(torch.float32))
            x = self.act(x)
            return_feats.append(x * self.alpha3)  # 96通道特征

            # 上采样3：96→48通道，128→256x256
            x = self.convt3(x)
            x = self.bnt3(x.to(torch.float32))
            x = self.act(x)
            return_feats.append(x * self.alpha4)  # 48通道特征

            # 特征列表反转：[48,96,192,384]通道（与编码器输出格式一致）
            return_feats = return_feats[::-1]

        # 输出逻辑：禁用rezero_for_mpe时仅返回特征列表
        if not self.rezero_for_mpe:
            return return_feats
        # 以下为rezero_for_mpe=True时的输出逻辑（当前禁用）
        # else:
        #     b, h, w = rel_pos.shape
        #     rel_pos = rel_pos.reshape(b, h * w)
        #     rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
        #     direct = direct.reshape(b, h * w, 4).to(torch.float32)
        #     direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6
        #
        #     return return_feats, rel_pos_emb, direct_emb


class EdgeGenerator(nn.Module):
    """
    边缘生成器（基于U-Net的Transformer架构）
    核心结构：
    - 编码器：4级下采样 + TransformerBlock（提取多尺度特征）
    - 解码器：4级上采样 + TransformerBlock（融合多尺度特征）
    - 精炼模块：额外TransformerBlock提升边缘细节
    特点：集成Mask-aware Pixel-Shuffle Down-Sampling (MPD)，支持掩码感知的特征采样
    """

    def __init__(self,
                 inp_channels=6,  # 输入通道数：edge_mask + mask = 3+3 =6
                 out_channels=3,  # 输出通道数（边缘图为3通道）
                 dim=48,  # 基础特征维度
                 num_blocks=[2, 4, 4, 6],  # 各层级TransformerBlock数量 [level1, level2, level3, level4]
                 num_refinement_blocks=4,  # 精炼模块的TransformerBlock数量
                 heads=[1, 2, 4, 8],  # 各层级注意力头数（随通道数增加而增加）
                 ffn_expansion_factor=2.66,  # FFN通道扩展因子
                 bias=False,  # 卷积层是否使用偏置
                 LayerNorm_type='WithBias',  ## 层归一化类型：'WithBias' / 'BiasFree'
                 ):
        super(EdgeGenerator, self).__init__()

        # 图像块嵌入层：将输入（边缘图+掩码）映射到基础维度dim
        # self.patch_embed = GatedEmb(inp_channels,dim)  # 备选：门控嵌入层
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)  # 重叠块嵌入层（更稳定）

        # ==================== 编码器部分 ====================
        # 编码器Level1：dim=48，heads=1
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # 下采样Level1→Level2：dim=48→96（2^1*dim）
        self.down1_2 = Downsample(dim)
        # 编码器Level2：dim=96，heads=2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        # 下采样Level2→Level3：dim=96→192（2^2*dim）
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        # 编码器Level3：dim=192，heads=4
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # 下采样Level3→Level4：dim=192→384（2^3*dim）
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        # 潜在特征层（编码器最深层）：dim=384，heads=8
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # ==================== 解码器部分 ====================
        # 上采样Level4→Level3：dim=384→192
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        # 通道压缩：384→192（拼接后通道数翻倍，需压缩）
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1,
                                            bias=bias)
        # 解码器Level3：dim=192，heads=4
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # 上采样Level3→Level2：dim=192→96
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        # 通道压缩：192→96
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # 解码器Level2：dim=96，heads=2
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        # 上采样Level2→Level1：dim=96→48（无通道压缩，后续拼接后通道为96）
        self.up2_1 = Upsample(int(dim * 2 ** 1))

        # 解码器Level1：dim=96（拼接后），heads=1
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # ==================== 精炼模块 ====================
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        # ==================== 输出层 ====================
        self.output = nn.Sequential(
            # 1x1卷积：将96通道特征映射到3通道边缘图
            nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, inp_img, mask_whole, inp_edge, gray_img, mask_half, mask_quarter,
                mask_tiny):
        """
        前向传播：U-Net架构的边缘生成流程
        Args:
            inp_img: 原始输入图像 [B, 3, 256, 256]
            mask_whole: 完整掩码 [B, 3, 256, 256]
            inp_edge: 输入边缘图 [B, 3, 256, 256]
            gray_img: 灰度图（未使用）
            mask_half: 半尺寸掩码 [B, 3, 128, 128]
            mask_quarter: 1/4尺寸掩码 [B, 3, 64, 64]
            mask_tiny: 1/8尺寸掩码 [B, 3, 32, 32]
        Returns:
            out_dec_level1: 生成的边缘图 [B, 3, 256, 256]（值∈[0,1]）
        """
        # 输入融合：使用边缘图作为核心输入
        inp_img_fuse = inp_edge

        # 步骤1：输入嵌入（边缘图 + 完整掩码）
        # 拼接通道：inp_edge(3) + mask_whole(3) = 6通道 → 嵌入到48通道
        inp_enc_level1 = self.patch_embed(
            torch.cat((inp_img_fuse, mask_whole), dim=1))  # [B, 48, 256, 256]

        # 步骤2：编码器Level1
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [B, 48, 256, 256]

        # 步骤3：下采样Level1→Level2（掩码感知）
        inp_enc_level2 = self.down1_2(out_enc_level1, mask_whole)  # [B, 96, 128, 128]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # [B, 96, 128, 128]

        # 步骤4：下采样Level2→Level3（掩码感知）
        inp_enc_level3 = self.down2_3(out_enc_level2, mask_half)  # [B, 192, 64, 64]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # [B, 192, 64, 64]

        # 步骤5：下采样Level3→Level4（掩码感知）
        inp_enc_level4 = self.down3_4(out_enc_level3, mask_quarter)  # [B, 384, 32, 32]
        latent = self.latent(inp_enc_level4)  # [B, 384, 32, 32]

        # ==================== 解码器流程 ====================
        # 步骤6：上采样Level4→Level3 + 跳跃连接
        inp_dec_level3 = self.up4_3(latent, mask_tiny)  # [B, 192, 64, 64]
        # 跳跃连接：上采样特征 + 编码器Level3特征 → 通道拼接 [192+192=384]
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)  # [B, 384, 64, 64]
        # 通道压缩：384→192
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # [B, 192, 64, 64]
        # 解码器Level3
        out_dec_level3 = self.decoder_level3(inp_dec_level3)  # [B, 192, 64, 64]

        # 步骤7：上采样Level3→Level2 + 跳跃连接
        inp_dec_level2 = self.up3_2(out_dec_level3, mask_quarter)  # [B, 96, 128, 128]
        # 跳跃连接：[96+96=192]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # [B, 192, 128, 128]
        # 通道压缩：192→96
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # [B, 96, 128, 128]
        # 解码器Level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # [B, 96, 128, 128]

        # 步骤8：上采样Level2→Level1 + 跳跃连接
        inp_dec_level1 = self.up2_1(out_dec_level2, mask_half)  # [B, 48, 256, 256]
        # 跳跃连接：[48+48=96]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # [B, 96, 256, 256]
        # 解码器Level1（无通道压缩）
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # [B, 96, 256, 256]

        # 步骤9：精炼模块（提升边缘细节）
        out_dec_level1 = self.refinement(out_dec_level1)  # [B, 96, 256, 256]

        # 步骤10：输出层 + 归一化
        out_dec_level1 = self.output(out_dec_level1)  # [B, 3, 256, 256]
        # tanh激活→[-1,1] → 归一化到[0,1]（符合图像像素值范围）
        out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2

        return out_dec_level1







# class EdgeGenerator(nn.Module):      # 加上mask-aware
#     def __init__(self,
#                  inp_channels=6,  # 6:edge_mask + mask = 3+3 =6
#                  out_channels=3,
#                  dim=48,
#                  num_blocks=[2, 4, 4, 6],  # 初始设置：num_blocks=[4, 6, 6, 8],
#                  num_refinement_blocks=4,
#                  heads=[1, 2, 4, 8],
#                  ffn_expansion_factor=2.66,
#                  bias=False,
#                  LayerNorm_type='WithBias',  ## Other option 'BiasFree'
#                  ):
#         super(EdgeGenerator, self).__init__()
#
#         # self.patch_embed = GatedEmb(inp_channels,dim)  # Gated Embedding layer,   即 inp_channels=4,该模块的输入为torch.cat((inp_img,mask_whole)
#         self.patch_embed = OverlapPatchEmbed(inp_channels, dim)  # inp_channels=6
#
#         # [Modified] Sequential -> ModuleList
#         self.encoder_level1 = nn.ModuleList([
#             TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                              LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#         self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
#
#         self.encoder_level2 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
#
#         self.encoder_level3 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4 ，输出 dim=int(dim * 2 ** 3)
#
#         self.latent = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
#
#         self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
#         self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1,
#                                             bias=bias)  # 1*1卷积核，进行改变通道维度操作
#
#         self.decoder_level3 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
#         self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
#
#         self.decoder_level2 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
#
#         self.decoder_level1 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#         self.refinement = nn.ModuleList([  # 最后的精炼模块
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
#
#         self.output = nn.Sequential(
#             nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#         )
#
#     def forward(self, inp_img, mask_whole, inp_edge, gray_img, mask_half, mask_quarter,
#                 mask_tiny):
#         # mask_whole, mask_half, mask_quarter, mask_tiny 用于 Attention
#
#         # inp_img_fuse = inp_edge (masked edge)
#         inp_img_fuse = inp_edge
#
#         inp_enc_level1 = self.patch_embed(
#             torch.cat((inp_img_fuse, mask_whole), dim=1))  # 拼接 edge 和 mask
#
#         # [Modified] Level 1: 使用 mask_whole
#         out_enc_level1 = inp_enc_level1
#         for blk in self.encoder_level1:
#             out_enc_level1 = blk(out_enc_level1, mask=mask_whole)
#
#         # Down 1->2
#         inp_enc_level2 = self.down1_2(out_enc_level1, mask_whole)
#
#         # [Modified] Level 2: 使用 mask_half
#         out_enc_level2 = inp_enc_level2
#         for blk in self.encoder_level2:
#             out_enc_level2 = blk(out_enc_level2, mask=mask_half)
#
#         # Down 2->3
#         inp_enc_level3 = self.down2_3(out_enc_level2, mask_half)
#
#         # [Modified] Level 3: 使用 mask_quarter
#         out_enc_level3 = inp_enc_level3
#         for blk in self.encoder_level3:
#             out_enc_level3 = blk(out_enc_level3, mask=mask_quarter)
#
#         # Down 3->4
#         inp_enc_level4 = self.down3_4(out_enc_level3, mask_quarter)
#
#         # [Modified] Latent: 使用 mask_tiny
#         latent = inp_enc_level4
#         for blk in self.latent:
#             latent = blk(latent, mask=mask_tiny)
#
#         # Up 4->3
#         inp_dec_level3 = self.up4_3(latent, mask_tiny)
#         inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
#         inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
#
#         # [Modified] Dec Level 3: 使用 mask_quarter
#         out_dec_level3 = inp_dec_level3
#         for blk in self.decoder_level3:
#             out_dec_level3 = blk(out_dec_level3, mask=mask_quarter)
#
#         # Up 3->2
#         inp_dec_level2 = self.up3_2(out_dec_level3, mask_quarter)
#         inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
#         inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
#
#         # [Modified] Dec Level 2: 使用 mask_half
#         out_dec_level2 = inp_dec_level2
#         for blk in self.decoder_level2:
#             out_dec_level2 = blk(out_dec_level2, mask=mask_half)
#
#         # Up 2->1
#         inp_dec_level1 = self.up2_1(out_dec_level2, mask_half)
#         inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
#
#         # [Modified] Dec Level 1: 使用 mask_whole
#         out_dec_level1 = inp_dec_level1
#         for blk in self.decoder_level1:
#             out_dec_level1 = blk(out_dec_level1, mask=mask_whole)
#
#         # [Modified] Refinement: 使用 mask_whole
#         for blk in self.refinement:
#             out_dec_level1 = blk(out_dec_level1, mask=mask_whole)
#
#         out_dec_level1 = self.output(out_dec_level1)
#         out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2
#         print('-------------------------------------')
#
#         return out_dec_level1




class InpaintGenerator(nn.Module):
    """
    图像修复生成器（基于Restormer架构的U-Net Transformer）
    核心改进：
    1. 集成StructureEncoder提取边缘结构特征，支持多尺度特征叠加
    2. 灵活的边缘信息融合策略：结构编码器特征叠加 / 通道拼接
    3. 掩码感知的下采样/上采样（MPD），适配图像修复任务
    整体架构：Encoder-Decoder + 多尺度特征融合 + 边缘结构增强
    """

    def __init__(self,
                 inp_channels=3,  # 基础输入通道数（可动态调整为4/5通道）
                 out_channels=3,  # 输出通道数（RGB图像修复结果）
                 dim=48,  # 基础特征维度（Restormer默认48）
                 num_blocks=[2, 4, 4, 6],  # 各层级TransformerBlock数量 [L1,L2,L3,L4]
                 num_refinement_blocks=4,  # 最终精炼模块的TransformerBlock数量
                 heads=[1, 2, 4, 8],  # 各层级注意力头数（随通道数增加而增加）
                 ffn_expansion_factor=2.66,  # FFN通道扩展因子（Restormer推荐值）
                 bias=False,  # 卷积层是否使用偏置（Restormer最佳实践）
                 LayerNorm_type='WithBias',  ## 层归一化类型：'WithBias'/'BiasFree'
                 use_decoder_output=True,  # StructureEncoder输出模式：True=解码器输出，False=编码器输出
                 ):
        super(InpaintGenerator, self).__init__()

        # ==================== 核心组件初始化 ====================
        # 图像块嵌入层：将输入图像映射到基础特征维度dim
        # 注：原注释中inp_channels=6是误写，实际根据融合策略动态调整为3/4/5通道
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 结构编码器：提取边缘图像的多尺度结构特征
        self.strenc = StructureEncoder()
        self.strenc.use_decoder_output = use_decoder_output  # 同步结构编码器输出模式

        # ==================== 编码器部分（Restormer架构） ====================
        # 编码器Level1：基础维度dim=48，注意力头数=1
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # 下采样模块L1→L2：维度翻倍（48→96），尺寸减半（256→128）
        self.down1_2 = Downsample(dim)
        # 编码器Level2：维度dim=96，注意力头数=2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        # 下采样模块L2→L3：维度翻倍（96→192），尺寸减半（128→64）
        self.down2_3 = Downsample(int(dim * 2 ** 1))

        # 编码器Level3：维度dim=192，注意力头数=4
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # 下采样模块L3→L4：维度翻倍（192→384），尺寸减半（64→32）
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        # 潜在特征层（编码器最深层）：维度dim=384，注意力头数=8
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # ==================== 解码器部分（Restormer架构） ====================
        # 上采样模块L4→L3：维度减半（384→192），尺寸翻倍（32→64）
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        # 通道压缩卷积：拼接后维度384→192（适配解码器输入）
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1,
                                            bias=bias)
        # 解码器Level3：维度dim=192，注意力头数=4
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # 上采样模块L3→L2：维度减半（192→96），尺寸翻倍（64→128）
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        # 通道压缩卷积：拼接后维度192→96
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # 解码器Level2：维度dim=96，注意力头数=2
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        # 上采样模块L2→L1：维度减半（96→48），尺寸翻倍（128→256）
        self.up2_1 = Upsample(int(dim * 2 ** 1))

        # 解码器Level1：维度dim=96（拼接后），注意力头数=1
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # ==================== 精炼模块 ====================
        # 最终特征精炼：提升修复图像的细节和一致性
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        # ==================== 输出层 ====================
        self.output = nn.Sequential(
            # 3x3卷积：将96维特征映射到3通道RGB图像
            nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, inp_img, mask_whole, edge, gray_img, mask_half, mask_quarter,
                mask_tiny, is_strenc=True, is_edge=True):
        """
        前向传播：支持多种边缘信息融合策略的图像修复流程
        Args:
            inp_img: 待修复输入图像 [B, 3, 256, 256]
            mask_whole: 完整尺寸掩码 [B, 3, 256, 256] (1=缺失区域，0=有效区域)
            edge: 边缘图像 [B, 3, 256, 256] (可由边缘检测器生成)
            gray_img: 灰度图（当前未使用，预留接口）
            mask_half: 1/2尺寸掩码 [B, 3, 128, 128] (MPD下采样用)
            mask_quarter: 1/4尺寸掩码 [B, 3, 64, 64] (MPD下采样用)
            mask_tiny: 1/8尺寸掩码 [B, 3, 32, 32] (MPD上采样用)
            is_strenc: 是否使用StructureEncoder提取边缘特征（True=特征叠加，False=通道拼接）
            is_edge: 是否使用边缘信息（True=使用，False=仅用图像+掩码）
        Returns:
            out_dec_level1: 修复后的图像 [B, 3, 256, 256] (像素值∈[0,1])
        """
        # 基础输入：待修复图像
        inp_img_fuse = inp_img

        # ==================== 分支1：使用边缘信息 ====================
        if is_edge:
            # 子分支1.1：使用StructureEncoder提取边缘结构特征（特征叠加）
            if is_strenc:
                # 结构编码器提取边缘的多尺度特征 [feat1(48), feat2(96), feat3(192), feat4(384)]
                feature_edge = self.strenc(edge)
                # 图像嵌入：仅输入图像（3通道）→ 48维特征
                inp_enc_level1 = self.patch_embed(inp_img_fuse)  # [B, 48, 256, 256]

            # 子分支1.2：不使用StructureEncoder（边缘+掩码通道拼接）
            else:
                # 掩码降维：3通道→1通道（取均值）[B, 1, H, W]
                mask_whole = mask_whole.mean(dim=1, keepdim=True)
                # 边缘图降维：3通道→1通道 [B, 1, H, W]
                edge = edge.mean(dim=1, keepdim=True)
                # 通道拼接：图像(3) + 掩码(1) + 边缘(1) = 5通道 → 48维特征
                inp_enc_level1 = self.patch_embed(
                    torch.cat((inp_img_fuse, mask_whole, edge), dim=1)
                )  # [B, 48, 256, 256]

        # ==================== 分支2：不使用边缘信息 ====================
        else:
            # 掩码降维：3通道→1通道
            mask_whole = mask_whole.mean(dim=1, keepdim=True)
            # 通道拼接：图像(3) + 掩码(1) = 4通道 → 48维特征
            inp_enc_level1 = self.patch_embed(
                torch.cat((inp_img_fuse, mask_whole), dim=1))  # [B, 48, 256, 256]

        # ==================== 编码器Level1：基础特征提取 ====================
        # 特征叠加：StructureEncoder的48维边缘特征 + 图像嵌入特征（增强结构信息）
        if is_strenc:
            inp_enc_level1 = inp_enc_level1 + feature_edge[0]

        # TransformerBlock组处理Level1特征
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [B, 48, 256, 256]

        # ==================== 编码器Level2：下采样+特征融合 ====================
        # 掩码感知下采样L1→L2（MPD）：48→96维，256→128尺寸
        inp_enc_level2 = self.down1_2(out_enc_level1, mask_whole)  # [B, 96, 128, 128]

        # 特征叠加：StructureEncoder的96维边缘特征 + Level2特征
        if is_strenc:
            inp_enc_level2 = inp_enc_level2 + feature_edge[1]

        # TransformerBlock组处理Level2特征
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # [B, 96, 128, 128]

        # ==================== 编码器Level3：下采样 ====================
        # 掩码感知下采样L2→L3：96→192维，128→64尺寸
        inp_enc_level3 = self.down2_3(out_enc_level2, mask_half)  # [B, 192, 64, 64]

        # （可选）Level3特征叠加（当前注释掉，可根据效果开启）
        # if is_strenc:
        #     inp_enc_level3 = inp_enc_level3 + feature_edge[2]

        # TransformerBlock组处理Level3特征
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # [B, 192, 64, 64]

        # ==================== 编码器Level4（Latent）：最深层特征 ====================
        # 掩码感知下采样L3→L4：192→384维，64→32尺寸
        inp_enc_level4 = self.down3_4(out_enc_level3, mask_quarter)  # [B, 384, 32, 32]

        # （可选）Level4特征叠加（当前注释掉，可根据效果开启）
        # if is_strenc:
        #     inp_enc_level4 = inp_enc_level4 + feature_edge[3]

        # TransformerBlock组处理Latent特征（最深层特征提取）
        latent = self.latent(inp_enc_level4)  # [B, 384, 32, 32]

        # ==================== 解码器Level3：上采样+跳跃连接 ====================
        # 掩码感知上采样L4→L3：384→192维，32→64尺寸
        inp_dec_level3 = self.up4_3(latent, mask_tiny)  # [B, 192, 64, 64]

        # 跳跃连接：上采样特征 + 编码器Level3特征（通道拼接）
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)  # [B, 384, 64, 64]

        # 通道压缩：384→192维（适配解码器输入维度）
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # [B, 192, 64, 64]

        # TransformerBlock组处理Level3解码特征
        out_dec_level3 = self.decoder_level3(inp_dec_level3)  # [B, 192, 64, 64]

        # ==================== 解码器Level2：上采样+跳跃连接 ====================
        # 掩码感知上采样L3→L2：192→96维，64→128尺寸
        inp_dec_level2 = self.up3_2(out_dec_level3, mask_quarter)  # [B, 96, 128, 128]

        # 跳跃连接：上采样特征 + 编码器Level2特征（通道拼接）
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # [B, 192, 128, 128]

        # 通道压缩：192→96维
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # [B, 96, 128, 128]

        # TransformerBlock组处理Level2解码特征
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # [B, 96, 128, 128]

        # ==================== 解码器Level1：上采样+跳跃连接 ====================
        # 掩码感知上采样L2→L1：96→48维，128→256尺寸
        inp_dec_level1 = self.up2_1(out_dec_level2, mask_half)  # [B, 48, 256, 256]

        # 跳跃连接：上采样特征 + 编码器Level1特征（通道拼接）
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # [B, 96, 256, 256]

        # TransformerBlock组处理Level1解码特征
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # [B, 96, 256, 256]

        # ==================== 特征精炼+输出 ====================
        # 最终特征精炼：提升修复图像的细节和一致性
        out_dec_level1 = self.refinement(out_dec_level1)  # [B, 96, 256, 256]

        # 输出层：96维特征→3通道RGB图像
        out_dec_level1 = self.output(out_dec_level1)  # [B, 3, 256, 256]

        # 归一化：tanh→[-1,1] → 映射到[0,1]（符合图像像素值范围）
        out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2

        return out_dec_level1

















# ##########################################################################
# ##---------- HINT -----------------------  根据原来的HINT更改架构，与Restormer主架构一致   加mask-aware
# class InpaintGenerator(nn.Module):
#     def __init__(self,
#                  inp_channels=3,        #  用结构编码器：3    通道连接：5   直接修复（不用边缘信息）： 4
#                  out_channels=3,
#                  dim=48,
#                  num_blocks=[2, 4, 4, 6],  # 初始设置：num_blocks=[4, 6, 6, 8], #  [2, 4, 4, 6]
#                  num_refinement_blocks=4,
#                  heads=[1, 2, 4, 8],
#                  ffn_expansion_factor=2.66,
#                  bias=False,
#                  LayerNorm_type='WithBias',  ## Other option 'BiasFree'
#                  ):
#         super(InpaintGenerator, self).__init__()
#
#         #self.patch_embed = GatedEmb(inp_channels,dim)  # Gated Embedding layer,   即 inp_channels=4,该模块的输入为torch.cat((inp_img,mask_whole)
#         self.patch_embed = OverlapPatchEmbed(inp_channels, dim)  # inp_channels=6
#
#         self.strenc = StructureEncoder()
#
#         # self.encoder_level1 = nn.Sequential(*[
#         #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#         #                   LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#         self.encoder_level1 = nn.ModuleList([
#             TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                           LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#         self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
#         # self.encoder_level2 = nn.Sequential(*[
#         #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#         #                   bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.encoder_level2 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                           bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
#
#
#         # self.encoder_level3 = nn.Sequential(*[
#         #     TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#         #                   bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.encoder_level3 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                           bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4 ，输出 dim=int(dim * 2 ** 3)
#
#
#         # self.latent = nn.Sequential(*[
#         #     TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
#         #                   bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
#
#         self.latent = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
#                           bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
#
#         self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
#         self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1,
#                                             bias=bias)  # 1*1卷积核，进行改变通道维度操作
#         # self.decoder_level3 = nn.Sequential(*[
#         #     TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#         #                   bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.decoder_level3 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                           bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
#         self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
#         # self.decoder_level2 = nn.Sequential(*[
#         #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#         #                   bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.decoder_level2 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                           bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
#
#         # self.decoder_level1 = nn.Sequential(*[
#         #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#         #                   bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#         self.decoder_level1 = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#                           bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#         # self.refinement = nn.Sequential(*[     #最后的精炼模块
#         #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#         #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
#
#         self.refinement = nn.ModuleList([
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
#
#         self.output = nn.Sequential(
#             nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#             )
#
#     def forward(self, inp_img, mask_whole, edge, gray_img, mask_half, mask_quarter,
#                 mask_tiny, is_strenc=True, is_edge=True):  # 在models.py中的InpaintingModel中的forward中用到此函数。mask_whole, mask_half, mask_quarter,mask_tiny 用作MPD时的mask与图片融合，上采样没用到。
#
#
#         inp_img_fuse = inp_img
#         if is_edge:
#             if is_strenc:
#                 feature_edge = self.strenc(edge)
#                 inp_enc_level1 = self.patch_embed(inp_img_fuse)   #in_ch ==3
#             else:   # 通道连接
#                 mask_whole = mask_whole.mean(dim=1, keepdim=True)
#                 edge = edge.mean(dim=1, keepdim=True)
#                 inp_enc_level1 = self.patch_embed(
#                     torch.cat((inp_img_fuse, mask_whole, edge), dim=1)
#                 )  # in_ch == 5
#         else:  # 不用边缘信息
#             mask_whole = mask_whole.mean(dim=1, keepdim=True)
#             inp_enc_level1 = self.patch_embed(
#                 torch.cat((inp_img_fuse, mask_whole), dim=1)) # in_ch == 4
#
#         if is_strenc:
#             inp_enc_level1 = inp_enc_level1 + feature_edge[0]
#
#         # [Modified] Level 1: 使用 mask_whole
#         out_enc_level1 = inp_enc_level1
#         for blk in self.encoder_level1:
#             out_enc_level1 = blk(out_enc_level1, mask=mask_whole)
#
#         # Down 1->2
#         inp_enc_level2 = self.down1_2(out_enc_level1, mask_whole)
#         if is_strenc:
#             inp_enc_level2 = inp_enc_level2 + feature_edge[1]
#
#         # [Modified] Level 2: 使用 mask_half
#         out_enc_level2 = inp_enc_level2
#         for blk in self.encoder_level2:
#             out_enc_level2 = blk(out_enc_level2, mask=mask_half)
#
#         # Down 2->3
#         inp_enc_level3 = self.down2_3(out_enc_level2, mask_half)
#         if is_strenc:
#             inp_enc_level3 = inp_enc_level3 + feature_edge[2]
#
#         # [Modified] Level 3: 使用 mask_quarter
#         out_enc_level3 = inp_enc_level3
#         for blk in self.encoder_level3:
#             out_enc_level3 = blk(out_enc_level3, mask=mask_quarter)
#
#         # Down 3->4
#         inp_enc_level4 = self.down3_4(out_enc_level3, mask_quarter)
#         if is_strenc:
#             inp_enc_level4 = inp_enc_level4 + feature_edge[3]
#
#         # [Modified] Latent: 使用 mask_tiny
#         latent = inp_enc_level4
#         for blk in self.latent:
#             latent = blk(latent, mask=mask_tiny)
#
#         # Up 4->3
#         inp_dec_level3 = self.up4_3(latent, mask_tiny)
#         inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
#         inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
#
#         # [Modified] Decoder Level 3: 使用 mask_quarter
#         out_dec_level3 = inp_dec_level3
#         for blk in self.decoder_level3:
#             out_dec_level3 = blk(out_dec_level3, mask=mask_quarter)
#
#         # Up 3->2
#         inp_dec_level2 = self.up3_2(out_dec_level3, mask_quarter)
#         inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
#         inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
#
#         # [Modified] Decoder Level 2: 使用 mask_half
#         out_dec_level2 = inp_dec_level2
#         for blk in self.decoder_level2:
#             out_dec_level2 = blk(out_dec_level2, mask=mask_half)
#
#         # Up 2->1
#         inp_dec_level1 = self.up2_1(out_dec_level2, mask_half)
#         inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
#
#         # [Modified] Decoder Level 1: 使用 mask_whole
#         out_dec_level1 = inp_dec_level1
#         for blk in self.decoder_level1:
#             out_dec_level1 = blk(out_dec_level1, mask=mask_whole)
#
#         # [Modified] Refinement: 使用 mask_whole
#         for blk in self.refinement:
#             out_dec_level1 = blk(out_dec_level1, mask=mask_whole)
#
#         out_dec_level1 = self.output(out_dec_level1)
#         out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2
#
#         return out_dec_level1
#
#         # if is_strenc:
#         #     inp_enc_level1 = inp_enc_level1 + feature_edge[0]
#         # out_enc_level1 = self.encoder_level1(
#         #     inp_enc_level1)  # 第一个TransformerBlock块组，  输出shape： out_enc_level1= torch.Size([1, 48, 256, 256])
#         #
#         # inp_enc_level2 = self.down1_2(out_enc_level1,
#         #                               mask_whole)  # 第一个MPD块       inp_enc_level2= torch.Size([1, 96, 128, 128])
#         # if is_strenc:
#         #     inp_enc_level2 = inp_enc_level2 + feature_edge[1]
#         # out_enc_level2 = self.encoder_level2(
#         #     inp_enc_level2)  # 第二个TransformerBlock块组    out_enc_level2= torch.Size([1, 96, 128, 128])
#         #
#         # inp_enc_level3 = self.down2_3(out_enc_level2,
#         #                               mask_half)  # 第二个MPD块          inp_enc_level3= torch.Size([1, 192, 64, 64])
#         #
#         # if is_strenc:
#         #     inp_enc_level3 = inp_enc_level3 + feature_edge[2]
#         # out_enc_level3 = self.encoder_level3(
#         #     inp_enc_level3)  # 第三个TransformerBlock块组       out_enc_level3= torch.Size([1, 192, 64, 64])
#         #
#         # inp_enc_level4 = self.down3_4(out_enc_level3,
#         #                               mask_quarter)  # 第三个MPD块          inp_enc_level4= torch.Size([1, 384, 32, 32])
#         #
#         # if is_strenc:
#         #     inp_enc_level4 = inp_enc_level4 + feature_edge[3]
#         # latent = self.latent(inp_enc_level4)  # 第四个TransformerBlock块组，即中间块      latent= torch.Size([1, 384, 32, 32])
#         #
#         # inp_dec_level3 = self.up4_3(latent,
#         #                             mask_tiny)  # 第一个上采样块，上采样操作：通道数减少，高宽增加       inp_dec_level3= torch.Size([1, 192, 64, 64])
#         # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3],
#         #                            1)  # 对第四个TransformerBlock块组进行上采样之后，与第三个三明治块组进行通道连接，作为输入，对应u-net网络中连接操作
#         # # inp_dec_level3= torch.Size([1, 384, 64, 64])
#         #
#         # inp_dec_level3 = self.reduce_chan_level3(
#         #     inp_dec_level3)  # 高宽不变，减少通道数，以适应下面TransformerBlock块的输入通道数    inp_dec_level3= torch.Size([1, 192, 64, 64])
#         #
#         # out_dec_level3 = self.decoder_level3(
#         #     inp_dec_level3)  # 第五个TransformerBlock块组           out_dec_level3= torch.Size([1, 192, 64, 64])
#         #
#         # inp_dec_level2 = self.up3_2(out_dec_level3,
#         #                             mask_quarter)  # 重复刚刚的操作        inp_dec_level2= torch.Size([1, 96, 128, 128])
#         #
#         # inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2],
#         #                            1)  # inp_dec_level2= torch.Size([1, 192, 128, 128])
#         #
#         # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # inp_dec_level2= torch.Size([1, 96, 128, 128])
#         #
#         # out_dec_level2 = self.decoder_level2(
#         #     inp_dec_level2)  # 第六个TransformerBlock块组       out_dec_level2= torch.Size([1, 96, 128, 128])
#         #
#         # inp_dec_level1 = self.up2_1(out_dec_level2, mask_half)  # inp_dec_level1= torch.Size([1, 48, 256, 256])
#         # inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1],
#         #                            1)  # inp_dec_level1= torch.Size([1, 96, 256, 256])#
#         #
#         # out_dec_level1 = self.decoder_level1(
#         #     inp_dec_level1)  # 第七个TransformerBlock块组    out_dec_level1= torch.Size([1, 96, 256, 256])
#         #
#         # out_dec_level1 = self.refinement(out_dec_level1)   # 最后的精炼模块
#         #
#         # out_dec_level1 = self.output(out_dec_level1)  # out_dec_level1= torch.Size([1, 3, 256, 256])
#         #
#         # out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2  # out_dec_level1= torch.Size([1, 3, 256, 256])
#         # print('-------------------------------------')
#         #
#         # return out_dec_level1



# ##########################################################################
# ##---------- HINT -----------------------  原来的
# class HINT_original(nn.Module):
#     def __init__(self,
#                  inp_channels=6,
#                  out_channels=3,
#                  dim=48,
#                  num_blocks=[1, 1, 1, 1],        #  初始设置：num_blocks=[4, 6, 6, 8],
#                  heads=[1, 2, 4, 8],
#                  ffn_expansion_factor=2.66,
#                  bias=False,
#                  LayerNorm_type='WithBias',  ## Other option 'BiasFree'
#                  ):
#
#         super(HINT_original, self).__init__()
#
#         self.patch_embed = GatedEmb(inp_channels, dim)    # Gated Embedding layer,   即 inp_channels=4,该模块的输入为torch.cat((inp_img,mask_whole)
#
#         self.encoder_level1 = nn.Sequential(*[
#             SandwichBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                              LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#         self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
#         self.encoder_level2 = nn.Sequential(*[
#             SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
#         self.encoder_level3 = nn.Sequential(*[
#             SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4 ，输出 dim=int(dim * 2 ** 3)
#         self.latent = nn.Sequential(*[
#             SandwichBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
#
#         self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
#         self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)   #1*1卷积核，进行改变通道维度操作
#         self.decoder_level3 = nn.Sequential(*[
#             SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
#
#         self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
#         self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
#         self.decoder_level2 = nn.Sequential(*[
#             SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
#
#         self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
#
#         self.decoder_level1 = nn.Sequential(*[
#             SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
#
#
#         self.output = nn.Sequential(nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#                                     )
#
#
#     def forward(self, inp_img, mask_whole, mask_half, mask_quarter,mask_tiny):  #在models.py中的InpaintingModel中的forward中用到此函数。mask_whole, mask_half, mask_quarter,mask_tiny 用作MPD时的mask与图片融合，上采样没用到。
#
#         inp_enc_level1 = self.patch_embed(torch.cat((inp_img,mask_whole),dim=1))      #对原始输入进行门控，输出shape： inp_enc_level1= torch.Size([1, 48, 256, 256])
#
#         out_enc_level1 = self.encoder_level1(inp_enc_level1)              #第一个三明治块组，  输出shape： out_enc_level1= torch.Size([1, 48, 256, 256])
#
#         inp_enc_level2 = self.down1_2(out_enc_level1,mask_whole)          #第一个MPD块       inp_enc_level2= torch.Size([1, 96, 128, 128])
#         out_enc_level2 = self.encoder_level2(inp_enc_level2)              #第二个三明治块组    out_enc_level2= torch.Size([1, 96, 128, 128])
#
#         inp_enc_level3 = self.down2_3(out_enc_level2,mask_half)             #第二个MPD块          inp_enc_level3= torch.Size([1, 192, 64, 64])
#         out_enc_level3 = self.encoder_level3(inp_enc_level3)                #第三个三明治块组       out_enc_level3= torch.Size([1, 192, 64, 64])
#
#         inp_enc_level4 = self.down3_4(out_enc_level3,mask_quarter)          #第三个MPD块          inp_enc_level4= torch.Size([1, 384, 32, 32])
#
#         latent = self.latent(inp_enc_level4)                                #第四个三明治块组，即中间块      latent= torch.Size([1, 384, 32, 32])
#
#         inp_dec_level3 = self.up4_3(latent,mask_tiny)                           #第一个上采样块，上采样操作：通道数减少，高宽增加       inp_dec_level3= torch.Size([1, 192, 64, 64])
#         inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)   #对第四个三明治块组进行上采样之后，与第三个三明治块组进行通道连接，作为输入，对应u-net网络中连接操作
#                                                                                     # inp_dec_level3= torch.Size([1, 384, 64, 64])
#
#         inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)                #高宽不变，减少通道数，以适应下面三明治块的输入通道数    inp_dec_level3= torch.Size([1, 192, 64, 64])
#
#         out_dec_level3 = self.decoder_level3(inp_dec_level3)                    #第五个三明治块组           out_dec_level3= torch.Size([1, 192, 64, 64])
#
#         inp_dec_level2 = self.up3_2(out_dec_level3,mask_quarter)                #重复刚刚的操作        inp_dec_level2= torch.Size([1, 96, 128, 128])
#
#         inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)         #inp_dec_level2= torch.Size([1, 192, 128, 128])
#
#         inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)                # inp_dec_level2= torch.Size([1, 96, 128, 128])
#
#         out_dec_level2 = self.decoder_level2(inp_dec_level2)                    #第六个三明治块组       out_dec_level2= torch.Size([1, 96, 128, 128])
#
#         inp_dec_level1 = self.up2_1(out_dec_level2,mask_half)                   # inp_dec_level1= torch.Size([1, 48, 256, 256])
#         inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)     #inp_dec_level1= torch.Size([1, 96, 256, 256])#
#
#         out_dec_level1 = self.decoder_level1(inp_dec_level1)                    #第七个三明治块组    out_dec_level1= torch.Size([1, 96, 256, 256])
#
#         out_dec_level1 = self.output(out_dec_level1)                    #out_dec_level1= torch.Size([1, 3, 256, 256])
#
#         out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2           #out_dec_level1= torch.Size([1, 3, 256, 256])
#         print('-------------------------------------')
#
#         return out_dec_level1
#



        
     
