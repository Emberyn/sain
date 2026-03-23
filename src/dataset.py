import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
# from scipy.misc import imread  # 废弃接口，改用imageio.imread
from skimage.color import rgb2gray
# from scipy.misc import imresize  # 废弃接口，改用PIL.Image.resize
from .utils import create_mask  # 自定义掩码生成工具
import cv2
from skimage.feature import canny  # 边缘检测工具


class Dataset(torch.utils.data.Dataset):
    """
    图像修复任务的自定义数据集类
    功能：
    1. 加载图像、掩码、边缘图数据
    2. 支持多种掩码生成策略（随机块、中心块、外部掩码等）
    3. 数据预处理（resize、中心裁剪、灰度转换）
    4. 格式转换（numpy→PIL→tensor）
    适配：PyTorch DataLoader，支持训练/测试模式切换
    """

    def __init__(self, config, flist, mask_flist, edge_flist, augment=True, training=True):
        """
        初始化数据集
        Args:
            config: 配置文件（包含输入尺寸、掩码类型等参数）
            flist: 图像文件路径列表/目录路径/文本文件（每行一个路径）
            mask_flist: 掩码文件路径列表/目录路径/文本文件
            edge_flist: 边缘图文件路径列表/目录路径/文本文件
            augment: 是否启用数据增强（当前代码未实现具体增强逻辑）
            training: 是否为训练模式（影响掩码选择策略）
        """
        super(Dataset, self).__init__()
        self.config = config  # 配置参数
        self.augment = augment  # 是否数据增强
        self.training = training  # 训练/测试模式

        # 加载各类数据的路径列表
        self.data = self.load_flist(flist)  # 原始图像路径列表
        self.mask_data = self.load_flist(mask_flist)  # 掩码图像路径列表
        self.edge_data = self.load_flist(edge_flist)  # 边缘图像路径列表

        self.input_size = config.INPUT_SIZE  # 输入图像尺寸（如256）
        self.mask = config.MASK  # 掩码类型（0-6，对应不同生成策略）

    def __len__(self):
        """返回数据集总长度（以图像数量为准）"""
        return len(self.data)

    def __getitem__(self, index):
        """
        按索引加载单条数据（DataLoader调用）
        Args:
            index: 数据索引
        Returns:
            张量形式的图像、掩码、边缘图、灰度图
        """
        item = self.load_item(index)
        return item

    def load_name(self, index):
        """获取指定索引的图像文件名（不含路径）"""
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        """
        加载单条数据并完成预处理
        Args:
            index: 数据索引
        Returns:
            img_tensor: 原始图像 [3, H, W]
            mask_tensor: 掩码 [1, H, W]
            edge_tensor: 边缘图 [3, H, W]
            gray_tensor: 灰度图 [1, H, W]
        """
        size = self.input_size  # 目标尺寸

        # 1. 加载原始图像
        # print("index=", index)
        # print("self.data[index]=", self.data[index])
        img = imread(self.data[index])  # 读取图像（numpy数组，HWC格式）

        # 调整图像尺寸并中心裁剪
        if size != 0:
            img = self.resize(img, size, size, centerCrop=True)

        # 2. 生成灰度图
        gray_img = img
        if gray_img.ndim == 3:  # 彩色图像（3通道）转灰度
            # 标准RGB转灰度公式：Y = 0.2989R + 0.5870G + 0.1140B
            gray_img = np.dot(gray_img[..., :3], [0.2989, 0.5870, 0.1140])

        # 3. 加载边缘图
        # print("self.edge_data[index]=", self.edge_data[index])
        edge = imread(self.edge_data[index])  # 读取边缘图
        # 调整边缘图尺寸并中心裁剪
        if size != 0:
            edge = self.resize(edge, size, size, centerCrop=True)

        # 4. 加载/生成掩码
        mask = self.load_mask(img, index)

        # 5. 转换为PyTorch张量并返回
        return self.to_tensor(img), self.to_tensor(mask), self.to_tensor(edge), self.to_tensor(gray_img)

    def sym2feature(self, sym):
        """
        （预留功能）从对称图中提取特征线（霍夫变换检测直线）
        Args:
            sym: 对称图（numpy数组）
        Returns:
            x1,y1,x2,y2: 最长直线的端点坐标
        """
        # 读取生成的label图像
        from src.sobel_detection import image_to_sobel
        from PIL import Image

        # 调整图像尺寸到256x256
        original_image = sym
        original_image = cv2.resize(original_image, (256, 256))
        # Sobel边缘检测
        sobel, gray_image1 = image_to_sobel(original_image)

        # 转换为PIL Image格式
        image_tensor_sobel = sobel.squeeze().numpy()
        image_tensor_sobel = (image_tensor_sobel * 255).astype('uint8')
        sobel = Image.fromarray(image_tensor_sobel)
        # 转换为灰度模式
        sobel = sobel.convert('L')
        # 转回numpy数组
        sobel = np.array(sobel)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            sobel,
            rho=1,  # 极坐标rho步长
            theta=np.pi / 180,  # 极坐标theta步长
            threshold=50,  # 累加器阈值
            minLineLength=50,  # 直线最小长度
            maxLineGap=10  # 允许的最大线段间隙
        )

        line_lengths = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 计算直线长度
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                line_lengths.append((line, length))

        # 选择最长的直线
        if line_lengths:
            longest_line = max(line_lengths, key=lambda x: x[1])[0]
            if longest_line is not None:
                x1, y1, x2, y2 = longest_line[0]
                return x1, y1, x2, y2
            else:
                print("No lines detected.")
                return 0, 0, 0, 0
        else:
            return 0, 0, 0, 0

    def load_lmk(self, target_shape, index, size_before, center_crop=True):
        """
        （预留功能）加载并调整关键点坐标
        Args:
            target_shape: 目标图像尺寸 (H, W)
            index: 数据索引
            size_before: 原始尺寸
            center_crop: 是否中心裁剪
        Returns:
            landmarks: 调整后的关键点坐标 [N, 2]
        """
        imgh, imgw = target_shape[0:2]
        # 加载关键点数据（文本文件）
        landmarks = np.genfromtxt(self.landmark_data[index])
        landmarks = landmarks.reshape(self.config.LANDMARK_POINTS, 2)

        # 调整关键点坐标以匹配图像缩放/裁剪
        if self.input_size != 0:
            if center_crop:
                side = np.minimum(size_before[0], size_before[1])
                i = (size_before[0] - side) // 2
                j = (size_before[1] - side) // 2
                # 减去裁剪偏移量
                landmarks[0:self.config.LANDMARK_POINTS, 0] -= j
                landmarks[0:self.config.LANDMARK_POINTS, 1] -= i

            # 缩放坐标到目标尺寸
            landmarks[0:self.config.LANDMARK_POINTS, 0] *= (imgw / side)
            landmarks[0:self.config.LANDMARK_POINTS, 1] *= (imgh / side)

        # 四舍五入为整数坐标
        landmarks = (landmarks + 0.5).astype(np.int16)
        return landmarks

    def load_mask(self, img, index):
        """
        根据掩码类型加载/生成掩码
        掩码类型说明：
        0: 无掩码（全0）
        1: 随机块掩码
        2: 中心块掩码
        3: 外部掩码（从mask_data加载）
        4: 随机选择1/3类型
        5: 训练时50%无掩码，25%随机块，25%外部掩码
        6: 测试模式（固定索引加载外部掩码）
        Args:
            img: 原始图像（用于获取尺寸）
            index: 数据索引
        Returns:
            mask: 掩码图像（numpy数组，HW格式）
        """
        imgh, imgw = img.shape[0:2]  # 图像尺寸
        mask_type = self.mask  # 掩码类型

        # 掩码类型5：训练时动态调整
        if mask_type == 5:
            # 50%概率无掩码，50%概率为类型4
            mask_type = 0 if np.random.uniform(0, 1) >= 0.5 else 4

        # 掩码类型4：随机选择1（随机块）或3（外部掩码）
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # 类型0：无掩码（全0）
        if mask_type == 0:
            return np.zeros((self.config.INPUT_SIZE, self.config.INPUT_SIZE))

        # 类型1：随机块掩码
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # 类型2：中心块掩码
        if mask_type == 2:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2, x=imgw // 4, y=imgh // 4)

        # 类型3：外部掩码（训练时随机选，测试时固定索引）
        if mask_type == 3:
            # 训练模式：随机选择掩码索引
            if self.config.MODE == 1:
                mask_index = random.randint(0, len(self.mask_data) - 1)
            # 测试模式：使用与图像相同的索引
            else:
                mask_index = index

            # 加载掩码图像
            mask = imread(self.mask_data[mask_index])
            # 调整尺寸匹配图像
            mask = self.resize(mask, imgh, imgw)
            # 二值化：>0的像素设为255（掩码区域），否则0
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # 类型6：测试模式（固定索引加载，不中心裁剪）
        if mask_type == 6:
            mask = imread(self.mask_data[index % len(self.mask_data)])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # 二值化
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        """
        将numpy数组转换为PyTorch张量
        步骤：numpy(HWC) → PIL → tensor(CHW) → float
        Args:
            img: numpy数组（HW/HWC格式）
        Returns:
            img_t: 张量（1/3, H, W）
        """
        img = Image.fromarray(img)  # 转换为PIL Image
        img_t = F.to_tensor(img).float()  # 转换为张量并转为float类型
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        """
        调整图像尺寸，可选中心裁剪
        Args:
            img: 输入图像（numpy数组）
            height: 目标高度
            width: 目标宽度
            centerCrop: 是否先中心裁剪为正方形
        Returns:
            img: 调整后的图像（numpy数组）
        """
        imgh, imgw = img.shape[0:2]  # 原始尺寸

        # 中心裁剪为正方形（避免缩放时变形）
        if centerCrop and imgh != imgw:
            side = np.minimum(imgh, imgw)  # 取最小边作为裁剪尺寸
            j = (imgh - side) // 2  # 垂直方向偏移
            i = (imgw - side) // 2  # 水平方向偏移
            img = img[j:j + side, i:i + side, ...]  # 裁剪

        # 调整尺寸（替代废弃的scipy.misc.imresize）
        img = np.array(Image.fromarray(img).resize((height, width)))
        return img

    def load_flist(self, flist):
        """
        灵活加载文件路径列表
        支持三种输入格式：
        1. 列表：直接返回
        2. 目录路径：加载目录下所有jpg/png文件
        3. 文本文件路径：读取文件每行作为路径
        Args:
            flist: 列表/目录路径/文本文件路径
        Returns:
            规范化的文件路径列表
        """
        # 输入为列表：直接返回
        if isinstance(flist, list):
            return flist

        # 输入为字符串：判断是目录还是文件
        if isinstance(flist, str):
            # 目录路径：加载所有jpg/png文件
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()  # 排序保证顺序一致
                return flist

            # 文件路径：读取文本文件（每行一个路径）
            if os.path.isfile(flist):
                try:
                    print("333")
                    # 读取文本文件，编码为utf-8
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except Exception as e:
                    print("444")
                    print(e)
                    # 读取失败则返回自身（单文件路径）
                    return [flist]

        # 无效输入：返回空列表
        return []

    def create_iterator(self, batch_size):
        """
        创建无限迭代器（用于持续训练）
        Args:
            batch_size: 批次大小
        Returns:
            无限生成批次数据的迭代器
        """
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True  # 丢弃最后不完整批次
            )

            for item in sample_loader:
                yield item


def image_transforms(load_size):
    """
    （备用）图像变换函数（未在主流程中使用）
    Args:
        load_size: 调整尺寸
    Returns:
        变换组合：Resize + Normalize
    """
    return transforms.Compose([
        # 调整尺寸（双线性插值）
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        # 归一化到[-1, 1]（GAN常用）
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])