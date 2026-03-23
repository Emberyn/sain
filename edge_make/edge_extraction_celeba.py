import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

'''
基于原始逻辑优化：使用 sobel 算子和 Prewitt 算子对图片进行批量检测,得到 edge 图
优化点：支持 JPG 读取、DataLoader 多进程加载、GPU 批量推理
'''


class SobelConv(nn.Module):
    def __init__(self, edge_to_binary=False):
        super(SobelConv, self).__init__()
        self.edge_to_binary = edge_to_binary

        # 使用 register_buffer，这样模型 to(device) 时算子会自动跟过去
        self.register_buffer('prewitt_x',
                             torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('prewitt_y',
                             torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, img):
        # 严格保持原代码的计算逻辑
        img_gray = img.mean(dim=1, keepdim=True)

        img_x = F.conv2d(img_gray, self.sobel_x, padding=0)
        img_y = F.conv2d(img_gray, self.sobel_y, padding=0)

        prewitt_x = F.conv2d(img_gray, self.prewitt_x, padding=0)
        prewitt_y = F.conv2d(img_gray, self.prewitt_y, padding=0)

        prewitt_grad = torch.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        sobel_grad = torch.sqrt(img_x ** 2 + img_y ** 2)

        combined_grad = sobel_grad + prewitt_grad
        edge_sigmoid_img = torch.sigmoid(F.pad(combined_grad, (1, 1, 1, 1), mode='constant', value=0))

        if self.edge_to_binary:
            edge_sigmoid_img = torch.where(edge_sigmoid_img <= 0.55, 0., 1.)

        return edge_sigmoid_img


# --- 新增：专门为 CelebA 设计的高效 Dataset ---
class CelebADataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        # 1. 自动适配 .jpg 后缀
        # 2. 排序，保证每次处理顺序一致，方便排错
        self.files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])
        # self.files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])[:100]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        path = os.path.join(self.folder, name)

        # 优化版读取逻辑（等价于原版的 imread_uint + uint2single + single2tensor3）
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 保证内存连续性并转为 Tensor (C, H, W)，归一化到 0-1
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return img, name


def run_extraction():
    # --- 核心路径配置 ---
    imgs_folder = '/root/autodl-tmp/data/celeba/img_align_celeba'
    edges_save_folder = '/root/autodl-tmp/data/celeba_edge'

    # --- 超参数 ---
    batch_size = 64  # GPU 一次处理 64 张，速度起飞
    num_workers = 4  # 开启 4 个 CPU 进程读图

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(edges_save_folder, exist_ok=True)

    # 初始化数据和模型
    dataset = CelebADataset(imgs_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    sobel_conv = SobelConv(edge_to_binary=False).to(device)
    sobel_conv.eval()  # 开启评估模式

    print(f"Total images to process: {len(dataset)}")

    # 不计算梯度，省显存提速
    with torch.no_grad():
        for imgs, names in tqdm(dataloader, desc="Extracting Edges"):
            imgs = imgs.to(device)
            edges = sobel_conv(imgs)

            # 将输出的 Tensor (B, 1, H, W) 转回 numpy array (B, H, W)
            edges_np = (edges.squeeze(1).cpu().numpy() * 255.0).round().astype(np.uint8)

            # 批量保存
            for i in range(len(names)):
                # 原图是 .jpg，但边缘图推荐存为 .png 以防压缩损坏线条
                save_name = names[i].replace('.jpg', '.png')
                save_path = os.path.join(edges_save_folder, save_name)
                cv2.imwrite(save_path, edges_np[i])


if __name__ == '__main__':
    run_extraction()