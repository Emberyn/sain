import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SobelConv(nn.Module):
    def __init__(self, device, edge_to_binary=False):
        super(SobelConv, self).__init__()

        self.prewitt_x = torch.tensor(
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(device=device)
        self.prewitt_y = torch.tensor(
            [[-1, -1, -1],
             [0, 0, 0],
             [1, 1, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(device=device)

        self.sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(device=device)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]
             ], dtype=torch.float32
        ).view(1, 1, 3, 3).to(device=device)
        self.edge_to_binary = edge_to_binary

    def forward(self, img):
        img = img.to(torch.float32)
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


def imread_uint(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def uint2single(img):
    return np.float32(img / 255.)


def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def tensor2uint(img):
    # tensor转回numpy并恢复0-255范围
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def imsave(img, img_path):
    # 保存图片，处理中文路径
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imencode('.png', img)[1].tofile(img_path)


def process_flist(flist_path, device):
    """
    处理.flist文件，对文件中的每张图片进行边缘检测
    :param flist_path: .flist文件的绝对路径
    :param device: 计算设备
    """
    # 获取flist文件所在目录的上级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(flist_path)))
    print(f"基础目录: {base_dir}")

    # 读取.flist文件中的所有图片路径
    with open(flist_path, 'r', encoding='utf-8') as f:
        img_paths = f.read().splitlines()

    print(f"找到 {len(img_paths)} 张图片需要处理")

    # 初始化边缘检测模块
    sobel_conv = SobelConv(device=device, edge_to_binary=False).to(device=device)
    for param in sobel_conv.parameters():
        param.requires_grad = False

    # 创建进度条
    progress_bar = tqdm(img_paths, ncols=100)

    # 处理每张图片
    for img_relative_path in progress_bar:
        try:
            # 获取绝对输入路径
            img_abs_path = os.path.join(base_dir, img_relative_path)

            # 生成输出路径：替换total_train为total_train_sobel
            output_relative_path = img_relative_path.replace('total_train', 'total_train_sobel')

            # 分离文件名和扩展名，然后修改扩展名为.png
            root, ext = os.path.splitext(output_relative_path)
            output_relative_path_png = root + ".png"

            output_abs_path = os.path.join(base_dir, output_relative_path_png)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)

            # 更新进度条描述
            progress_bar.set_description(f"处理: {os.path.basename(img_relative_path)}")

            # 检查文件是否存在
            if not os.path.exists(img_abs_path):
                print(f"\n文件不存在: {img_abs_path}")
                continue

            # 读取图片
            img = imread_uint(img_abs_path)

            # 转换为tensor并处理
            img_tensor = single2tensor3(uint2single(img)).unsqueeze(0).to(device)
            edge_img = sobel_conv(img_tensor)

            # 保存边缘检测结果
            imsave(tensor2uint(edge_img), output_abs_path)

        except Exception as e:
            print(f"\n处理图片时出错: {img_relative_path}")
            print(f"错误信息: {str(e)}")

    print("处理完成!")


if __name__ == '__main__':
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA可用，使用GPU进行计算")
    else:
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU进行计算")

    # 设置.flist文件路径
    flist_path = os.path.abspath(r'../checkpoint/place_train.flist')
    print(f"使用flist文件: {flist_path}")

    # 处理flist中的所有图片
    process_flist(flist_path, device)