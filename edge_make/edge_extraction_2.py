import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

'''
使用sobel算子和Prewitt算子对图片进行检测,得到edge图
'''

class SobelConv(nn.Module):
    def __init__(self, device, edge_to_binary=False):
        super(SobelConv, self).__init__()

        # 定义Prewitt算子
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

        # 定义Sobel卷积算子
        self.sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],dtype=torch.float32
        ).view(1, 1, 3, 3).to(device=device)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]
            ],dtype=torch.float32
        ).view(1, 1, 3, 3).to(device=device)
        self.edge_to_binary = edge_to_binary

    def forward(self, img):
        img = img.to(torch.float32)
        # 将传入的x的每个通道进行平均操作转为单通道灰度图
        img_gray = img.mean(dim=1, keepdim=True)
        # 开始sobel卷积操作
        img_x = F.conv2d(img_gray, self.sobel_x, padding=0)
        img_y = F.conv2d(img_gray, self.sobel_y, padding=0)

        #------------加入Prewitt算子---------------begin
        # 计算Prewitt梯度
        prewitt_x = F.conv2d(img_gray, self.prewitt_x, padding=0)
        prewitt_y = F.conv2d(img_gray, self.prewitt_y, padding=0)
        prewitt_grad = torch.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        sobel_grad = torch.sqrt(img_x ** 2 + img_y ** 2)
        # 合并Sobel和Prewitt结果
        combined_grad = sobel_grad + prewitt_grad
        edge_sigmoid_img = torch.sigmoid(F.pad(combined_grad, (1, 1, 1, 1), mode='constant', value=0))      #torch.sigmoid     relu  tanh
        # ------------加入Prewitt算子---------------end

        # # 通过计算这两个输出的平方和的平方根来得到图像的梯度强度并将外圈添加一圈0元素再进行sigmoid操作        -----原来只有sobel的
        # edge_sigmoid_img = torch.sigmoid(F.pad(torch.sqrt(img_x ** 2 + img_y ** 2), (1, 1, 1, 1), mode='constant', value=0))

        # 将结果的每一个像素修正为“0”和“1”的二元输出
        if self.edge_to_binary:
            edge_sigmoid_img = torch.where(edge_sigmoid_img <= 0.55, 0., 1.)
        return edge_sigmoid_img

def imread_uint(path):
    img = cv2.imdecode(np.fromfile(
        path,
        dtype=np.uint8
    ), -1)
    # img = cv2.cvtColor(img, cv2.IMREAD_UNCHANGED)
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
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    # 当保存的文件名为中文时，直接使用“cv2.imwrite”文件名会出现乱码
    cv2.imencode('.png', img)[1].tofile(img_path)
    # cv2.imwrite(img_path, img)

def get_minimum_quantity_of_white_pixel(edge_folder):
    edge_names = [edge_name for edge_name in os.listdir(edge_folder) if edge_name.endswith('.png')]
    min = float('inf')
    for edge_name in edge_names:
        edge = uint2single(
            cv2.cvtColor(
                imread_uint(
                    os.path.join(edge_folder, edge_name)
                ),
                cv2.COLOR_RGB2GRAY
            )
        )
        white_count = np.sum(edge == 1.)
        if white_count < min:
            min = white_count
    return min

def edge_extraction(imgs_folder, edges_save_folder, device):
    os.makedirs(edges_save_folder, exist_ok=True)
    img_names = [img_name for img_name in os.listdir(imgs_folder) if img_name.endswith('.png')]
    sobel_conv = SobelConv(device=device, edge_to_binary=False).to(device=device)
    # 固定整个模型的权重
    for param in sobel_conv.parameters():
        param.requires_grad = False
    for img_name in tqdm(img_names, ncols=100):
        img = single2tensor3(
            uint2single(
                imread_uint(
                    os.path.join(imgs_folder, img_name)
                )
            )
        ).unsqueeze(0).to(device)
        imsave(
            tensor2uint(sobel_conv(img)),
            os.path.join(edges_save_folder, img_name)
        )

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU for computation.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU for computation.")
    edge_extraction(
        imgs_folder='/root/autodl-tmp/data/celeba/img_align_celeba',  # 指向现在的平级目录
        edges_save_folder='/root/autodl-tmp/data/celeba_edge',
        device=device
    )
    # print(get_minimum_quantity_of_white_pixel(
    #     edge_folder='./dataset/256x256/train/imgs_edge'
    # ))



