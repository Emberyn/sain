# 原图片文件夹路径
"""检测sobel"""
import os
from PIL import Image
from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray

import cv2
import numpy as np
from matplotlib import pyplot as plt

def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def image_to_edge(image, sigma):

    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma,low_threshold=0.2,high_threshold=0.25)))
    #edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))     #使用默认阈值，两个阈值是0.1和0.2

    gray_image = image_to_tensor()(Image.fromarray(gray_image))


    return edge, gray_image

def image_to_sobel(image):

    gray_image = rgb2gray(np.array(tensor_to_image()(image)))

    # Sobel边缘检测器 -----此时应用这个
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=1, scale=1., delta=0)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=1, scale=1., delta=0)
    sobel_edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 将边缘强度转换为布尔类型数组，边缘像素为True，非边缘像素为False。 使用Sobel算子检测到的边缘强度的最大值的一定比例作为默认阈值
    threshold_ratio = 0.1  # 阈值比例，可以根据需要进行调整
    threshold_value = threshold_ratio * np.max(sobel_edges)
    sobel_edges_bool = sobel_edges > threshold_value
    #sobel_edges_bool = sobel_edges > 0.2

    # 将布尔类型数组转换为PyTorch张量
    edge = image_to_tensor()(Image.fromarray(sobel_edges_bool))

    gray_image = image_to_tensor()(Image.fromarray(gray_image))



    return edge, gray_image

# folder_path = r"E:\image_inpainting\face_sym_predict_5\checkpoint\face\result_comparison\7\results\results"
# #sobel文件夹保存路径
# folder_sobel_img_path=r"E:\image_inpainting\face_sym_predict_5\checkpoint\face\result_comparison\7\results\sobel"
# # 获取文件夹中的所有文件名
# file_names = os.listdir(folder_path)
#
# i = 0
# # 遍历文件夹中的每个文件
# for file_name in file_names:
#     # 构造完整的文件路径
#     file_path = os.path.join(folder_path, file_name)
#
#     # 打开图像
#     #image=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)         #灰度图打开，对应使用sobel时使用
#     image = cv2.imread(file_path)
#
#     #sobel图片文件保存路径
#     output_mask_img_path = os.path.join(folder_sobel_img_path, 'sobel_{}.jpg'.format(file_name.split('.')[0]))
#
#     # canny图片文件保存路径
#     output_mask_canny_img_path = os.path.join(folder_sobel_img_path, 'canny_{}.jpg'.format(file_name.split('.')[0]))
#
#     # #canny---------------begin
#     # edge, gray_image = image_to_edge(image, 2)
#     #
#     # # 调整tensor的形状为(height, width)，并将其转换为PIL Image
#     # image_tensor = edge.squeeze().numpy()
#     # edge = Image.fromarray((image_tensor * 255).astype('uint8'))
#     # # 保存图像
#     # edge.save(output_mask_canny_img_path)
#     # # canny---------------end
#
#
#     # --------------------------Sobel边缘检测---------------begin---------------
#     sobel, gray_image1 = image_to_sobel(image)
#     # sobel=transforms.ToTensor()(sobel)
#     #print("sobel=", sobel)
#     #print("gray_image1=", gray_image1)
#
#     # 调整tensor的形状为(height, width)，并将其转换为PIL Image
#     image_tensor_sobel = sobel.squeeze().numpy()
#     image_tensor_sobel=(image_tensor_sobel * 255).astype('uint8')
#     sobel = Image.fromarray(image_tensor_sobel)
#     #print("sobel.size=", sobel.size)
#     #print("sobel=",sobel)
#     #print("type.sobel=", type(sobel))
#
#     # 保存图像
#     sobel.save(output_mask_img_path)
#
#
#
#     # plt.axis('off')
#     # plt.imshow(sobel_edges, cmap='gray')
#     # # 保存图像
#     # plt.savefig(output_mask_img_path, bbox_inches='tight',
#     #             pad_inches=0)
#     # i+=1
#     # print("i=",i)
#     # plt.close()
#     # --------------------------Sobel边缘检测---------------end---------------