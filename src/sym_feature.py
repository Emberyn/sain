import os
import cv2
import numpy as np
import torch
import torch.nn as nn
 #  该文件用于测试对于找到的对称轴进行特征翻转

def flip_image(image, x1, y1, x2, y2):   # image类型：<class 'torch.Tensor'>  torch.Size([1, 1, 256, 256])，返回flipped_image类型相同
    # 获取图像的维度
    batch_size, channels, height, width = image.shape

    # 计算直线方程的系数
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # 创建坐标网格
    y0, x0 = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    # 计算系数 k
    k = -2 * (A * x0 + B * y0 + C) / (A ** 2 + B ** 2)

    # 计算对称点的坐标
    x_prime = (x0 + k * A).round().long()
    y_prime = (y0 + k * B).round().long()

    # 限制坐标范围在有效值内
    x_prime = torch.clamp(x_prime, 0, width - 1)
    y_prime = torch.clamp(y_prime, 0, height - 1)

    # 创建一个新的张量用于存储翻转后的结果
    flipped_image = torch.zeros_like(image)

    # 使用反向映射方式来填充图像
    for b in range(batch_size):
        for c in range(channels):
            flipped_image[b, c, y0, x0] = image[b, c, y_prime, x_prime]

    return flipped_image

# 输入和输出文件夹路径
#读取预测对称轴图片输入
input_folder = r'E:\image_inpainting\face_sym_predict_5\checkpoint\face\results_3k_test\results'
#读取叶子图片输入
input_yezi_folder=r"E:\image_inpainting\face_sym_predict_5\checkpoint\face\results_3k_test\test"
#输出最后对称后的特征文件夹
output_folder = r'E:\image_inpainting\face_sym_predict_5\checkpoint\face\results_3k_test\feature_sym'
#对称轴文件夹保存路径
folder_sym_img_path=r"E:\image_inpainting\face_sym_predict_5\checkpoint\face\results_3k_test\sym_img"


# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(folder_sym_img_path):
    os.makedirs(folder_sym_img_path)


def get_image_files(folder):
    """获取文件夹中的所有图片文件"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[-1].lower() in image_extensions]

# 遍历输入文件夹中的所有图片文件
#for filename in os.listdir(input_folder):
def load_images(input_folder, input_yezi_folder):

    files1 = get_image_files(input_folder)
    files2 = get_image_files(input_yezi_folder)

    # 确保两个文件夹中有相同数量的图片文件
    assert len(files1) == len(files2), "两个文件夹中的图片数量不一致"

    for file1, file2 in zip(files1, files2):

        # 读取生成的label图像
        from src.sobel_detection import image_to_sobel
        from PIL import Image
        original_image = cv2.imread(file1)
        original_image = cv2.resize(original_image, (256, 256))
        sobel, gray_image1 = image_to_sobel(original_image)

        # 调整tensor的形状为(height, width)，并将其转换为PIL Image
        image_tensor_sobel = sobel.squeeze().numpy()
        image_tensor_sobel = (image_tensor_sobel * 255).astype('uint8')
        sobel = Image.fromarray(image_tensor_sobel)
        #print("sobel.shape=",sobel.shape)
        print("sobel.size=", sobel.size)
        # 将图像转换为灰度模式
        sobel = sobel.convert('L')
        # 将PIL图像转换为NumPy数组
        sobel = np.array(sobel)

        # 读取叶子图片，以灰度图读取
        image_yezi = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
        image_yezi = cv2.resize(image_yezi, (256, 256))
        image_yezi_fu=image_yezi

        # 执行霍夫变换检测直线
        lines = cv2.HoughLinesP(sobel, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)  #sobel

        line_lengths = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                line_lengths.append((line, length))

        if line_lengths:
            # 选择长度最长的那条直线
            longest_line = max(line_lengths, key=lambda x: x[1])[0]

            # 创建黑色背景
            black_bg = np.zeros_like(original_image)

            # 绘制最长的那条直线在黑色背景上
            if longest_line is not None:
                print("longest_line=", longest_line)
                x1, y1, x2, y2 = longest_line[0]
                #计算直线斜率
                m = (y2 - y1) / (x2 - x1)
                # 计算截距 b
                b = y1 - m * x1
                y_128 = m*128 + b
                sym = [128,y_128,m]
                sym = torch.tensor(sym).view(batchsize, 3)      # 将sym转为torch.Size([1, 3])  1：表示batchsize
                image_yezi = image_yezi.reshape(1, 1, 256, 256)
                print("image_yezi.shape=", image_yezi.shape)         # image_yezi.shape= (1, 1, 256, 256)
                print("type(image_yezi)=", type(image_yezi))         # type(image_yezi)= <class 'numpy.ndarray'>
                image_yezi = torch.from_numpy(image_yezi)
                print("image_yezi.shape=", image_yezi.shape)         # image_yezi.shape= torch.Size([1, 1, 256, 256])
                print("type(image_yezi)=", type(image_yezi))         # type(image_yezi)= <class 'torch.Tensor'>
                #x_v = sym_.forward(image_yezi, sym)
                x_v = flip_image(image_yezi, x1, y1, x2, y2)
                print("x_v.shape=", x_v.shape)                      # x_v.shape= torch.Size([1, 1, 256, 256])
                print("type(x_v)=", type(x_v))                      # type(x_v)= <class 'torch.Tensor'>
                x_v = x_v.squeeze().numpy()
                print("x_v.shape=", x_v.shape)                      # x_v.shape= (256, 256)
                print("type(x_v)=", type(x_v))                      # type(x_v)= <class 'numpy.ndarray'>
                cv2.line(image_yezi_fu, (x1, y1), (x2, y2), (255, 255, 255), 5)    # image_yezi_fu  black_bg

            # 保存结果图像
            filename = file2.split('\\')[-1]
            print(f"Processed: {filename}")
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, x_v)

            output_mask_img_path = os.path.join(folder_sym_img_path, 'sym_{}.jpg'.format(filename.split('.')[0]))
            cv2.imwrite(output_mask_img_path, image_yezi_fu)    # image_yezi_fu  black_bg
            # #sobel.save(output_mask_img_path)

        else:
            print("No lines detected.")



load_images(input_folder,input_yezi_folder)
print("All images processed and saved.")
