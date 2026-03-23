import cv2
import os

'''  将单通道图片改为三通道  '''

# 指定图片文件夹路径
input_folder = '../data/Place/masked_256/20_50'
output_folder = '../data/masked_img/test_edge_3ch'

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的每张图片
for filename in os.listdir(input_folder):
    # 构建文件路径
    img_path = os.path.join(input_folder, filename)

    # 读取单通道图片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为单通道灰度图像

    if img is not None:
        # 将单通道图像转换为三通道图像
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存为三通道图片
        cv2.imwrite(output_path, img_3ch)
    else:
        print(f"无法读取图片: {filename}")
