"""
测试脚本：验证模拟受损叶片生成功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from generate_masked_images import generate_masked_image


def create_test_data():
    """创建测试用的简单图像和掩码"""
    # 创建测试目录
    os.makedirs('./test_data', exist_ok=True)
    
    # 创建一个简单的彩色测试图像 (100x100)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[20:80, 20:80] = [0, 255, 0]  # 绿色方块
    test_image[40:60, 40:60] = [255, 0, 0]  # 红色小方块
    
    # 保存测试图像
    import cv2
    cv2.imwrite('./test_data/test_gt.png', test_image)
    
    # 创建一个简单的掩码 (白色圆形区域表示破损)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 25, 255, -1)  # 白色圆圈
    cv2.imwrite('./test_data/test_mask.png', mask)
    
    print("测试数据已创建:")
    print("- 真实图像: ./test_data/test_gt.png")
    print("- 掩码图像: ./test_data/test_mask.png")


def test_single_generation():
    """测试单张图像生成"""
    print("\n=== 测试单张图像生成 ===")
    
    # 确保测试数据存在
    if not os.path.exists('./test_data/test_gt.png'):
        create_test_data()
    
    # 测试不同颜色选项
    colors = ['white', 'black', 'gray']
    
    for color in colors:
        output_path = f'./test_data/masked_{color}.png'
        try:
            result = generate_masked_image(
                gt_image_path='./test_data/test_gt.png',
                mask_path='./test_data/test_mask.png',
                output_path=output_path,
                masked_color=color,
                alpha=1.0
            )
            print(f"✓ {color.capitalize()} 填充测试通过: {output_path}")
        except Exception as e:
            print(f"✗ {color.capitalize()} 填充测试失败: {str(e)}")


def test_alpha_blending():
    """测试透明度混合效果"""
    print("\n=== 测试透明度混合 ===")
    
    if not os.path.exists('./test_data/test_gt.png'):
        create_test_data()
    
    alphas = [0.3, 0.6, 0.9]
    
    for alpha in alphas:
        output_path = f'./test_data/masked_alpha_{alpha}.png'
        try:
            result = generate_masked_image(
                gt_image_path='./test_data/test_gt.png',
                mask_path='./test_data/test_mask.png',
                output_path=output_path,
                masked_color='white',
                alpha=alpha
            )
            print(f"✓ Alpha={alpha} 测试通过: {output_path}")
        except Exception as e:
            print(f"✗ Alpha={alpha} 测试失败: {str(e)}")


def main():
    print("开始测试模拟受损叶片生成功能...")
    
    # 创建测试数据
    create_test_data()
    
    # 运行各项测试
    test_single_generation()
    test_alpha_blending()
    
    print("\n🎉 所有测试完成!")
    print("查看 ./test_data/ 目录中的结果图像")
    print("\n提示: 您可以使用以下命令进行实际使用:")
    print("python script/generate_masked_images.py --gt your_image.jpg --mask your_mask.png --output result.jpg")


if __name__ == "__main__":
    main()