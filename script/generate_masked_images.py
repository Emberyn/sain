import os
import cv2
import numpy as np
import argparse
from PIL import Image


def generate_masked_image(gt_image_path, mask_path, output_path=None, 
                         masked_color='white', alpha=1.0):
    """
    生成模拟受损叶片图像
    
    Args:
        gt_image_path: 真实完整叶片图像路径 (Ground Truth)
        mask_path: 黑白掩码图路径 (Mask)
        output_path: 输出图像路径，如果为None则返回图像数组
        masked_color: 破损区域颜色 ('white', 'black', 'gray')
        alpha: 透明度混合系数 (0-1)，用于控制原始图像和遮罩颜色的混合程度
        
    Returns:
        masked_image: 生成的模拟受损叶片图像 (numpy array)
    """
    # 读取真实完整叶片图像
    gt_image = cv2.imread(gt_image_path)
    if gt_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {gt_image_path}")
    
    # 读取掩码图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码文件: {mask_path}")
    
    # 确保掩码是单通道
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 调整掩码大小以匹配原始图像
    if mask.shape[:2] != gt_image.shape[:2]:
        mask = cv2.resize(mask, (gt_image.shape[1], gt_image.shape[0]))
    
    # 归一化掩码到 [0, 1] 范围
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # 创建三通道掩码以便与彩色图像运算
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    # 根据指定的颜色设置遮罩颜色
    if masked_color.lower() == 'white':
        color_value = 255
    elif masked_color.lower() == 'black':
        color_value = 0
    elif masked_color.lower() == 'gray':
        color_value = 127
    else:
        raise ValueError("masked_color 必须是 'white', 'black' 或 'gray'")
    
    # 创建纯色背景图像
    color_bg = np.full_like(gt_image, color_value)
    
    # 应用公式: I_masked = I_gt ⊙ (1 - M) + Color * M
    # 其中 ⊙ 表示逐元素乘法
    masked_image = gt_image.astype(np.float32) * (1 - mask_3ch) + color_bg.astype(np.float32) * mask_3ch
    
    # 如果需要透明度混合效果
    if alpha < 1.0:
        blended = gt_image.astype(np.float32) * (1 - alpha * mask_3ch) + color_bg.astype(np.float32) * (alpha * mask_3ch)
        masked_image = blended
    
    # 转换回 uint8 类型
    masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
    
    # 如果指定了输出路径，则保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, masked_image)
        print(f"模拟受损图像已保存到: {output_path}")
    
    return masked_image


def process_batch(gt_dir, mask_dir, output_dir, masked_color='white', alpha=1.0):
    """
    批量处理目录中的所有图像对
    
    Args:
        gt_dir: 包含真实完整叶片图像的目录
        mask_dir: 包含对应掩码图像的目录
        output_dir: 输出目录
        masked_color: 破损区域颜色
        alpha: 透明度混合系数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有支持的图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(image_extensions)]
    
    processed_count = 0
    for gt_filename in gt_files:
        # 构造对应的掩码文件名（假设同名）
        mask_filename = gt_filename
        
        gt_path = os.path.join(gt_dir, gt_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        # 检查掩码文件是否存在
        if not os.path.exists(mask_path):
            print(f"警告: 找不到对应的掩码文件 {mask_path}，跳过 {gt_filename}")
            continue
        
        # 构造输出文件路径
        output_filename = f"masked_{gt_filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # 生成模拟受损图像
            generate_masked_image(gt_path, mask_path, output_path, masked_color, alpha)
            processed_count += 1
            print(f"处理完成: {gt_filename}")
        except Exception as e:
            print(f"处理 {gt_filename} 时出错: {str(e)}")
    
    print(f"\n批量处理完成! 共处理 {processed_count}/{len(gt_files)} 张图像")


def main():
    parser = argparse.ArgumentParser(description='生成模拟受损叶片图像')
    parser.add_argument('--gt', type=str, required=True, help='真实完整叶片图像路径或目录')
    parser.add_argument('--mask', type=str, required=True, help='掩码图像路径或目录')
    parser.add_argument('--output', type=str, default='./masked_images', help='输出路径或目录')
    parser.add_argument('--color', type=str, choices=['white', 'black', 'gray'], 
                       default='white', help='破损区域填充颜色')
    parser.add_argument('--alpha', type=float, default=1.0, 
                       help='透明度混合系数 (0-1)')
    parser.add_argument('--batch', action='store_true', 
                       help='是否进行批量处理（当输入为目录时自动启用）')
    
    args = parser.parse_args()
    
    # 判断是否为批量处理模式
    if os.path.isdir(args.gt) and os.path.isdir(args.mask):
        print("检测到目录输入，启动批量处理模式...")
        process_batch(args.gt, args.mask, args.output, args.color, args.alpha)
    else:
        # 单图像处理模式
        if os.path.isfile(args.gt) and os.path.isfile(args.mask):
            print("启动单图像处理模式...")
            generate_masked_image(args.gt, args.mask, args.output, args.color, args.alpha)
        else:
            print("错误: 请提供有效的图像文件路径或目录路径")
            print("对于单张图片: --gt image.jpg --mask mask.png --output result.jpg")
            print("对于批量处理: --gt ./images/ --mask ./masks/ --output ./results/")


if __name__ == "__main__":
    main()