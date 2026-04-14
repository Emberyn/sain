"""
使用示例：演示如何使用模拟受损叶片生成工具
"""
import os
from generate_masked_images import generate_masked_image, process_batch


def example_single_image():
    """单张图像处理示例"""
    print("=== 单张图像处理示例 ===")
    
    # 假设您有以下文件（请根据实际情况修改路径）
    gt_image = "path/to/your/leaf_image.jpg"      # 真实完整叶片图像
    mask_image = "path/to/your/mask_image.png"     # 对应的掩码图像
    output_image = "path/to/output/masked_leaf.jpg" # 输出路径
    
    # 检查文件是否存在（在实际使用时注释掉这部分检查）
    if not os.path.exists(gt_image):
        print(f"注意: 示例文件 {gt_image} 不存在，请替换为实际文件路径")
        return
    
    if not os.path.exists(mask_image):
        print(f"注意: 示例文件 {mask_image} 不存在，请替换为实际文件路径")
        return
    
    try:
        # 生成白色填充的模拟受损图像（推荐用于论文配图）
        result = generate_masked_image(
            gt_image_path=gt_image,
            mask_path=mask_image,
            output_path=output_image,
            masked_color='white',  # 白色填充破损区域
            alpha=1.0              # 完全不透明
        )
        print(f"✓ 成功生成模拟受损图像: {output_image}")
        
    except Exception as e:
        print(f"✗ 生成失败: {str(e)}")


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 假设您有以下目录（请根据实际情况修改路径）
    gt_directory = "./dataset/gt_images/"      # 包含所有真实图像的目录
    mask_directory = "./dataset/masks/"        # 包含对应掩码的目录  
    output_directory = "./dataset/masked_results/"  # 输出目录
    
    # 检查目录是否存在（在实际使用时注释掉这部分检查）
    if not os.path.exists(gt_directory):
        print(f"注意: 示例目录 {gt_directory} 不存在，请替换为实际目录路径")
        return
        
    if not os.path.exists(mask_directory):
        print(f"注意: 示例目录 {mask_directory} 不存在，请替换为实际目录路径")
        return
    
    try:
        # 批量处理所有图像对
        process_batch(
            gt_dir=gt_directory,
            mask_dir=mask_directory,
            output_dir=output_directory,
            masked_color='white',  # 白色填充
            alpha=1.0              # 完全不透明
        )
        print(f"✓ 批量处理完成，结果保存在: {output_directory}")
        
    except Exception as e:
        print(f"✗ 批量处理失败: {str(e)}")


def example_different_styles():
    """不同风格效果示例"""
    print("\n=== 不同视觉效果示例 ===")
    
    gt_image = "path/to/your/leaf_image.jpg"
    mask_image = "path/to/your/mask_image.png"
    
    # 检查文件是否存在
    if not os.path.exists(gt_image) or not os.path.exists(mask_image):
        print("注意: 请提供实际的图像文件路径来查看不同效果")
        return
    
    styles = [
        {'color': 'white', 'alpha': 1.0, 'desc': '白色完全覆盖（推荐用于论文）'},
        {'color': 'black', 'alpha': 1.0, 'desc': '黑色完全覆盖'},
        {'color': 'gray', 'alpha': 1.0, 'desc': '灰色完全覆盖'},
        {'color': 'white', 'alpha': 0.7, 'desc': '白色半透明覆盖'},
    ]
    
    for i, style in enumerate(styles):
        output_path = f"./style_example_{i+1}.jpg"
        try:
            result = generate_masked_image(
                gt_image_path=gt_image,
                mask_path=mask_image,
                output_path=output_path,
                masked_color=style['color'],
                alpha=style['alpha']
            )
            print(f"✓ {style['desc']}: {output_path}")
        except Exception as e:
            print(f"✗ {style['desc']} 失败: {str(e)}")


def main():
    print("模拟受损叶片生成工具 - 使用示例\n")
    
    print("本脚本展示了如何使用 generate_masked_images.py 工具")
    print("=" * 50)
    
    # 运行各个示例
    example_single_image()
    example_batch_processing() 
    example_different_styles()
    
    print("\n" + "=" * 50)
    print("使用说明:")
    print("1. 单张图像: python script/generate_masked_images.py --gt image.jpg --mask mask.png --output result.jpg")
    print("2. 批量处理: python script/generate_masked_images.py --gt ./images/ --mask ./masks/ --output ./results/")
    print("3. 更多选项: python script/generate_masked_images.py --help")


if __name__ == "__main__":
    main()