# 模拟受损叶片图像生成工具使用说明

## 功能概述

本工具用于根据真实完整叶片图像（Ground Truth）和对应的黑白掩码图（Mask），生成模拟受损的叶片图像。这在图像修复任务中非常有用，可以创建训练数据或可视化效果。

## 数学原理

该工具实现了以下公式来生成模拟受损图像：

$$I_{masked} = I_{gt} \odot (1 - M) + C \cdot M$$

其中：
- $I_{gt}$: 真实完整叶片图像
- $M$: 归一化后的掩码（0表示完好区域，1表示破损区域）
- $C$: 填充颜色值（白色255、黑色0或灰色127）
- $\odot$: 逐元素乘法操作

## 安装依赖

确保您的环境中已安装以下库：
```bash
pip install opencv-python numpy Pillow
```

## 使用方法

### 单张图像处理

```bash
python script/generate_masked_images.py --gt path/to/ground_truth.jpg --mask path/to/mask.png --output path/to/output.jpg
```

### 批量处理

```bash
python script/generate_masked_images.py --gt ./gt_images/ --mask ./masks/ --output ./masked_results/
```

### 参数说明

| 参数 | 描述 | 默认值 | 选项 |
|------|------|--------|------|
| `--gt` | 真实完整叶片图像路径或目录 | 必需 | 文件路径或目录路径 |
| `--mask` | 掩码图像路径或目录 | 必需 | 文件路径或目录路径 |
| `--output` | 输出路径或目录 | `./masked_images` | 文件路径或目录路径 |
| `--color` | 破损区域填充颜色 | `white` | `white`, `black`, `gray` |
| `--alpha` | 透明度混合系数 | `1.0` | 0.0-1.0之间的浮点数 |
| `--batch` | 强制批量处理模式 | 自动检测 | 布尔标志 |

### 示例

#### 1. 基本用法（白色填充）
```bash
python script/generate_masked_images.py --gt leaf.jpg --mask mask.png --output masked_leaf.jpg
```

#### 2. 使用黑色填充破损区域
```bash
python script/generate_masked_images.py --gt leaf.jpg --mask mask.png --output masked_leaf_black.jpg --color black
```

#### 3. 使用灰色填充并调整透明度
```bash
python script/generate_masked_images.py --gt leaf.jpg --mask mask.png --output masked_leaf_gray.jpg --color gray --alpha 0.8
```

#### 4. 批量处理整个目录
```bash
python script/generate_masked_images.py --gt ./dataset/gt/ --mask ./dataset/masks/ --output ./dataset/masked/
```

## 代码集成

您也可以直接在Python代码中使用此功能：

```python
from script.generate_masked_images import generate_masked_image

# 生成单张模拟受损图像
masked_img = generate_masked_image(
    gt_image_path='leaf.jpg',
    mask_path='mask.png', 
    output_path='result.jpg',
    masked_color='white',
    alpha=1.0
)
```

## 注意事项

1. **图像格式支持**: 支持 JPG, PNG, BMP, TIFF 等常见图像格式
2. **掩码要求**: 掩码应为灰度图像，白色区域(255)表示需要被遮盖的破损区域
3. **尺寸匹配**: 如果掩码和原图尺寸不一致，程序会自动调整掩码大小以匹配原图
4. **文件命名**: 批量处理时，假设GT图像和掩码图像具有相同的文件名

## 在论文配图中的应用建议

对于学术论文中的配图，推荐设置：
- `--color white`: 白色背景使破损区域更加明显
- `--alpha 1.0`: 完全覆盖，清晰展示修复前的状态
- 这样可以在论文中清楚地对比原始图像、受损图像和修复后图像的效果