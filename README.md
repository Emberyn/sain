# SAIN (Symmetry-Aware Image Inpainting Network)

基于对称性感知的图像修复网络，专门用于叶片图像的修复任务。

## 项目简介

本项目实现了一个深度学习驱动的图像修复系统，特别关注利用图像的对称性特征来提升修复质量。主要应用于植物叶片的损伤修复和分析。

## 核心功能

- 🖼️ **图像修复**: 基于深度学习的图像缺失区域修复
- 📐 **对称性感知**: 专门的对称特征提取模块
- 🔍 **边缘检测**: Sobel算子边缘提取和生成
- 🌿 **叶片专用**: 针对植物叶片优化的修复算法
- 🎨 **数据生成**: 模拟受损叶片图像生成工具

## 新增工具：模拟受损叶片生成

### 功能说明

`script/generate_masked_images.py` - 根据真实完整叶片图像和掩码图生成模拟受损叶片图像

**数学原理**:
```
I_masked = I_gt ⊙ (1 - M) + C · M
```
其中 I_gt 为原图，M 为归一化掩码，C 为填充颜色

### 快速开始

```bash
# 单张图像处理
python script/generate_masked_images.py --gt leaf.jpg --mask mask.png --output masked_leaf.jpg

# 批量处理
python script/generate_masked_images.py --gt ./images/ --mask ./masks/ --output ./results/

# 查看帮助
python script/generate_masked_images.py --help
```

### 使用示例

```python
from script.generate_masked_images import generate_masked_image

# 生成白色填充的模拟受损图像（推荐用于论文配图）
result = generate_masked_image(
    gt_image_path='leaf.jpg',
    mask_path='mask.png', 
    output_path='masked_leaf.jpg',
    masked_color='white',  # 'white', 'black', 'gray'
    alpha=1.0              # 透明度系数 0-1
)
```

详细使用说明请参考: [模拟受损叶片生成工具文档](script/MASKED_IMAGE_GENERATION_README.md)

## 项目结构

```
sain/
├── src/                    # 核心源代码
│   ├── models.py          # 模型定义
│   ├── networks.py        # 网络架构
│   ├── dataset.py         # 数据集处理
│   ├── loss.py           # 损失函数
│   ├── sym_feature.py    # 对称特征提取
│   └── ...
├── script/                 # 工具脚本
│   ├── generate_masked_images.py  # 🆕 模拟受损叶片生成
│   ├── generate_masks.py         # 掩码生成
│   └── flist.py                 # 文件列表处理
├── checkpoint/            # 模型检查点
├── frontend/             # Web前端界面
└── app.py               # Flask应用入口
```

## 环境要求

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- NumPy, Pillow, Scikit-image

详细依赖请查看 `environment.yml`

## 使用方法

### 训练模型
```bash
python train.py
```

### 测试模型
```bash
python test.py
```

### 启动Web应用
```bash
python app.py
```

## 技术特点

- **对称性感知**: 利用叶片的自然对称特性提升修复质量
- **多尺度处理**: 支持不同分辨率的图像输入
- **灵活掩码**: 支持多种类型的破损掩码生成
- **可视化界面**: 提供Web界面进行交互式操作

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件