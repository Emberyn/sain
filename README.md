# 寻脉溯源・智能重构 —— 面向大豆叶片损伤的图像修复系统

## 3. 作品简介

本项目基于自研 SAIN 深度学习架构，构建大豆叶片损伤智能修复平台。系统支持交互式遮罩标注，利用生成对抗网络自动修复损伤，精准还原叶片的完整叶脉与纹理。采用 Flask+PyTorch+OpenCV 技术栈，为精准农业提供高效、直观的智能化技术支撑。

## 项目简介

本项目实现了一个深度学习驱动的图像修复系统，特别关注利用图像的对称性特征来提升修复质量。主要应用于植物叶片的损伤修复和分析。

## 核心功能

- 🖼️ **交互式遮罩标注**: 支持画笔、橡皮擦、多种预设形状（圆形、椭圆、矩形、三角形）绘制损伤区域
-  **批量处理**: 支持同时上传最多20张叶片图片进行批量修复
- 🤖 **智能修复**: 基于生成对抗网络(GAN)自动修复损伤区域，精准还原叶脉结构
- 📐 **叶脉提取**: SAIN结构感知叶脉图生成，辅助农业专家分析
- 🔄 **遮罩合并**: 支持手绘遮罩与上传黑白掩码文件的智能合并
- 📥 **一键下载**: 批量结果打包为ZIP格式下载

## 测试数据说明

### 数据集结构

本项目包含完整的训练和测试数据集，位于 `DataSet_padded_enhance/` 目录：

```
DataSet_padded_enhance/
├── 3ch/                              # 三通道数据集
│   ├── train_padded_enhance_edge_3ch/  # 训练集边缘图 (10000张)
│   └── test_padded_enhance_edge_3ch/   # 测试集边缘图 (1932张)
├── mask/                             # 掩码数据集
│   ├── mask_train/                     # 训练集掩码 (6000张)
│   ├── mask_test/                      # 测试集掩码 (2000张)
│   ├── mask_train_10_20/              # 不同破损程度掩码
│   └── mask_test_10_20/
├── train_padded_enhance/             # 训练集原始图像 (10000张)
└── test_padded_enhance/              # 测试集原始图像
```

### 数据特点

- **图像格式**: PNG/JPG格式，统一分辨率处理
- **掩码类型**: 黑白二值掩码，白色区域表示损伤/破损部位
- **边缘图**: 基于Sobel算子提取的叶脉边缘结构图
- **数据增强**: 经过padded和enhance处理，提升模型泛化能力
- **破损模拟**: 提供多种破损程度(10-20%, 40-60%)的测试数据

### 使用测试数据

```bash
# 使用测试集进行模型评估
python test.py --input ./DataSet_padded_enhance/test_padded_enhance/ \
               --mask ./DataSet_padded_enhance/mask/mask_test/ \
               --edge ./DataSet_padded_enhance/3ch/test_padded_enhance_edge_3ch/ \
               --output ./results/
```

### 自定义测试

您也可以使用自己的大豆叶片图片进行测试：

1. 准备原始叶片图片（PNG/JPG格式）
2. 准备对应的黑白掩码文件（白色=损伤区域）
3. 通过Web界面上传或命令行指定路径
4. 系统自动完成修复并输出结果

## 工具脚本：模拟受损叶片生成

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

## 环境准备

### 系统要求

1. **操作系统**: Windows 10/11 (推荐) 或 Linux/macOS
2. **开发工具**: PyCharm 或其他 Python IDE
3. **环境管理**: Anaconda 或 Miniconda
4. **GPU支持** (可选): NVIDIA GPU + CUDA 11.0+ (可显著提升修复速度)

## 快速开始

### 步骤1: 克隆项目代码

```bash
git clone https://github.com/Emberyn/sain.git
cd sain
```

### 步骤2: 创建虚拟环境

```bash
# 使用 conda 创建环境
conda env create -f environment.yml

# 激活环境
conda activate sain_env
```

### 步骤3: 下载模型权重

本项目需要预训练模型权重才能运行，请确保 `checkpoint/` 目录下包含以下文件：

```
checkpoint/
├── EdgeModel_dis.pth      # 边缘检测模型-判别器
├── EdgeModel_gen.pth      # 边缘检测模型-生成器
├── InpaintingModel_dis.pth # 修复模型-判别器
├── InpaintingModel_gen.pth # 修复模型-生成器
└── config.yml             # 配置文件
```

> **注意**: 模型权重文件较大，可能需要通过其他方式获取或联系项目作者。

### 步骤4: 启动Web应用

```bash
python app.py
```

启动成功后，终端会显示类似以下信息：

```
Loading models to cuda...
Models loaded successfully and ready for inference!
 * Running on http://127.0.0.1:5000
```

### 步骤5: 访问系统

在浏览器中打开: **http://127.0.0.1:5000**

即可使用大豆叶片损伤智能修复系统！

## 其他运行模式

### 训练模型

如果您想重新训练模型：

```bash
# 训练边缘检测模型
python train.py

# 或直接运行主程序（mode=1 为训练模式）
python main.py
```

训练配置可在 `checkpoint/config.yml` 中调整。

### 测试模型

使用命令行进行批量测试：

```bash
# 使用默认测试集
python test.py

# 或指定测试数据路径
python test.py --input ./DataSet_padded_enhance/test_padded_enhance/ \
               --mask ./DataSet_padded_enhance/mask/mask_test/ \
               --edge ./DataSet_padded_enhance/3ch/test_padded_enhance_edge_3ch/ \
               --output ./results/
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