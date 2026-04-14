# 快速入门：生成模拟受损叶片图像

## 🚀 5分钟快速上手

### 第一步：准备数据

确保您有以下两种图像：
1. **真实完整叶片图像** (Ground Truth) - 例如：`leaf_001.jpg`
2. **对应的黑白掩码图** (Mask) - 例如：`mask_001.png`

> 💡 **掩码图说明**：白色区域(255)表示破损/需要修复的区域，黑色区域(0)表示完好区域

### 第二步：运行命令

#### 方式一：单张图像处理

```bash
python script/generate_masked_images.py --gt leaf_001.jpg --mask mask_001.png --output masked_leaf_001.jpg
```

#### 方式二：批量处理整个文件夹

假设您的文件结构如下：
```
dataset/
├── gt_images/      # 所有完整叶片图像
│   ├── leaf_001.jpg
│   ├── leaf_002.jpg
│   └── ...
└── masks/          # 对应的掩码图像（文件名需相同）
    ├── leaf_001.jpg
    ├── leaf_002.jpg
    └── ...
```

运行批量处理：
```bash
python script/generate_masked_images.py --gt dataset/gt_images/ --mask dataset/masks/ --output dataset/masked_results/
```

### 第三步：查看结果

处理完成后，在输出目录中查看生成的模拟受损叶片图像。

---

## 🎨 常用参数配置

### 论文配图推荐设置

```bash
# 白色背景（最清晰，推荐用于论文）
python script/generate_masked_images.py \
    --gt leaf.jpg \
    --mask mask.png \
    --output result_white.jpg \
    --color white \
    --alpha 1.0
```

### 其他视觉效果

```bash
# 黑色背景
python script/generate_masked_images.py \
    --gt leaf.jpg \
    --mask mask.png \
    --output result_black.jpg \
    --color black

# 灰色背景
python script/generate_masked_images.py \
    --gt leaf.jpg \
    --mask mask.png \
    --output result_gray.jpg \
    --color gray

# 半透明效果
python script/generate_masked_images.py \
    --gt leaf.jpg \
    --mask mask.png \
    --output result_semi.jpg \
    --color white \
    --alpha 0.7
```

---

## 💻 Python代码调用

如果您想在Python脚本中直接使用：

```python
from script.generate_masked_images import generate_masked_image

# 基本用法
result = generate_masked_image(
    gt_image_path='leaf.jpg',
    mask_path='mask.png',
    output_path='masked_leaf.jpg',
    masked_color='white',
    alpha=1.0
)

print(f"生成完成！图像尺寸: {result.shape}")
```

---

## ❓ 常见问题

### Q1: 掩码图和原图尺寸不一样怎么办？
A: 程序会自动调整掩码大小以匹配原图，无需手动处理。

### Q2: 支持哪些图像格式？
A: 支持 JPG, PNG, BMP, TIFF 等常见格式。

### Q3: 批量处理时文件名必须完全一样吗？
A: 是的，GT图像和掩码图像的文件名应该相同。例如：
- GT: `leaf_001.jpg` → Mask: `leaf_001.jpg`

### Q4: 如何选择合适的填充颜色？
A: 
- **白色**: 最适合论文配图，对比度高
- **黑色**: 适合深色背景展示
- **灰色**: 中性效果，较为柔和

### Q5: alpha参数有什么用？
A: 控制透明度混合程度：
- `alpha=1.0`: 完全覆盖（默认）
- `alpha=0.5`: 半透明效果
- `alpha=0.0`: 无变化

---

## 📊 实际应用场景

### 场景1: 创建训练数据集

```bash
# 为模型训练生成大量模拟受损样本
python script/generate_masked_images.py \
    --gt ./train/gt/ \
    --mask ./train/masks/ \
    --output ./train/masked/ \
    --color white
```

### 场景2: 论文对比图制作

生成三组对比图：
1. 原始完整叶片
2. 模拟受损叶片（本工具生成）
3. 模型修复后的叶片

```bash
# 生成模拟受损版本用于对比
python script/generate_masked_images.py \
    --gt original_leaf.jpg \
    --mask damage_mask.png \
    --output paper_comparison/masked.jpg \
    --color white \
    --alpha 1.0
```

### 场景3: 效果演示

```bash
# 生成多种效果进行展示
for color in white black gray; do
    python script/generate_masked_images.py \
        --gt demo_leaf.jpg \
        --mask demo_mask.png \
        --output demo_${color}.jpg \
        --color ${color}
done
```

---

## 🔗 相关资源

- 📖 [详细文档](MASKED_IMAGE_GENERATION_README.md)
- 🧪 [测试脚本](test_masked_generation.py)
- 💡 [使用示例](example_usage.py)

---

## ✨ 小贴士

1. **备份原图**: 处理前建议备份原始图像
2. **检查掩码**: 确保掩码图中白色区域确实对应要遮盖的部分
3. **统一命名**: 批量处理时使用统一的文件命名规范
4. **预览效果**: 先用单张图片测试，确认效果后再批量处理

---

**祝您使用愉快！** 🎉

如有问题，请查看详细文档或运行 `python script/generate_masked_images.py --help` 获取帮助。