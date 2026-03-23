import os
import yaml  # 用于解析YAML配置文件


class Config(dict):
    """
    配置管理类（继承自dict）
    功能：
    1. 从YAML文件加载配置
    2. 支持属性式访问（config.LR 等价于 config['LR']）
    3. 配置项优先级：YAML文件配置 > 默认配置（DEFAULT_CONFIG）
    4. 提供配置打印功能，便于调试和日志记录
    """

    def __init__(self, config_path):
        """
        初始化配置类，加载YAML配置文件
        Args:
            config_path: YAML配置文件路径（如configs/train.yaml）
        """
        super().__init__()
        # 读取YAML配置文件内容
        with open(config_path, 'r') as f:
            self._yaml = f.read()  # 保存原始YAML文本（用于打印）
            self._dict = yaml.safe_load(self._yaml)  # 解析为字典
            # 自动添加配置文件所在路径到配置中（便于加载相对路径资源）
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        """
        重写属性访问方法，支持config.name形式访问配置项
        优先级：YAML文件配置 > DEFAULT_CONFIG默认配置 > None
        Args:
            name: 配置项名称（如LR、BATCH_SIZE）
        Returns:
            配置项值（存在则返回，否则返回None）
        """
        # 1. 优先返回YAML文件中定义的配置项
        if self._dict.get(name) is not None:
            return self._dict[name]

        # 2. YAML中无该配置项时，返回默认配置
        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        # 3. 无匹配配置项时返回None
        return None

    def print(self):
        """打印完整配置信息（原始YAML文本+分隔符），便于调试"""
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)  # 打印原始YAML配置文本
        print('')
        print('---------------------------------')
        print('')


# 默认配置常量（所有未在YAML文件中定义的配置项均使用此默认值）
DEFAULT_CONFIG = {
    # ==================== 运行模式配置 ====================
    'MODE': 1,  # 运行模式：1=训练，2=测试，3=评估
    'MODEL': 1,  # 模型类型：
    # 1=边缘生成模型，2=图像修复模型，
    # 3=边缘-修复联合模型，4=端到端联合模型
    'MASK': 3,  # 掩码生成类型：
    # 1=随机块掩码，2=半幅掩码，3=外部掩码文件，
    # 4=（外部掩码+随机块掩码）随机选择，
    # 5=（外部掩码+随机块掩码+半幅掩码）随机选择
    'NMS': 1,  # 非极大值抑制（NMS）：
    # 0=禁用，1=对外部边缘图应用NMS（与Canny边缘相乘）
    'SEED': 10,  # 随机种子（保证实验可复现）
    'GPU': [0],  # GPU ID列表（多卡训练时指定，如[0,1]）
    'AUGMENTATION_TRAIN': 0,  # 关键点预测器训练时是否启用数据增强：1=启用，0=禁用

    # ==================== 优化器配置 ====================
    'LR': 0.0001,  # 基础学习率（生成器）
    'D2G_LR': 0.1,  # 判别器/生成器学习率比值（判别器LR = D2G_LR * 生成器LR）
    'BETA1': 0.0,  # Adam优化器beta1参数（一阶矩估计衰减率）
    'BETA2': 0.9,  # Adam优化器beta2参数（二阶矩估计衰减率）
    'BATCH_SIZE': 4,  # 训练批次大小（需根据GPU显存调整）
    'INPUT_SIZE': 256,  # 输入图像尺寸（0表示使用原始图像尺寸）
    'MAX_ITERS': 2e6,  # 最大训练迭代次数（200万次）

    # ==================== 损失函数权重配置 ====================
    'L1_LOSS_WEIGHT': 1,  # L1损失权重（像素级重构损失）
    'STYLE_LOSS_WEIGHT': 1,  # 风格损失权重（保持图像风格一致性）
    'CONTENT_LOSS_WEIGHT': 1,  # 感知损失权重（特征级重构损失，如VGG特征）
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,  # 对抗损失权重（提升生成图像真实性）
    'TV_LOSS_WEIGHT': 0.1,  # 总变分损失权重（平滑生成图像，减少伪影）

    # ==================== GAN相关配置 ====================
    'GAN_LOSS': 'lsgan',  # GAN损失类型：nsgan(标准GAN) | lsgan(最小二乘GAN) | hinge(铰链损失GAN)
    'GAN_POOL_SIZE': 0,  # 假图像缓存池大小（0=禁用，用于稳定GAN训练）

    # ==================== 训练过程控制 ====================
    'SAVE_INTERVAL': 1000,  # 模型保存间隔（迭代次数），0=不保存
    'SAMPLE_INTERVAL': 1000,  # 结果采样间隔（迭代次数），0=不采样
    'SAMPLE_SIZE': 12,  # 每次采样生成的图像数量
    'EVAL_INTERVAL': 0,  # 模型评估间隔（迭代次数），0=不评估
    'LOG_INTERVAL': 10,  # 训练日志打印间隔（迭代次数），0=不打印
}

