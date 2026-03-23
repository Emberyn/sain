# 导入必要的系统和第三方库
import os  # 操作系统相关功能（路径、文件操作）
import cv2  # OpenCV库，用于图像处理
import random  # 随机数生成
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import argparse  # 命令行参数解析
from shutil import copyfile  # 文件复制功能
from src.config import Config  # 自定义配置类（读取yml配置文件）
from src.leaf_inpaint import LEAFINPAINT  # 核心模型类（叶片图像修复）
import wandb  # 实验跟踪工具（可选，用于可视化训练过程）


def main(mode=None):
    r"""
    主函数：初始化配置、设备、随机种子，构建并启动模型（训练/测试）

    Args:
        mode (int, optional): 运行模式 
            1: 训练模式 
            2: 测试模式 
            若未指定，从配置文件读取模式
    """
    # 加载配置文件（根据mode覆盖命令行参数）
    config = load_config(mode)

    # 设置可见的GPU设备（将config中的GPU列表转为字符串，如[0,1]转为"0,1"）
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # 初始化计算设备（优先使用GPU，否则使用CPU）
    if torch.cuda.is_available():
        print('CUDA可用，使用GPU训练/测试')
        config.DEVICE = torch.device("cuda")  # 设置设备为GPU
        # 启用cudnn自动调优，提升卷积运算效率（适用于固定输入尺寸的场景）
        torch.backends.cudnn.benchmark = True
    else:
        print('CUDA不可用，使用CPU训练/测试')
        config.DEVICE = torch.device("cpu")  # 设置设备为CPU

    # 设置OpenCV的线程数为1（避免与PyTorch的数据加载器(dataloader)产生死锁）
    cv2.setNumThreads(0)

    # 初始化随机种子（保证实验可复现）
    torch.manual_seed(config.SEED)  # PyTorch CPU随机种子
    torch.cuda.manual_seed_all(config.SEED)  # PyTorch GPU随机种子（多卡）
    np.random.seed(config.SEED)  # NumPy随机种子
    random.seed(config.SEED)  # Python原生随机种子

    # 构建叶片修复模型实例，并加载预训练权重（若有）
    model = LEAFINPAINT(config)
    model.load()

    # 训练模式
    if config.MODE == 1:
        config.print()  # 打印当前配置信息（便于核对参数）
        print('\n开始训练模型...\n')
        model.train()  # 调用模型的训练方法

    # 测试模式
    elif config.MODE == 2:
        print('\n开始测试模型...\n')
        model.test()  # 调用模型的测试方法


def load_config(mode=None):
    r"""
    加载配置文件：解析命令行参数，合并命令行参数与yml配置文件的参数

    Args:
        mode (int, optional): 运行模式（1:训练，2:测试），若未指定则从配置文件读取

    Returns:
        Config: 合并后的配置对象（包含所有训练/测试参数）
    """
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser()

    # 通用参数：模型 checkpoint 路径（默认：./checkpoint）
    parser.add_argument('--path', '--checkpoint', type=str, default='./checkpoint',
                        help='模型权重保存/加载路径（默认: ./checkpoint）')

    # 通用参数：选择模型类型
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4],
                        help='模型类型：1:边缘检测模型, 2:图像修复模型, 3:边缘-修复联合模型, 4:端到端联合模型')

    # 测试模式专属参数（仅在mode=2时生效）
    if mode == 2:
        parser.add_argument('--input', type=str, help='测试输入图像的目录路径或单张图像路径')
        parser.add_argument('--mask', type=str, help='测试掩码（缺失区域）的目录路径或单张掩码路径')
        parser.add_argument('--edge', type=str, help='测试边缘图像的目录路径或单张边缘图像路径')
        parser.add_argument('--output', type=str, help='测试结果的保存目录路径')

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 拼接配置文件路径（checkpoint目录下的config.yml），统一路径分隔符为/
    config_path = os.path.join(args.path, 'config.yml').replace('\\', '/')

    # 若checkpoint目录不存在，创建该目录（用于保存权重、配置文件）
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # 若配置文件不存在，从模板文件（config.yml.example）复制一份到checkpoint目录
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # 加载yml配置文件为Config对象（自定义类，支持属性化访问配置项）
    config = Config(config_path)
    print("当前加载的配置文件路径=", config_path)

    # 训练模式：覆盖配置文件中的模式和模型类型
    if mode == 1:
        config.MODE = 1  # 强制设置为训练模式
        if args.model:  # 若命令行指定了model参数，覆盖配置文件中的模型类型
            config.MODEL = args.model

    # 测试模式：覆盖配置文件中的模式、模型类型和测试数据路径
    elif mode == 2:
        config.MODE = 2  # 强制设置为测试模式
        # 若命令行未指定model，默认使用3（边缘-修复联合模型）
        config.MODEL = args.model if args.model is not None else 3

        # 覆盖测试输入图像路径（若命令行指定）
        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        # 覆盖测试掩码路径（若命令行指定）
        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        # 覆盖测试边缘图像路径（若命令行指定）
        if args.edge is not None:
            config.TEST_INPAINT_EDGE_FLIST = args.edge

        # 覆盖测试结果保存路径（若命令行指定）
        if args.output is not None:
            config.RESULTS = args.output

    # 返回最终的配置对象
    return config


# 程序入口：当脚本直接运行时执行main函数
if __name__ == "__main__":
    main()