import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.leaf_inpaint import LEAFINPAINT
import wandb


def main(mode=None):
    r"""
    主函数：初始化配置、设备、随机种子，构建并启动模型（训练/测试）

    Args:
        mode (int, optional): 运行模式 
            1: 训练模式 
            2: 测试模式 
            若未指定，从配置文件读取模式
    """
    config = load_config(mode)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    if torch.cuda.is_available():
        print('CUDA可用，使用GPU训练/测试')
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print('CUDA不可用，使用CPU训练/测试')
        config.DEVICE = torch.device("cpu")

    cv2.setNumThreads(0)

    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    model = LEAFINPAINT(config)
    model.load()

    if config.MODE == 1:
        config.print()
        print('\n开始训练模型...\n')
        model.train()

    elif config.MODE == 2:
        print('\n开始测试模型...\n')
        model.test()


def load_config(mode=None):
    r"""
    加载配置文件：解析命令行参数，合并命令行参数与yml配置文件的参数

    Args:
        mode (int, optional): 运行模式（1:训练，2:测试），若未指定则从配置文件读取

    Returns:
        Config: 合并后的配置对象（包含所有训练/测试参数）
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', '--checkpoint', type=str, default='./checkpoint',
                        help='模型权重保存/加载路径（默认: ./checkpoint）')

    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4],
                        help='模型类型：1:边缘检测模型, 2:图像修复模型, 3:边缘-修复联合模型, 4:端到端联合模型')

    if mode == 2:
        parser.add_argument('--input', type=str, help='测试输入图像的目录路径或单张图像路径')
        parser.add_argument('--mask', type=str, help='测试掩码（缺失区域）的目录路径或单张掩码路径')
        parser.add_argument('--edge', type=str, help='测试边缘图像的目录路径或单张边缘图像路径')
        parser.add_argument('--output', type=str, help='测试结果的保存目录路径')

    args = parser.parse_args()

    # 拼接配置文件路径（checkpoint目录下的config.yml），统一路径分隔符为/
    config_path = os.path.join(args.path, 'config.yml').replace('\\', '/')

    # 若checkpoint目录不存在，创建该目录（用于保存权重、配置文件）
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # 若配置文件不存在，从模板文件（config.yml.example）复制一份到checkpoint目录
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    config = Config(config_path)
    print("当前加载的配置文件路径=", config_path)

    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3

        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.edge is not None:
            config.TEST_INPAINT_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    return config


if __name__ == "__main__":
    main()
