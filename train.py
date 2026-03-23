import torch

if __name__ == '__main__':
    print("torch.cuda.current_device() = ", torch.cuda.current_device())

from main import main
main(mode=1)

#  CUDA_VISIBLE_DEVICES=1 python train.py   k2289093153@.   @Debug 此处需重新训练