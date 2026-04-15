import torch

if __name__ == '__main__':
    print("torch.cuda.current_device() = ", torch.cuda.current_device())

from main import main
main(mode=1)