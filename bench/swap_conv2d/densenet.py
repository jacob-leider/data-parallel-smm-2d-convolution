import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print(' DENSENET')
    input_data = torch.randn(1, 3, 224, 224)
    dense = torchvision.models.DenseNet()
    dense.eval()
    swap_conv2d_and_time(dense, input_data)


if __name__ == "__main__":
    run()
