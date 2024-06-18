from .alexnet import *
import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def run():
    print(' ALEXNET')
    input_data = torch.randn(1, 3, 224, 224)
    orig = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.DEFAULT)
    orig.eval()
    swap_conv2d_and_time(orig, input_data)


if __name__ == "__main__":
    run()
