import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def run():
    print(' INCEPTION')
    input_data = torch.randn(1, 3, 224, 224)
    inception = torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    inception.eval()
    swap_conv2d_and_time(inception, input_data)


if __name__ == "__main__":
    run()
