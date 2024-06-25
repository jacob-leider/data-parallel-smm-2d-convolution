import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def run():
    print('  REGNET')
    print('Skipping as regnet requires groups > 1')
    return
    input_data = torch.randn(1, 3, 224, 224)
    regnet = torchvision.models.regnet_y_16gf()
    regnet.eval()
    swap_conv2d_and_time(regnet, input_data)


if __name__ == "__main__":
    run()
