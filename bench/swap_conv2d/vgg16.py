import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def run():
    print('  VGG16')
    input_data = torch.randn(1, 3, 224, 224)
    vgg16 = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.DEFAULT)
    vgg16.eval()
    swap_conv2d_and_time(vgg16, input_data)


if __name__ == "__main__":
    run()
