import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' VGG16')
    input_data = torch.randn(1, 3, 224, 224)
    vgg16 = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.DEFAULT)
    vgg16.eval()
    swap_and_compare(vgg16, input_data, "vgg16")


if __name__ == "__main__":
    run()
