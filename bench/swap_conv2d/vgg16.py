import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def conv2d_selector(orig: torch.nn.Conv2d) -> str:
    in_channels = orig.weight.shape[1]
    if in_channels > 200:
        return "smm"
    return "direct"


def run():
    print(' VGG16')
    input_data = torch.randn(1, 3, 224, 224)
    vgg16 = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.DEFAULT)
    vgg16.eval()
    swap_conv2d_and_time(vgg16, input_data, selector_func=conv2d_selector)


if __name__ == "__main__":
    run()
