import torch
import ai3
from test import compare_tensors
import torchvision


def run():
    print("VGG16 SWAPPING LAYERS")
    input_data = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        vgg16 = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.DEFAULT)
        vgg16.eval()
        target = vgg16(input_data)
        ai3.swap_conv2d(
            vgg16)
        output = vgg16(input_data)
        compare_tensors(output, target, "vgg16 swap conv2d")


if __name__ == "__main__":
    run()
