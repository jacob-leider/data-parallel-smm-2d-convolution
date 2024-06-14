import torch
import ai3
from tests import compare_tensors
import torchvision


def run():
    print("VGG16 SWAPPING LAYERS")
    input_data = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        pytorch_vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        pytorch_vgg16.eval()
        target = pytorch_vgg16(input_data)
        with_ai3_conv = ai3.swap_conv2d(
            pytorch_vgg16)
        output = with_ai3_conv(input_data)
        compare_tensors(output, target, "vgg16 swap conv2d")


if __name__ == "__main__":
    run()
