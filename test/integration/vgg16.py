import torch
import ai3
from test import compare_tensors
import torchvision


def run():
    print("VGG16")
    input_data = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        pytorch_vgg16 = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.DEFAULT)
        pytorch_vgg16.eval()
        target = pytorch_vgg16(input_data)
        ai3_model = ai3.swap_backend(pytorch_vgg16)
        output = ai3_model.predict(input_data)
        compare_tensors(output, target, "vgg16 multi sample")


if __name__ == "__main__":
    run()
