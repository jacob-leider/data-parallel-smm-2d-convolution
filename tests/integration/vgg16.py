import torch
import torchvision
from torchvision.models.vgg import VGG16_Weights
import ai3
from tests import compare_tensors


def run():
    input_data = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.eval()
        output_original = model(input_data)
        ai3.optimize(model)
        output = model(input_data)
    compare_tensors(output, output_original, "vgg16")


if __name__ == "__main__":
    run()
