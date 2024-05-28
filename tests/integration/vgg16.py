import torch
import torchvision
from torchvision.models.vgg import VGG16_Weights
import ai3
from tests import compare_tensors


# TODO for this also have to do AdaptiveAvgPool2d https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
# idk what that is just do 2d avg pooling, https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
# define the output size and then the rest is decided from that, so we can just do average pooling and do the setting of kernel size in the python
def run():
    input_data = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.eval()
        # print(model)
        original = model(input_data)
        ai3.optimize(model, replace=True)
        output = model(input_data)
    compare_tensors(output, original, "vgg16")


if __name__ == "__main__":
    run()
