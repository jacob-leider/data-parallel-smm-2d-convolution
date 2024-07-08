import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' RESNET')
    input_data = torch.randn(1, 3, 224, 224)
    resnet = torchvision.models.resnet152()
    resnet.eval()
    swap_and_compare(resnet, input_data, "resnet")


if __name__ == "__main__":
    run()
