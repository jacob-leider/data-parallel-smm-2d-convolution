import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' SQUEEZENET')
    input_data = torch.randn(1, 3, 224, 224)
    squeeze = torchvision.models.squeezenet1_1()
    squeeze.eval()
    swap_and_compare(squeeze, input_data, "squeezenet")


if __name__ == "__main__":
    run()
