import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' ALEXNET')
    input_data = torch.randn(1, 3, 224, 224)
    orig = torchvision.models.alexnet()
    orig.eval()
    swap_and_compare(orig, input_data)


if __name__ == "__main__":
    run()
