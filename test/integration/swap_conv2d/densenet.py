import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' DENSENET')
    input_data = torch.randn(1, 3, 224, 224)
    dense = torchvision.models.DenseNet()
    dense.eval()
    swap_and_compare(dense, input_data, "densenet")


if __name__ == "__main__":
    run()
