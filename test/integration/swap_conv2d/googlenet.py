import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' GOOGLENET')
    input_data = torch.randn(1, 3, 224, 224)
    googlenet = torchvision.models.googlenet(
        weights=torchvision.models.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()
    swap_and_compare(googlenet, input_data, "google net")


if __name__ == "__main__":
    run()
