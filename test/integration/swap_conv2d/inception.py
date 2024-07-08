import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' INCEPTION')
    input_data = torch.randn(1, 3, 224, 224)
    inception = torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    inception.eval()
    swap_and_compare(inception, input_data, "inception")


if __name__ == "__main__":
    run()
