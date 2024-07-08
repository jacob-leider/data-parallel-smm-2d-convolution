import torch
import torchvision
from test import TEST_GROUPED_CONVOLUTION
from test.integration.swap_conv2d import swap_and_compare


def run():
    print('  REGNET')
    if TEST_GROUPED_CONVOLUTION:
        input_data = torch.randn(1, 3, 224, 224)
        regnet = torchvision.models.regnet_y_16gf()
        regnet.eval()
        swap_and_compare(regnet, input_data, "regnet")
    else:
        print('Skipping as regnet requires groups > 1')


if __name__ == "__main__":
    run()
