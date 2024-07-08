import torch
import torchvision
from test import TEST_GROUPED_CONVOLUTION
from test.integration.swap_conv2d import swap_and_compare


def run():
    print('  MOBILENET')
    if TEST_GROUPED_CONVOLUTION:
        input_data = torch.randn(1, 3, 224, 224)
        mnas = torchvision.models.mobilenet_v3_large()
        mnas.eval()
        swap_and_compare(mnas, input_data, "mobilenet")
    else:
        print('Skipping as mobilenet requires groups > 1')


if __name__ == "__main__":
    run()
