import torch
import torchvision
from test import TEST_GROUPED_CONVOLUTION
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' MNASNET')
    if TEST_GROUPED_CONVOLUTION:
        input_data = torch.randn(1, 3, 224, 224)
        mnas = torchvision.models.mnasnet1_0(
            weights=torchvision.models.MNASNet1_0_Weights.DEFAULT)
        mnas.eval()
        swap_and_compare(mnas, input_data, "mnasnet")
    else:
        print('Skipping as mnasnet requires groups > 1')


if __name__ == "__main__":
    run()
