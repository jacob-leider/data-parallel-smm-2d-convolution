import torch
import torchvision
from test import TEST_GROUPED_CONVOLUTION
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' SHUFFLENETV2')
    if TEST_GROUPED_CONVOLUTION:
        input_data = torch.randn(1, 3, 224, 224)
        shuffle = torchvision.models.shufflenet_v2_x2_0()
        shuffle.eval()
        swap_and_compare(shuffle, input_data, "shufflenet")
    else:
        print('Skipping as shufflenet requires groups > 1')


if __name__ == "__main__":
    run()
