import torch
import torchvision
from test import TEST_GROUPED_CONVOLUTION
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' EFFICIENT NET')
    if TEST_GROUPED_CONVOLUTION:
        input_data = torch.randn(1, 3, 224, 224)
        dense = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        dense.eval()
        swap_and_compare(dense, input_data, "efficient net")
    else:
        print('Skipping as efficientnet requires groups > 1')


if __name__ == "__main__":
    run()
