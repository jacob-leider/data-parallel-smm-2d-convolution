import torch
import torchvision
from test import TEST_GROUPED_CONVOLUTION
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' CONVNEXT')
    if TEST_GROUPED_CONVOLUTION:
        input_data = torch.randn(1, 3, 224, 224)
        orig = torchvision.models.convnext_base(weights='DEFAULT')
        orig.eval()
        swap_and_compare(orig, input_data, "convnext")
    else:
        print('Skipping as convnext requires groups > 1')


if __name__ == "__main__":
    run()
