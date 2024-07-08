import torch
import torchvision
from test import TEST_GROUPED_CONVOLUTION
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' MAXVIT')
    if TEST_GROUPED_CONVOLUTION:
        input_data = torch.randn(1, 3, 224, 224)
        mod = torchvision.models.maxvit_t()
        mod.eval()
        swap_and_compare(mod, input_data, "maxvit")
    else:
        print('Skipping as maxvit requires groups > 1')


if __name__ == "__main__":
    run()
