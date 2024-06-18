import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def run():
    print('  CONVNEXT')
    print('Skipping as convnext requires groups > 1')
    return
    input_data = torch.randn(1, 3, 224, 224)
    orig = torchvision.models.convnext_base(weights='DEFAULT')
    orig.eval()
    swap_conv2d_and_time(orig, input_data)


if __name__ == "__main__":
    run()
