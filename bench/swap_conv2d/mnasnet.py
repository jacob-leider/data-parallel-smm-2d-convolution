import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print('  MNASNET')
    print('Skipping as mnasnet requires groups > 1')
    return
    input_data = torch.randn(1, 3, 224, 224)
    mnas = torchvision.models.mnasnet1_0(weights=torchvision.models.MNASNet1_0_Weights.DEFAULT)
    mnas.eval()
    swap_conv2d_and_time(mnas, input_data)


if __name__ == "__main__":
    run()
