import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print('  MOBILENET')
    print('Skipping as mobilenet requires groups > 1')
    return
    input_data = torch.randn(1, 3, 224, 224)
    mnas = torchvision.models.mobilenet_v3_large()
    mnas.eval()
    swap_conv2d_and_time(mnas, input_data)


if __name__ == "__main__":
    run()
