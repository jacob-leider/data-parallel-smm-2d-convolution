import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print('  MAXVIT')
    print('Skipping as maxvit requires groups > 1')
    return
    input_data = torch.randn(1, 3, 224, 224)
    mod = torchvision.models.maxvit_t()
    mod.eval()
    swap_conv2d_and_time(mod, input_data)


if __name__ == "__main__":
    run()
