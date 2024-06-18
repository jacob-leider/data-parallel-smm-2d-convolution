import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print('  SHUFFLENETV2')
    print('Skipping as shufflenet requires groups > 1')
    return
    input_data = torch.randn(1, 3, 224, 224)
    shuffle = torchvision.models.shufflenet_v2_x2_0()
    shuffle.eval()
    swap_conv2d_and_time(shuffle, input_data)


if __name__ == "__main__":
    run()
