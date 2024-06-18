import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print(' SQUEEZENET')
    input_data = torch.randn(1, 3, 224, 224)
    squeeze = torchvision.models.squeezenet1_1()
    squeeze.eval()
    swap_conv2d_and_time(squeeze, input_data)


if __name__ == "__main__":
    run()
