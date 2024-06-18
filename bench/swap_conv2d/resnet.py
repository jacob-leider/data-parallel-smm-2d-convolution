import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print('  RESNET')
    input_data = torch.randn(1, 3, 224, 224)
    resnet = torchvision.models.resnet152()
    resnet.eval()
    swap_conv2d_and_time(resnet, input_data)


if __name__ == "__main__":
    run()
