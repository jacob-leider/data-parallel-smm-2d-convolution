import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time

def run():
    print('  GOOGLENET')
    input_data = torch.randn(1, 3, 224, 224)
    googlenet = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT)
    googlenet.eval()
    swap_conv2d_and_time(googlenet, input_data)


if __name__ == "__main__":
    run()
