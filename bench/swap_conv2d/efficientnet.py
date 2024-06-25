import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def run():
    print(' EFFICIENT NET')
    print('Skipping as efficientnet requires groups > 1')
    return
    input_data = torch.randn(1, 3, 224, 224)
    dense = torchvision.models.efficientnet_v2_s(
        weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
    dense.eval()
    swap_conv2d_and_time(dense, input_data)


if __name__ == "__main__":
    run()
