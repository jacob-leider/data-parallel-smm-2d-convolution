import torch
import torchvision
from bench.swap_conv2d import swap_conv2d_and_time


def run():
    print(' VISION TRANSFORMER')
    input_data = torch.randn(1, 3, 224, 224)
    vision_transformer = torchvision.models.vit_b_16()
    vision_transformer.eval()
    swap_conv2d_and_time(vision_transformer, input_data)


if __name__ == "__main__":
    run()
