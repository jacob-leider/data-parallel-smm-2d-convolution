import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' VISION TRANSFORMER')
    input_data = torch.randn(1, 3, 224, 224)
    vision_transformer = torchvision.models.vit_b_16()
    vision_transformer.eval()
    swap_and_compare(vision_transformer, input_data, "vision transformer")


if __name__ == "__main__":
    run()
