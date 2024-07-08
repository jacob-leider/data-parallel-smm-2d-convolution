import torch
import torchvision
from test.integration.swap_conv2d import swap_and_compare


def run():
    print(' SWIN TRANSFORMER')
    input_data = torch.randn(1, 3, 224, 224)
    mod = torchvision.models.swin_b()
    mod.eval()
    swap_and_compare(mod, input_data, "swin transformer")


if __name__ == "__main__":
    run()
