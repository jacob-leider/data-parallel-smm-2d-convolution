import torch
import torch.nn.functional as F
import ai3_functions

if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 5, 5)
    kernel_tensor = torch.randn(1, 1, 3, 3)

    output = F.conv2d(input_tensor, kernel_tensor, bias=None, stride=1, padding='valid')

    print(output)
