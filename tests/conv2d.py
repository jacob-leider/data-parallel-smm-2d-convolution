import torch
import numpy as np
import torch.nn.functional as F
from ai3 import functions

def compare_tensors(out_tensor: torch.Tensor, tar_tensor: torch.Tensor, mes: str, atol=1e-5) -> None:
    out = np.array(out_tensor)
    tar = np.array(tar_tensor)

    if tar.shape != out.shape:
        print("Tensors have different shapes:", tar.shape, "and", out.shape)
        print(f"Target:\n{tar}")
        print(f"Output:\n{out}")
        return

    different_elements = np.where(np.abs(out - tar) > atol)

    if len(different_elements[0]) == 0:
        print(f"Passed Test {mes}")
    else:
        print(f"Failed Test {mes}")
        print(f"Target:\n{tar}")
        print(f"Output:\n{out}")
        print("Tensors differ at the following indices:")
        for index in zip(*different_elements):
            print("at:", index, "target:", tar[index], "output:", out[index])

if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 5, 5)
    kernel_tensor = torch.randn(1, 1, 3, 3)
    torch_output = F.conv2d(input_tensor, kernel_tensor)
    ai3_output = functions.conv2d(input_tensor, kernel_tensor)

    compare_tensors(ai3_output, torch_output, "basic")

    bias_tensor = torch.randn(1)
    torch_output = F.conv2d(input_tensor, kernel_tensor, bias=bias_tensor)
    ai3_output = functions.conv2d(input_tensor, kernel_tensor, bias=bias_tensor)

    compare_tensors(ai3_output, torch_output, "bias")

    input_channels = 3
    output_channels = 4
    groups = 1
    input_tensor = torch.randn(1, input_channels, 5, 5)
    kernel_tensor = torch.randn(output_channels, int(input_channels / groups), 3, 3)
    torch_output = F.conv2d(input_tensor, kernel_tensor, bias=None)
    ai3_output = functions.conv2d(input_tensor, kernel_tensor, bias=None)

    compare_tensors(ai3_output, torch_output, "multi channel no bias")

    bias_tensor = torch.randn(output_channels)
    torch_output = F.conv2d(input_tensor, kernel_tensor, bias=bias_tensor)
    ai3_output = functions.conv2d(input_tensor, kernel_tensor, bias=bias_tensor)

    compare_tensors(ai3_output, torch_output, "multi channel with bias")
