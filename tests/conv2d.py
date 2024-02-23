from typing import Union, Sequence
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


def test(*, input_channels: int, in_height: int, in_width: int,
         output_channels: int, kernel_height: int, kernel_width: int,
         with_bias: bool = False,
         padding: Union[int, Sequence[int]] = 0,
         test_name: str, atol=1e-5) -> None:
        input= torch.randn(1, input_channels, in_height, in_width)
        kernel = torch.randn(output_channels, input_channels, kernel_height, kernel_width)
        if with_bias:
            bias = torch.randn(output_channels)
        else:
            bias = None

        ai3_output = functions.conv2d(input, kernel, bias=bias, padding=padding)
        torch_output = F.conv2d(input, kernel, bias=bias, padding=padding)
        compare_tensors(ai3_output, torch_output, test_name, atol=atol)

if __name__ == "__main__":
    test(input_channels = 4,
         in_height=30,
         in_width=40,
         output_channels = 6,
         kernel_height = 7,
         kernel_width = 5,
         with_bias = True,
         padding = 5,
         atol=1e-4,
         test_name="1d padding")

    test(input_channels = 4,
         in_height=30,
         in_width=40,
         output_channels = 6,
         kernel_height = 7,
         kernel_width = 5,
         with_bias = True,
         padding = (2,5),
         atol=1e-4,
         test_name="2d padding")

    test(input_channels = 1,
         in_height=5,
         in_width=5,
         output_channels = 1,
         kernel_height = 3,
         kernel_width = 3,
         test_name="basic")

    test(input_channels = 1,
         in_height=100,
         in_width=150,
         output_channels = 1,
         kernel_height = 15,
         kernel_width = 12,
         with_bias = True,
         atol=1e-4,
         test_name="with bias")

    test(input_channels = 1,
         in_height=10,
         in_width=15,
         output_channels = 1,
         kernel_height = 10,
         kernel_width = 15,
         with_bias = True,
         atol=1e-4,
         test_name="kern.shape = input.shape")

    test(input_channels = 3,
         in_height=50,
         in_width=150,
         output_channels = 4,
         kernel_height = 10,
         kernel_width = 10,
         atol=1e-4,
         test_name="multi channel")

    test(input_channels = 4,
         in_height=50,
         in_width=150,
         output_channels = 6,
         kernel_height = 10,
         kernel_width = 10,
         with_bias = True,
         atol=1e-4,
         test_name="multi channel with bias")

    test(input_channels = 4,
         in_height=50,
         in_width=150,
         output_channels = 6,
         kernel_height = 10,
         kernel_width = 10,
         with_bias = True,
         atol=1e-4,
         test_name="multi channel with bias")

