import torch
from torch import nn
from ai3 import functions
from typing import (
    Union,
    Sequence
)

#  missing
#  groups: int,
#  output_padding: Tuple[int, ...],
#  padding_mode: str,
#  transposed
#  device=None,


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 bias: bool,
                 stride: Union[int, Sequence[int]] = 1,
                 padding: Union[str, Union[int, Sequence[int]]] = 0,
                 dilation: Union[int, Sequence[int]] = 1,
                 dtype=None) -> None:
        super(Conv2d, self).__init__()
        kernel_size = functions.make_2d(kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels,
                                               kernel_size[0], kernel_size[1], dtype=dtype))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, dtype=dtype))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        return functions.conv2d(x, self.weight, bias=self.bias,
                                padding=self.padding, stride=self.stride,
                                dilation=self.dilation)
