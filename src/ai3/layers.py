import torch
from torch import nn, Tensor
from ai3 import functions
from typing import (
    Optional,
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
                 dtype=None,
                 replace_weight: Optional[Tensor] = None,
                 replace_bias: Optional[Tensor] = None) -> None:
        super(Conv2d, self).__init__()
        kernel_size = functions.make_2d(kernel_size)

        weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        if replace_weight is not None:
            assert weight_shape == replace_weight.shape
            self.weight = replace_weight
        else:
            self.weight = nn.Parameter(torch.randn(weight_shape, dtype=dtype))

        self.bias = None
        if bias:
            if replace_bias is not None:
                assert (out_channels,) == replace_bias.shape
                self.bias = replace_bias
            else:
                self.bias = nn.Parameter(torch.randn(out_channels, dtype=dtype))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation


    def forward(self, x):
        return functions.conv2d(x, self.weight, bias=self.bias,
                                padding=self.padding, stride=self.stride,
                                dilation=self.dilation)
