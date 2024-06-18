from ai3 import core, utils
import torch
from torch import fx, nn
from typing import (
    Union,
    Sequence,
    Optional,
    List
)
from abc import ABC


class Layer(ABC):
    def __init__(self, core):
        self.core = core
        ...


class Conv2D(Layer):
    def __init__(self, dtype, weight, bias,
                 stride: Union[int, Sequence[int]],
                 padding: Union[str, Union[int, Sequence[int]]],
                 dilation: Union[int, Sequence[int]], padding_mode: str, groups: int):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        padding = utils.make_padding_2d(
            padding, stride, dilation, weight.size())
        assert (padding_mode in ['zeros', 'reflect', 'replicate', 'circular'])
        pad_mode = {
            'zeros': core.PaddingMode.zeros,
            'reflect': core.PaddingMode.reflect,
            'replicate': core.PaddingMode.replicate,
            'circular': core.PaddingMode.circular
        }[padding_mode]

        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = utils.get_correct_from_type(dtype, core.Conv2D_float,
                                                core.Conv2D_double)(weight_addr,
                                                                    weight_shape,
                                                                    bias_addr,
                                                                    padding,
                                                                    stride,
                                                                    dilation,
                                                                    pad_mode,
                                                                    groups)

    def forward(self, input) -> Union[core.Tensor_float, core.Tensor_double]:
        return self.core.forward(utils.get_address(input), utils.get_shape(input))


class Linear(Layer):
    def __init__(self, dtype, weight, bias):
        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = utils.get_correct_from_type(
            dtype, core.Linear_float, core.Linear_double)(weight_addr, weight_shape, bias_addr)


class ReLU(Layer):
    def __init__(self, dtype):
        self.core = utils.get_correct_from_type(
            dtype, core.ReLU_float, core.ReLU_double)()


class MaxPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]],
                 dilation: Union[int, Sequence[int]],
                 ceil_mode: bool):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)
        ceil_mode = ceil_mode

        self.core = utils.get_correct_from_type(dtype, core.MaxPool2D_float, core.MaxPool2D_double)(kernel_shape,
                                                                                                    padding, stride, dilation, ceil_mode)


class AvgPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]],
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int]):
        stride = utils.make_2d(stride)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)

        self.core = utils.get_correct_from_type(dtype, core.AvgPool2D_float, core.AvgPool2D_double)(kernel_shape,
                                                                                                    padding, stride, ceil_mode, count_include_pad, divisor_override)


class AdaptiveAvgPool2D(Layer):
    def __init__(self, dtype, output_shape: Optional[Union[int, Sequence[Optional[int]]]]):
        if isinstance(output_shape, list):
            assert (len(output_shape) == 2)
        if isinstance(output_shape, int):
            output_shape = utils.make_2d(output_shape)
        self.core = utils.get_correct_from_type(
            dtype, core.AdaptiveAvgPool2D_float, core.AdaptiveAvgPool2D_double)(output_shape)


class Flatten(Layer):
    def __init__(self, dtype, start_dim: int, end_dim: int):
        self.core = utils.get_correct_from_type(
            dtype, core.Flatten_float, core.Flatten_double)(start_dim, end_dim)
