from typing import (
    Union,
    Sequence,
    Optional,
    Tuple
)
from abc import ABC
from ai3 import core, errors, utils
from ai3.tensor import Tensor


class Layer(ABC):
    def __init__(self, core, algorithm):
        self.core = core
        self.algorithm = algorithm
        ...


class Conv2D(Layer):
    def __init__(self, dtype, weight, bias, _stride: Union[int, Tuple[int, ...]],
                 padding: Union[str, Union[int, Tuple[int, ...]]], _dilation:
                 Union[int, Tuple[int, ...]], padding_mode: str, groups: int,
                 algorithm: str):
        stride = utils.make_2d(_stride)
        dilation = utils.make_2d(_dilation)
        padding = utils.make_padding_2d(
            padding, stride, dilation, weight.size())
        errors.bail_if(padding_mode not in [
                       'zeros', 'reflect', 'replicate', 'circular'], f"invalid padding mode: {padding_mode}")
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
        self.core = utils.get_item(dtype, core.Conv2D_float,
                                   core.Conv2D_double)(weight_addr,
                                                       weight_shape,
                                                       bias_addr,
                                                       padding[0],
                                                       padding[1],
                                                       stride[0],
                                                       stride[1],
                                                       dilation[0],
                                                       dilation[1],
                                                       core.PaddingMode(
                                                           pad_mode),
                                                       groups,
                                                       algorithm)

    def set_algo(self, algo: str):
        self.core.algorithm = algo

    def forward(self, input) -> Tensor:
        return Tensor(self.core.forward(utils.get_address(input), utils.get_shape(input)))


class Linear(Layer):
    def __init__(self, dtype, weight, bias, algorithm: str):
        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = utils.get_item(
            dtype, core.Linear_float, core.Linear_double)(weight_addr, weight_shape, bias_addr, algorithm)


class ReLU(Layer):
    def __init__(self, dtype, algorithm: str):
        layer = utils.get_item(
            dtype, core.ReLU_float, core.ReLU_double)
        self.core = layer(algorithm)


class MaxPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 dilation: Union[int, Tuple[int, int]],
                 ceil_mode: bool,
                 algorithm: str):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)
        ceil_mode = ceil_mode

        self.core = utils.get_item(
            dtype, core.MaxPool2D_float, core.MaxPool2D_double)(kernel_shape[0], kernel_shape[1], padding[0], padding[1], stride[0],
                                                                stride[1], dilation[0], dilation[1], ceil_mode,
                                                                algorithm)


class AvgPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int],
                 algorithm: str):
        padding_2d = utils.make_2d(padding)
        stride_2d = utils.make_2d(stride)
        kernel_shape = utils.make_2d(kernel_shape)

        self.core = utils.get_item(
            dtype, core.AvgPool2D_float, core.AvgPool2D_double)(kernel_shape[0], kernel_shape[1],
                                                                padding_2d[0], padding_2d[1], stride_2d[0], stride_2d[1], ceil_mode, count_include_pad, divisor_override, algorithm)


class AdaptiveAvgPool2D(Layer):
    def __init__(self, dtype, output_shape: Optional[Union[int, Sequence[Optional[int]]]], algorithm: str):
        if isinstance(output_shape, int):
            output_shape = utils.make_2d(output_shape)
        elif output_shape is None:
            output_shape = [None, None]
        self.core = utils.get_item(
            dtype, core.AdaptiveAvgPool2D_float, core.AdaptiveAvgPool2D_double)(output_shape[0], output_shape[1], algorithm)


class Flatten(Layer):
    def __init__(self, dtype, start_dim: int, end_dim: int, algorithm: str):
        self.core = utils.get_item(
            dtype, core.Flatten_float, core.Flatten_double)(start_dim, end_dim, algorithm)
