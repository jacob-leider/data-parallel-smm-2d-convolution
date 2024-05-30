from ai3 import core
from ai3 import utils
from typing import (
    Union,
    Sequence,
    Optional
)

class Model():
    def __init__(self, dtype, layers):
        cores = [layer.core for layer in layers]
        self.core = utils.get_correct_from_type(dtype, core.Model_float, core.Model_double)(cores)

    def predict(self, input):
        return self.core.predict(utils.get_address(input), utils.get_shape(input))

class Conv2D():
    def __init__(self, dtype, weight, bias,
                 stride: Union[int, Sequence[int]],
                 padding: Union[str, Union[int, Sequence[int]]],
                 dilation: Union[int, Sequence[int]]):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        padding = utils.make_padding_2d(padding, stride, dilation, weight.size())

        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = utils.get_correct_from_type(dtype, core.Conv2D_float, core.Conv2D_double)(weight_addr, weight_shape, bias_addr,
                                 padding, stride, dilation)

class Linear():
    def __init__(self, dtype, weight, bias):
        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = utils.get_correct_from_type(dtype, core.Linear_float, core.Linear_double)(weight_addr, weight_shape, bias_addr)


class ReLU():
    def __init__(self, dtype):
        self.core = utils.get_correct_from_type(dtype, core.ReLU_float, core.ReLU_double)()


class MaxPool2D():
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

class AvgPool2D():
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

class AdaptiveAvgPool2D():
    def __init__(self, dtype, output_shape: Union[int, None, Sequence[Optional[int]]]):
        if isinstance(output_shape, list):
            assert(len(output_shape) == 2)
        if isinstance(output_shape, int):
            output_shape = utils.make_2d(output_shape)
        self.core = utils.get_correct_from_type(dtype, core.AdaptiveAvgPool2D_float, core.AdaptiveAvgPool2D_double)(output_shape)
