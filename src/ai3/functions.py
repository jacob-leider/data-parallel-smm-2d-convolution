import torch
from ai3 import core
from typing import (
    Union,
    Sequence
)

def predict(model, input):
    if isinstance(input, torch.Tensor):
        return model.predict(input.data_ptr(), input.size())

    assert False and 'bad input type'

def get_dtype_str(dtype):
    if str(dtype) == 'torch.float32':
        return 'float'
    if str(dtype) == 'torch.float64':
        return 'double'
    assert False and f'using bad dtype: {str(dtype)}'

def form_model(dtype, layers):
    dtype = get_dtype_str(dtype)
    if dtype == 'float':
        return core.Model_float(layers)
    elif dtype == 'double':
        return core.Model_double(layers)

def form_conv2d(dtype, weight, bias, *,
                 stride: Union[int, Sequence[int]],
                 padding: Union[str, Union[int, Sequence[int]]],
                 dilation: Union[int, Sequence[int]]):
    dtype = get_dtype_str(dtype)
    stride = make_2d(stride)
    dilation = make_2d(dilation)
    padding = make_padding_2d(padding, stride, dilation, weight.size())

    if isinstance(weight, torch.Tensor):
        weight_addr = weight.data_ptr()
        weight_shape = weight.size()
    else:
        assert False
    if bias is not None:
        if isinstance(bias, torch.Tensor):
            bias_addr = bias.data_ptr()
        else:
            assert False
    else:
        bias_addr = None
    if dtype == "float":
        return core.Conv2D_float(weight_addr, weight_shape, bias_addr,
                                 padding, stride, dilation)
    elif dtype == "double":
        return core.Conv2D_double(weight_addr, weight_shape, bias_addr,
                                 padding, stride, dilation)
    assert False

def conv2d_output_shape(input: torch.Tensor, kernel: torch.Tensor, padding: Sequence[int],
                        stride: Sequence[int], dilation: Sequence[int]) -> tuple:
    batch_size, _, input_height, input_width = input.shape
    output_channels, _, kernel_height, kernel_width = kernel.shape
    output_height = (input_height + 2 * padding[0] -
                     dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] -
                    dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    return (batch_size, output_channels, output_height, output_width)


def make_padding_2d(padding: Union[str, Union[int, Sequence[int]]],
                    stride: Sequence[int], dilation: Sequence[int],
                    kernel_shape: Sequence[int], dtype=int) -> Sequence[int]:
    if isinstance(padding, str):
        if padding == 'valid':
            return [0, 0]
        if padding == 'same':
            assert all(x == 1 for x in stride)
            return [dilation[0] * (kernel_shape[2] - 1) // 2, dilation[1] * (kernel_shape[3] - 1) // 2]
        assert False and f'invalid padding string: {padding}'
    else:
        return make_2d(padding, dtype=dtype)


def make_2d(a: Union[int, Sequence[int]], dtype=int) -> Sequence[int]:
    if isinstance(a, dtype):
        return [a, a]
    assert len(a) == 2 and all(isinstance(val, dtype) for val in a)
    return a
