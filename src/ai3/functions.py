import torch
from ai3 import core
from typing import (
    Optional,
    Union,
    Sequence
)


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

# TODO other parameters from the PyTorch impl
# groups


def conv2d(input: torch.Tensor, kernel: torch.Tensor, bias: Optional[torch.Tensor] = None,
           padding: Union[str, Union[int, Sequence[int]]] = 0,
           stride: Union[int, Sequence[int]] = 1,
           dilation: Union[int, Sequence[int]] = 1
           ) -> torch.Tensor:
    assert input.dtype == kernel.dtype
    if bias is not None:
        assert input.dtype == bias.dtype
        bias_ptr = bias.untyped_storage()
    else:
        bias_ptr = None
    stride = make_2d(stride)
    dilation = make_2d(dilation)
    padding = make_padding_2d(padding, stride, dilation, kernel.shape)

    output_size = conv2d_output_shape(input, kernel, padding, stride, dilation)
    output_tensor = torch.empty(
        output_size, dtype=input.dtype, requires_grad=False)

    return core.kn2row_conv2d(input.untyped_storage(), input.shape,
                              kernel.untyped_storage(), kernel.shape,
                              str(input.dtype),
                              bias_ptr,
                              padding,
                              stride,
                              dilation,
                              output_tensor)
