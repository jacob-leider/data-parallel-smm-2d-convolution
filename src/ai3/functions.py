import torch
from ai3 import core
from typing import (
    Optional,
    Union,
    Sequence
)

def get_conv2d_output_shape(input: torch.Tensor, kernel: torch.Tensor, padding: Sequence[int], stride: Sequence[int]) -> tuple:
    # same
    # output_height = input_height;
    # output_width = input_width;
    # full
    # output_height = input_height + kernel_height - 1;
    # output_width = input_width + kernel_width - 1;
    # valid
    batch_size, _, input_height, input_width = input.shape
    output_channels, _, kernel_height, kernel_width = kernel.shape
    output_height = (input_height + 2 * padding[0] - kernel_height) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_width) // stride[1] + 1;
    return (batch_size, output_channels, output_height, output_width)

# TODO when doing training step need to have bias True or False if True then give a tensor of (out_channels) shape
# TODO need overloaded function somewhere to have support for padding as a string
def conv2d(input: torch.Tensor, kernel: torch.Tensor, bias: Optional[torch.Tensor] = None,
           padding: Union[int, Sequence[int]] = 0,
           stride: Union[int, Sequence[int]] = 0) -> torch.Tensor:
    assert input.dtype == kernel.dtype
    if bias is not None:
        assert input.dtype == bias.dtype
        bias_ptr = bias.untyped_storage()
    else:
        bias_ptr = None

    if isinstance(padding, int):
        padding = [padding, padding]
    else:
        assert len(padding) == 2
    assert padding[0] >= 0 and type(padding[0]) == int
    assert padding[1] >= 0 and type(padding[1]) == int

    if isinstance(stride, int):
        stride = [stride, stride]
    else:
        assert len(stride) == 2
    assert stride[0] > 0 and type(stride[0]) == int
    assert stride[1] > 0 and type(stride[1]) == int

    output_size = get_conv2d_output_shape(input, kernel, padding, stride)
    output_tensor = torch.empty(output_size, dtype=input.dtype, requires_grad=False)

    return core.kn2row_conv2d(input.untyped_storage(), input.shape,
                              kernel.untyped_storage(), kernel.shape,
                              str(input.dtype),
                              bias_ptr,
                              padding,
                              stride,
                              output_tensor)
