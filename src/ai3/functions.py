import torch
from ai3 import core
from typing import (
    Optional,
)

def get_conv2d_output_shape(input: torch.Tensor, kernel: torch.Tensor) -> tuple:
    # same
    # output_height = input_height;
    # output_width = input_width;
    # full
    # output_height = input_height + kernel_height - 1;
    # output_width = input_width + kernel_width - 1;
    # valid
    batch_size, _, input_height, input_width = input.shape
    output_channels, _, kernel_height, kernel_width = kernel.shape
    output_height = input_height - kernel_height + 1;
    output_width = input_width - kernel_width + 1;
    return (batch_size, output_channels, output_height, output_width)

# TODO when doing training step need to have bias True or False if True then give a tensor of (out_channels) shape
def conv2d(input: torch.Tensor, kernel: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert input.dtype == kernel.dtype
    if bias is not None:
        assert input.dtype == bias.dtype
        bias_ptr = bias.untyped_storage()
    else:
        bias_ptr = None

    output_size = get_conv2d_output_shape(input, kernel)
    output_tensor = torch.empty(output_size, dtype=input.dtype, requires_grad=False)

    return core.kn2row_conv2d(input.untyped_storage(), input.shape,
                              kernel.untyped_storage(), kernel.shape,
                              str(input.dtype),
                              bias_ptr, output_tensor)
