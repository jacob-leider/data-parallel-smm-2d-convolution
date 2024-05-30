import torch # TODO could put code that accesses torch all in one file to cleanup to maybe allow easior conditional backends between onnx and torch
from typing import (
    Union,
    Sequence
)

def get_correct_from_type(dtype, float_item, double_item):
    if str(dtype) == 'torch.float32':
        return float_item
    if str(dtype) == 'torch.float64':
        return double_item
    assert False, f'using bad dtype: {str(dtype)}'

def get_address(frontend_data) -> int:
    if isinstance(frontend_data, torch.Tensor):
        return frontend_data.data_ptr()
    assert False and 'bad input type'

def get_shape(frontend_data) -> Sequence[int]:
    if isinstance(frontend_data, torch.Tensor):
        return frontend_data.size()
    assert False and 'bad input type'

def make_padding_2d(padding: Union[str, Union[int, Sequence[int]]],
                    stride: Sequence[int], dilation: Sequence[int],
                    kernel_shape: Sequence[int], dtype=int) -> Sequence[int]:
    if isinstance(padding, str):
        if padding == 'valid':
            return [0, 0]
        if padding == 'same':
            assert all(x == 1 for x in stride)
            k_height = kernel_shape[len(kernel_shape) - 2]
            k_width = kernel_shape[len(kernel_shape) - 1]
            return [dilation[0] * (k_height - 1) // 2, dilation[1] * (k_width - 1) // 2]
        assert False and f'invalid padding string: {padding}'
    else:
        return make_2d(padding, dtype=dtype)


def make_2d(a: Union[int, Sequence[int]], dtype=int) -> Sequence[int]:
    if isinstance(a, dtype):
        return [a, a]
    assert len(a) == 2 and all(isinstance(val, dtype) for val in a)
    return a

