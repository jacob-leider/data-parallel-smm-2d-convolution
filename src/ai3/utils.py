from typing import (
    Union,
    Sequence,
    Any,
    Tuple,
    Mapping,
    Callable,
    Optional
)
import torch
import inspect
from ai3 import errors

FLOAT32_STR = "float32"
FLOAT64_STR = "float64"


def check_callable_params_with_shape(
        algos: Mapping[str, Union[str, Sequence[str],
                                  Callable]],
        sample_input_shape: Optional[Sequence[int]] = None):
    for key, value in algos.items():
        if callable(value):
            if len(inspect.signature(value).parameters) == 2:
                errors.bail_if(
                    sample_input_shape is None,
                    f'for {key} using function selector which depends on input shape, but no sample input shape provided')
            else:
                errors.bail_if(
                    sample_input_shape is not None,
                    f'provided sample input shape but function selector for {key} doesn\'t take an input shape')


def get_item(dtype, float_item, double_item):
    if str(dtype) == 'torch.float32':
        return float_item
    if str(dtype) == 'torch.float64':
        return double_item
    assert False, f'using bad dtype: {str(dtype)}'


def get_address(frontend_data) -> int:
    if isinstance(frontend_data, torch.Tensor):
        return frontend_data.data_ptr()
    assert False and 'bad input type when getting data address'


def get_shape(frontend_data) -> tuple:
    if isinstance(frontend_data, torch.Tensor):
        return frontend_data.size()
    assert False and 'bad input type when getting shape'


def make_padding_2d(padding: Union[str, Union[int, Tuple[int, ...]]],
                    stride: Tuple[int, int], dilation: Tuple[int, int],
                    kernel_shape: Sequence[int], dtype=int) -> tuple[int, int]:
    if isinstance(padding, str):
        if padding == 'valid':
            return (0, 0)
        if padding == 'same':
            assert all(x == 1 for x in stride)
            k_height = kernel_shape[len(
                kernel_shape) - 2]
            k_width = kernel_shape[len(
                kernel_shape) - 1]
            return (dilation[0] * (k_height - 1) // 2, dilation[1] * (k_width - 1) // 2)
        assert False and f'invalid padding string: {padding}'
    else:
        return make_2d(padding, dtype=dtype)


def make_2d(a: Union[Any, Tuple[Any, Any]],
            dtype: type = int) -> Tuple[Any, Any]:
    if isinstance(a, dtype):
        return (a, a)
    assert len(a) == 2 and all(
        isinstance(val, dtype) for val in a)
    return a
