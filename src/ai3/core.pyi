from collections.abc import Buffer
from typing import Sequence, Union, Optional
from enum import Enum

class PaddingMode(Enum):
    zeros: int
    reflect: int
    replicate: int
    circular: int

class Tensor_float(Buffer):
    data: Sequence[float]
    shape: Sequence[int]

class Tensor_double(Buffer):
    data: Sequence[float]
    shape: Sequence[int]

class Model_double():
    def __init__(self, layers: Sequence):
        ...

    def predict(self, input_address: int, input_shape: Sequence[int]):
        ...

class Model_float():
    def __init__(self, layers: Sequence):
        ...

    def predict(self, input_address: int, input_shape: Sequence[int]):
        ...

class Conv2D_double():
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int],
                 padding: Sequence[int],
                 stride: Sequence[int],
                 dilation: Sequence[int],
                 padding_mode: PaddingMode,
                 groups: int):
        ...

    def forward(self, input_address: int, input_shape: Sequence[int]):
        ...

class Conv2D_float():
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int],
                 padding: Sequence[int],
                 stride: Sequence[int],
                 dilation: Sequence[int],
                 padding_mode: PaddingMode,
                 groups: int):
        ...

    def forward(self, input_address: int, input_shape: Sequence[int]):
        ...

class MaxPool2D_double():
    def __init__(self,
                 kernel_shape: Sequence[int],
                 padding: Sequence[int],
                 stride: Sequence[int],
                 dilation: Sequence[int],
                 ceil_mode: bool):
        ...

class MaxPool2D_float():
    def __init__(self,
                 kernel_shape: Sequence[int],
                 padding: Sequence[int],
                 stride: Sequence[int],
                 dilation: Sequence[int],
                 ceil_mode: bool):
        ...

class AvgPool2D_float():
    def __init__(self,
                 kernel_shape: Sequence[int],
                 padding: Sequence[int],
                 stride: Sequence[int],
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int]):
        ...

class AvgPool2D_double():
    def __init__(self,
                 kernel_shape: Sequence[int],
                 padding: Sequence[int],
                 stride: Sequence[int],
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int]):
       ...

class AdaptiveAvgPool2D_float():
    def __init__(self, output_shape: Optional[Union[Sequence[Optional[int]], Sequence[int]]]):
        ...

class AdaptiveAvgPool2D_double():
    def __init__(self, output_shape: Optional[Union[Sequence[Optional[int]], Sequence[int]]]):
        ...

class Linear_double():
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int]):
        ...

class Linear_float():
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int]):
        ...

class ReLU_double():
    def __init__(self):
        ...

class ReLU_float():
    def __init__(self):
        ...

class Flatten_double():
    def __init__(self, start_dim: int, end_dim: int):
        ...

class Flatten_float():
    def __init__(self, start_dim: int, end_dim: int):
        ...
