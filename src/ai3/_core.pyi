from collections.abc import Buffer
from typing import Sequence, Optional
from enum import Enum

def using_mps() -> bool:
    ...
def using_cuda_tools() -> bool:
    ...
def using_sycl() -> bool:
    ...

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

class Model():
    def __init__(self, layers: Sequence):
        ...

    def predict(self, input_address: int, input_shape: Sequence[int]):
        ...

class Model_float(Model):
    ...
class Model_double(Model):
    ...

def output_hw_for_2d(input: int, kernel: int,
                               padding: int ,
                               dilation: Optional[int], stride):
    ...


def conv2d_float(input_address: int, input_shape: Sequence[int],
                 weight_address: int, weight_shape: Sequence[int], bias_addr:
                 Optional[int], padding_h: int, padding_w: int, stride_h: int,
                 stride_w: int, dilation_h: int, dilation_w: int, padding_mode:
                 int, groups: int, algorithm: str) -> Tensor_float:
    ...

def conv2d_double(input_address: int, input_shape: Sequence[int],
                 weight_address: int, weight_shape: Sequence[int], bias_addr:
                 Optional[int], padding_h: int, padding_w: int, stride_h: int,
                 stride_w: int, dilation_h: int, dilation_w: int, padding_mode:
                 int, groups: int, algorithm: str) -> Tensor_double:
    ...

class Conv2D():
    algorithm: str
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int],
                 padding_h: int,
                 padding_w: int,
                 stride_h: int,
                 stride_w: int,
                 dilation_h: int,
                 dilation_w: int,
                 padding_mode: PaddingMode,
                 groups: int,
                 algorithm: str):
        ...

    def forward(self, input_address: int, input_shape: Sequence[int]):
        ...

class Conv2D_double(Conv2D):
    ...
class Conv2D_float(Conv2D):
    ...

class MaxPool2D():
    def __init__(self,
                 kernel_h: int,
                 kernel_w: int,
                 padding_h: int,
                 padding_w: int,
                 stride_h: int,
                 stride_w: int,
                 dilation_h: int,
                 dilation_w: int,
                 ceil_mode: bool,
                 algorithm: str):
        ...

class MaxPool2D_float(MaxPool2D):
    ...
class MaxPool2D_double(MaxPool2D):
    ...

class AvgPool2D():
    def __init__(self,
                 kernel_h: int,
                 kernel_w: int,
                 padding_h: int,
                 padding_w: int,
                 stride_h: int,
                 stride_w: int,
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int],
                 algorithm: str):
        ...

class AvgPool2D_float(AvgPool2D):
    ...
class AvgPool2D_double(AvgPool2D):
    ...

class AdaptiveAvgPool2D():
    def __init__(self, output_h: Optional[int], output_w: Optional[int], algorithm:str):
        ...

class AdaptiveAvgPool2D_float(AdaptiveAvgPool2D):
    ...
class AdaptiveAvgPool2D_double(AdaptiveAvgPool2D):
    ...

class Linear:
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int],
                 algorithm: str):
        ...

class Linear_double(Linear):
    ...

class Linear_float(Linear):
    ...

class ReLU():
    def __init__(self, algorithm: str):
        ...

class ReLU_float(ReLU):
    ...
class ReLU_double(ReLU):
    ...

class Flatten():
    def __init__(self, start_dim: int, end_dim: int, algorithm: str):
        ...

class Flatten_double(Flatten):
    ...
class Flatten_float(Flatten):
    ...
