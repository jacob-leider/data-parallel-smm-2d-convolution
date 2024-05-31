from typing import Optional, Sequence, Union

Possible_Layers = Union[Conv2D_double, Conv2D_float]

class Tensor_float:
    def __init__(self, d: Sequence[float], s: Sequence[int]) -> None:
        ...

class Tensor_double:
    def __init__(self, d: Sequence[float], s: Sequence[int]) -> None:
        ...

class Model_double():
    def __init__(self, layers: Sequence[Possible_Layers]):
        ...

    def predict(self, input_address: int, input_shape: Sequence[int]):
        ...

class Model_float():
    def __init__(self, layers: Sequence[Possible_Layers]):
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
                 dilation: Sequence[int]):
        ...
class Conv2D_float():
    def __init__(self,
                 weight_address: int,
                 weight_shape: Sequence[int],
                 bias_addr: Optional[int],
                 padding: Sequence[int],
                 stride: Sequence[int],
                 dilation: Sequence[int]):
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
