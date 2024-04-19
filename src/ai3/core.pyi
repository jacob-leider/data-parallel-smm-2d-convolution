from typing import Optional, Sequence

class Tensor_float:
    def __init__(self, d: Sequence[float], s: Sequence[int]) -> None:
        ...

class Tensor_double:
    def __init__(self, d: Sequence[float], s: Sequence[int]) -> None:
        ...

# TODO put some types here
class Model_double():
    def __init__(self, layers):
        ...

    def predict(self, input_address: int, input_shape: Sequence[int]):
        ...

class Model_float():
    def __init__(self, layers):
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
