from typing import Optional, Sequence

# TODO try to do generics if those fail then
# TODO probably better to have wrapper classes and have these behind like a ._self or something
# use those instead of the form_ functions in functions.py

class Tensor_float:
    def __init__(self, d: Sequence[float], s: Sequence[int]) -> None:
        ...

class Tensor_double:
    def __init__(self, d: Sequence[float], s: Sequence[int]) -> None:
        ...

class Model_double():
    def __init__(self, layers):
        ...

    def _predict(self, input_address: int, input_shape: Sequence[int]):
        ...

class Model_float():
    def __init__(self, layers):
        ...

    def _predict(self, input_address: int, input_shape: Sequence[int]):
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

# # Define a generic type for the data type of the tensor
# dtype = TypeVar('dtype')
#
# class Tensor:
#     def __init__(self, d: List[dtype], s: List[int]) -> None:
#         self.data = d
#         self.shape = s
#
#     def __getitem__(self, index: int) -> dtype:
#         return self.data[index]
