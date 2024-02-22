from torch import (
        UntypedStorage,
        Tensor,
        Size,
        )
from typing import (
        Optional
        )


def kn2row_conv2d(input: UntypedStorage, input_shape: Size,
                  weight: UntypedStorage, weight_shape: Size, dtype: str,
                  bias: Optional[UntypedStorage], output: Tensor) -> Tensor:
    ...

# def kn2row_conv2d(input: UntypedStorage, weight: UntypedStorage, dtype: str,
#                   bias: Optional[UntypedStorage] = None,
#                   stride: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]] = 1,
#                   padding: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]] = 0,
#                   dilation: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]] = 1,
#                   groups: Union[_int, SymInt] = 1) -> None:
#     ...
