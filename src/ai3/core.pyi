from torch import (
        UntypedStorage,
        Tensor,
        Size,
        )
from typing import (
        Optional, Sequence
        )


def kn2row_conv2d(input: UntypedStorage, input_shape: Size,
                  kernel: UntypedStorage, kernel_shape: Size, dtype: str,
                  bias: Optional[UntypedStorage], padding: Sequence[int],
                  output: Tensor) -> Tensor:
    ...


# def kn2row_conv2d(input: UntypedStorage, weight: UntypedStorage, dtype: str,
#                   bias: Optional[UntypedStorage] = None,
#                   stride: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]] = 1,
#                   padding: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]] = 0,
#                   dilation: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]] = 1,
#                   groups: Union[_int, SymInt] = 1) -> None:
#     ...
