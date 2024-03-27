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
                  stride: Sequence[int],
                  dilation: Sequence[int],
                  output: Tensor) -> Tensor:
    ...
