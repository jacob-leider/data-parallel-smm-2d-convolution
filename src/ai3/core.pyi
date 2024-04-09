from torch import (
        UntypedStorage,
        Tensor,
        )
from typing import (
        Optional, Sequence
        )

def kn2row_conv2d(input: UntypedStorage, input_shape: Sequence[int],
                  kernel: UntypedStorage, kernel_shape: Sequence[int],
                  bias: Optional[UntypedStorage], output_shape: Sequence[int], dtype: str, padding: Sequence[int],
                  stride: Sequence[int],
                  dilation: Sequence[int]) -> Tensor:
    ...
