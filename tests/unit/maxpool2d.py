import torch
from torch import nn
from ai3.model import Model, MaxPool2D
from tests import compare_tensors
from typing import Union, Sequence, Optional

def test(*, input_channels: int, in_height: int, in_width: int,
         kernel_height: int, kernel_width: int,
         padding: Union[int, Sequence[int]] = 0,
         dilation: Union[int, Sequence[int]] = 1,
         stride: Optional[Union[int, Sequence[int]]] = None,
         test_name: str, atol=1e-5) -> None:
    input = torch.randn(input_channels, in_height, in_width,dtype=torch.float32)
    kernel_shape = (kernel_height, kernel_width)

    if stride is None:
        stride = kernel_shape

    model = Model(input.dtype, [MaxPool2D(input.dtype, kernel_shape,
                                                 stride, padding, dilation)])
    ai3_output = model.predict(input)
    torch_output = nn.MaxPool2d(kernel_shape, dilation=dilation,
                             padding=padding, stride=stride)(input)
    compare_tensors(ai3_output, torch_output, test_name, atol=atol)


def run():
    print('MAX POOL 2D')
    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=(2, 2),
         atol=1e-4,
         test_name='same odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=8,
         kernel_width=4,
         dilation=(2, 2),
         atol=1e-4,
         test_name='same even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=8,
         kernel_width=4,
         dilation=(1, 2),
         atol=1e-4,
         test_name='valid even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=(1, 2),
         atol=1e-4,
         test_name='valid odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=(1, 2),
         atol=1e-4,
         test_name='2d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         dilation=3,
         atol=1e-4,
         test_name='1d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         padding=2,
         atol=1e-4,
         test_name='1d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=10,
         padding=(2, 5),
         atol=1e-4,
         test_name='2d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         stride=2,
         atol=1e-4,
         test_name='1d stride')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         kernel_height=7,
         kernel_width=5,
         stride=(2, 3),
         atol=1e-4,
         test_name='2d stride')

    test(input_channels=1,
         in_height=5,
         in_width=5,
         kernel_height=3,
         kernel_width=3,
         test_name='basic')

    test(input_channels=1,
         in_height=10,
         in_width=15,
         kernel_height=10,
         kernel_width=15,
         atol=1e-4,
         test_name='kern.shape = input.shape')

    test(input_channels=3,
         in_height=50,
         in_width=150,
         kernel_height=10,
         kernel_width=10,
         atol=1e-4,
         test_name='multi channel')


if __name__ == "__main__":
    run()
