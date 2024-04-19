import torch
import torch.nn.functional as F
from ai3.functions import form_conv2d, form_model, predict
from tests import compare_tensors
from typing import Union, Sequence

def test(*, input_channels: int, in_height: int, in_width: int,
         output_channels: int, kernel_height: int, kernel_width: int,
         with_bias: bool = False,
         padding: Union[str, Union[int, Sequence[int]]] = 0,
         dilation: Union[int, Sequence[int]] = 1,
         stride: Union[int, Sequence[int]] = 1,
         groups: int = 1,
         test_name: str, atol=1e-5) -> None:
    input = torch.randn(input_channels, in_height, in_width)
    kernel = torch.randn(output_channels, input_channels // groups,
                         kernel_height, kernel_width)
    assert (str(input.dtype) == "torch.float32") and (str(kernel.dtype) == "torch.float32")
    if with_bias:
        bias = torch.randn(output_channels)
    else:
        bias = None

    model = form_model(input.dtype, [form_conv2d(input.dtype, kernel, bias,
                                                 stride=stride, padding=padding, dilation=dilation)])
    ai3_output = predict(model, input)
    torch_output = F.conv2d(input, kernel, bias=bias, dilation=dilation,
                            padding=padding, stride=stride, groups=groups)
    compare_tensors(ai3_output, torch_output, test_name, atol=atol)


def run():
    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding='same',
         dilation=(2, 2),
         atol=1e-4,
         test_name='same odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=8,
         kernel_width=4,
         with_bias=True,
         padding='same',
         dilation=(2, 2),
         atol=1e-4,
         test_name='same even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=8,
         kernel_width=4,
         with_bias=True,
         padding='valid',
         dilation=(1, 2),
         atol=1e-4,
         test_name='valid even kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding='valid',
         dilation=(1, 2),
         atol=1e-4,
         test_name='valid odd kernel')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         dilation=(1, 2),
         atol=1e-4,
         test_name='2d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         dilation=3,
         atol=1e-4,
         test_name='1d dilation')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding=5,
         atol=1e-4,
         test_name='1d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         padding=(2, 5),
         atol=1e-4,
         test_name='2d padding')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         stride=2,
         atol=1e-4,
         test_name='1d stride')

    test(input_channels=4,
         in_height=30,
         in_width=40,
         output_channels=6,
         kernel_height=7,
         kernel_width=5,
         with_bias=True,
         stride=(2, 3),
         atol=1e-4,
         test_name='2d stride')

    test(input_channels=1,
         in_height=5,
         in_width=5,
         output_channels=1,
         kernel_height=3,
         kernel_width=3,
         test_name='basic')

    test(input_channels=1,
         in_height=100,
         in_width=150,
         output_channels=1,
         kernel_height=15,
         kernel_width=12,
         with_bias=True,
         atol=1e-4,
         test_name='with bias')

    test(input_channels=1,
         in_height=10,
         in_width=15,
         output_channels=1,
         kernel_height=10,
         kernel_width=15,
         with_bias=True,
         atol=1e-4,
         test_name='kern.shape = input.shape')

    test(input_channels=3,
         in_height=50,
         in_width=150,
         output_channels=4,
         kernel_height=10,
         kernel_width=10,
         atol=1e-4,
         test_name='multi channel')

    test(input_channels=4,
         in_height=50,
         in_width=150,
         output_channels=6,
         kernel_height=10,
         kernel_width=10,
         with_bias=True,
         atol=1e-4,
         test_name='multi channel with bias')

    test(input_channels=4,
         in_height=50,
         in_width=150,
         output_channels=6,
         kernel_height=10,
         kernel_width=10,
         with_bias=True,
         atol=1e-4,
         test_name='multi channel with bias')


if __name__ == "__main__":
    run()
