import torch
from torch import nn
from ai3 import Model
from ai3.layers import AdaptiveAvgPool2D
from test import compare_tensors


def test(*, input_channels: int, in_height: int, in_width: int,
         output_shape,
         test_name: str, atol=1e-5) -> None:
    input = torch.randn(input_channels, in_height,
                        in_width, dtype=torch.float32)

    model = Model(input.dtype, [AdaptiveAvgPool2D(input.dtype, output_shape)])
    ai3_output = model.predict(input)
    torch_output = nn.AdaptiveAvgPool2d(output_shape)(input)
    compare_tensors(ai3_output, torch_output, test_name, atol=atol)


def run():
    print('ADAPTIVE AVG POOL 2D')
    test(input_channels=3,
         in_height=30,
         in_width=30,
         output_shape=(6, 6),
         test_name="out is multiple of in")
    test(input_channels=3,
         in_height=40,
         in_width=30,
         output_shape=(4, 3),
         test_name="separate multiples")
    # TODO avgpool2d non-multiples should be implemented at some point
    # test(input_channels=3,
    #      in_height=41,
    #      in_width=32,
    #      output_shape=(4,3),
    #      test_name="not multiples")


if __name__ == "__main__":
    run()