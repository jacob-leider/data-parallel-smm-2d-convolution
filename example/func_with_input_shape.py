import torch
import ai3
from .manual_conv2d import ConvNet
from typing import Sequence


def conv2d_selector(orig: torch.nn.Conv2d, input_shape: Sequence[int]) -> str:
    out_channels = orig.weight.shape[0]
    if out_channels < 50 and input_shape[0] < 50 and input_shape[1] > 150 and input_shape[2] > 150:
        return "direct"
    return 'smm'


input_data = torch.randn(10, 3, 224, 224)
orig = ConvNet()
orig.eval()

with torch.inference_mode():
    torch_out = orig(input_data)
    ai3.swap_conv2d(
        orig, conv2d_selector, (3, 224, 224))
    swap_out = orig(input_data)
    assert torch.allclose(
        torch_out, swap_out, atol=1e-6)
    model = ai3.swap_backend(
        orig, {'conv2d': conv2d_selector}, (3, 224, 224))
    ai3_out = model.predict(input_data)
    assert isinstance(ai3_out, ai3.Tensor)
    assert torch.allclose(
        torch_out, ai3_out.torch(), atol=1e-6)
