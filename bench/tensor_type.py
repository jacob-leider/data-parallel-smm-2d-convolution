import torch
import time
import ai3
from test import compare_tensors


def _run(orig_torch):
    model = ai3.Model(torch.get_default_dtype(), [])
    tens = model.predict(orig_torch)
    assert (isinstance(tens, ai3.core.Tensor_float)
            or isinstance(tens, ai3.core.Tensor_double))
    start = time.time()
    back_to_torch = ai3.utils.tensor_to_type(tens, torch.Tensor)
    end = time.time()
    print(
        f" {orig_torch.size()} ai3 -> torch: {end-start}")
    compare_tensors(back_to_torch, orig_torch)


def run():
    print("Tensor Type Change")
    _run(torch.randn(1))
    _run(torch.randn(2, 1000, 1000))
    _run(torch.randn(100, 2, 1000, 1000))


if __name__ == "__main__":
    run()
