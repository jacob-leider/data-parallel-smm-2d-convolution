import torch
import time
import ai3
from test import compare_tensors


def _run(orig_torch):
    model = ai3.Model(torch.get_default_dtype(), [])
    tens= model.predict(orig_torch)
    assert(isinstance(tens, ai3.Tensor))
    start = time.time()
    back_to_torch = tens.torch()
    end = time.time()
    print(
        f" {orig_torch.size()} ai3 -> torch: {end-start}")
    start = time.time()
    as_numpy = tens.numpy()
    end = time.time()
    print(
        f" {orig_torch.size()} ai3 -> numpy: {end-start}")
    compare_tensors(back_to_torch, orig_torch)
    compare_tensors(as_numpy, orig_torch)


def run():
    print("Tensor Type Change")
    _run(torch.randn(1))
    _run(torch.randn(2, 1000, 1000))
    _run(torch.randn(100, 2, 1000, 1000))


if __name__ == "__main__":
    run()
