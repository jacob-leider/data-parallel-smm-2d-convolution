import torch
import ai3
from ai3.errors import UnsupportedCallableError
from test import compare_tensors
import models
import sys


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    algos = ['default', 'direct', 'smm']
    if torch.backends.cudnn.is_available():
        algos.extend(["implicit precomp gemm",
                     "implicit gemm", "gemm", "guess"])
    with torch.inference_mode():
        for algo in algos:
            try:
                ai3_model = ai3.swap_backend(
                    module, {"conv2d": algo})
            except UnsupportedCallableError as e:
                print(f"  {e} so skipping")
                return
            output = ai3_model(input_data)
            compare_tensors(
                output, target,
                f"{name} swap backend, {models.BATCH} samples")


if __name__ == "__main__":
    models.from_args(runner, sys.argv)
