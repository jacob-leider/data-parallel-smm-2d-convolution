import torch
import ai3
import runners
import sys
from test import compare_tensors


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    algos = ['default', 'direct', 'smm']
    if torch.backends.cudnn.is_available():
        algos.extend(["implicit precomp gemm",
                     "implicit gemm", "gemm", "guess"])
    with torch.inference_mode():
        for algo in algos:
            ai3.swap_conv2d(module, algo)
            output = module(input_data)
            compare_tensors(
                output, target,
                f"{name} swap conv2d with {algo}, {runners.BATCH} samples")


if __name__ == "__main__":
    runners.from_args(runner, sys.argv)
