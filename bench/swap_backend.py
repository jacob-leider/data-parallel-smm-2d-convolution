import torch
import ai3
from ai3.errors import UnsupportedCallableError
from test import compare_tensors
from bench import predict_show_time
import models
import sys


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = predict_show_time(
        module, input_data, name + " torch")
    try:
        ai3_model = ai3.swap_backend(module)
    except UnsupportedCallableError as e:
        print(f"  {e} so skipping")
        return
    output = predict_show_time(
        ai3_model, input_data, name + "ai3")
    compare_tensors(
        output, target, f'{name} ai3, {models.BATCH} samples',
        print_pass=False)


if __name__ == "__main__":
    models.from_args(runner, sys.argv)
