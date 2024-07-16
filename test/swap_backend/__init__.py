import torch
import ai3
from ai3.errors import UnsupportedCallableError
from test import compare_tensors
from runners import BATCH


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    with torch.inference_mode():
        for algo in ['default', 'direct', 'smm']:
            try:
                ai3_model = ai3.swap_backend(module, {"conv2d": algo})
            except UnsupportedCallableError as e:
                print(f"  {e} so skipping")
                return
            output = ai3_model(input_data)
            compare_tensors(
                output, target, f"{name} swap backend, {BATCH} samples")
