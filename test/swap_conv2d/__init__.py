import torch
import ai3
from test import compare_tensors
from runners import BATCH


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    with torch.inference_mode():
        for algo in ['default', 'direct', 'smm']:
            ai3.swap_conv2d(module, algo)
            output = module(input_data)
            compare_tensors(
                output, target, f"{name} swap conv2d with {algo}, {BATCH} samples")
