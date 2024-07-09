from typing import Sequence
import inspect
import torch
import os

GROUPED_CONVOLUTION = False


def wrapped_run(module: torch.nn.Module, input_sample_shape: Sequence[int], runner):
    name, _ = os.path.splitext(os.path.basename(caller_name()))
    name = name.upper()
    print(f"{name}")
    module.eval()
    if no_grouped_convolution(module):
        runner(module, input_sample_shape, name)
    else:
        print(f"  skipping {name} as it requires groups > 1")


def caller_name():
    caller_frame = inspect.stack()[2]
    name = caller_frame.filename
    return name


def no_grouped_convolution(module: torch.nn.Module):
    grouped_conv = False
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Conv2d):
            if submodule.groups > 1:
                grouped_conv = True
    if grouped_conv and not GROUPED_CONVOLUTION:
        return False
    return True
