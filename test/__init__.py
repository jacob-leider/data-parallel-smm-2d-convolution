from typing import Optional
import torch
import numpy as np
import atexit
import torch
from ai3 import utils
import inspect

FAILED_TESTS = []


def show_failed():
    if FAILED_TESTS:
        print(f"Failed {len(FAILED_TESTS)} tests:")
        for test in FAILED_TESTS:
            print(f"  - {test}")


atexit.register(show_failed)


def add_fail(mes):
    FAILED_TESTS.append(f"{mes} from: {calling_file()}")


def compare_tensors(out_tensor, tar_tensor: torch.Tensor, mes: Optional[str] = None, atol=1e-5) -> None:
    if isinstance(out_tensor, np.ndarray):
        out = out_tensor
    elif isinstance(out_tensor, torch.Tensor):
        out = np.array(out_tensor)
    else:
        out = utils.tensor_to_type(out_tensor, np.ndarray)
    tar = np.array(tar_tensor)
    assert (isinstance(out, np.ndarray))

    if np.isnan(tar).any():
        add_fail(mes)
        print(f'Failed Test `{mes}`, target has NaNs')
        return

    if np.isnan(out).any():
        add_fail(mes)
        print(f'Failed Test `{mes}`, output has NaNs')
        return
    if tar.shape != out.shape:
        add_fail(mes)
        print(
            f'Failed Test `{mes}`, Tensors have different shapes, target: {tar.shape} and output {out.shape}')
        return

    different_elements = np.where(np.abs(out - tar) > atol)

    if len(different_elements[0]) == 0:
        if mes:
            print(f'  Passed Test {mes}')
    else:
        add_fail(mes)
        print(f'Failed Test {mes}')
        print('  Tensors differ at the following indices:')
        for index in zip(*different_elements):
            print('  at:', index, 'target:', tar[index], 'output:', out[index])


def calling_file():
    current_frame = inspect.currentframe()
    if current_frame is None:
        return
    caller_frame = current_frame.f_back
    if caller_frame is None:
        return None
    caller_frame = caller_frame.f_back
    if caller_frame is None:
        return None
    caller_frame = caller_frame.f_back
    if caller_frame is None:
        return None
    caller_file = caller_frame.f_globals["__file__"]

    return caller_file
