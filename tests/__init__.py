import torch
import numpy as np
import atexit
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

def compare_tensors(out_tensor, tar_tensor: torch.Tensor, mes: str, atol=1e-5) -> None:
    out = np.array(out_tensor.data).reshape(out_tensor.shape)
    tar = np.array(tar_tensor)

    if tar.shape != out.shape:
        add_fail(mes)
        print(
            f'Failed Test `{mes}`, Tensors have different shapes, target: {tar.shape} and output {out.shape}')
        return

    different_elements = np.where(np.abs(out - tar) > atol)

    if len(different_elements[0]) == 0:
        print(f'Passed Test {mes}')
    else:
        add_fail(mes)
        print(f'Failed Test {mes}')
        print('  Tensors differ at the following indices:')
        for index in zip(*different_elements):
            print('  at:', index, 'target:', tar[index], 'output:', out[index])

def calling_file():
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back.f_back.f_back
    caller_file = caller_frame.f_globals["__file__"]

    return caller_file

