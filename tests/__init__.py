import torch
import numpy as np


def compare_tensors(out_tensor, tar_tensor: torch.Tensor, mes: str, atol=1e-5) -> None:
    out = np.array(out_tensor.data).reshape(out_tensor.shape)
    tar = np.array(tar_tensor)

    if tar.shape != out.shape:
        print(
            f'For test `{mes}`, Tensors have different shapes, target: {tar.shape} and output {out.shape}')
        return

    different_elements = np.where(np.abs(out - tar) > atol)

    if len(different_elements[0]) == 0:
        print(f'Passed Test {mes}')
    else:
        print(f'Failed Test {mes}')
        print('Tensors differ at the following indices:')
        for index in zip(*different_elements):
            print('at:', index, 'target:', tar[index], 'output:', out[index])
