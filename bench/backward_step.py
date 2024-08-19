import torch
from torch import nn
from torch import optim
import ai3
import models
import sys
import time

criterion = nn.CrossEntropyLoss()


def time_backward_step(module, input_data, name: str):
    input_data.requires_grad_(True)
    # module.train() # CHECK for some reason this breaks inception check later
    optimizer = optim.Adam(  # type: ignore
        module.parameters(), lr=0.01)
    optimizer.zero_grad()
    out = module(input_data)
    tar = torch.randn(out.shape)
    loss = criterion(out, tar)

    start = time.time()
    loss.backward()
    end = time.time()
    backward_time = end - start
    print(f"  Time Backward {name}: {backward_time:.4f} seconds")

    start = time.time()
    optimizer.step()
    end = time.time()
    backward_time = end - start
    print(f"  Time step {name}: {backward_time:.4f} seconds")


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    time_backward_step(module, input_data, f'{name} torch')
    ai3.swap_conv2d(module)
    time_backward_step(module, input_data, f'{name} ai3')


if __name__ == "__main__":
    print('BACKWARD')
    models.from_args(runner, sys.argv)