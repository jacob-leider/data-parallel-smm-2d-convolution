import torch
import ai3
from bench import predict_show_time
from test import compare_tensors
from runners import BATCH

def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    torch_out = predict_show_time(module, input_data, name + " torch")

    for algo in ['default', 'direct', 'smm']:
        ai3.swap_conv2d(module, algo)
        ai3_out = predict_show_time(module, input_data, f"{name} ai3 {algo}")
        compare_tensors(ai3_out, torch_out,
                        f"{name} ai3 {algo}, {BATCH} samples", print_pass=False)
