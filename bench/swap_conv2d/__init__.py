import torch
import ai3
from bench import predict_show_time
from test import compare_tensors, BATCH
from typing import Sequence


def run():
    print('SWAPPING CONV2D')
    from . import (vgg16, alexnet, convnext, densenet, efficientnet, googlenet,
                   inception, shufflenetv2, mobilenet, mnasnet, squeezenet, vision_transformer,
                   swin_transformer, maxvit, regnet, resnet)


def runner(module: torch.nn.Module, input_sample_shape: Sequence[int], name: str):
    input_data = torch.randn(BATCH, *input_sample_shape)
    torch_out = predict_show_time(module, input_data, name + " torch")

    ai3.swap_conv2d(module, "default")
    ai3_out = predict_show_time(module, input_data, name + " ai3 default")

    ai3.swap_conv2d(module, "smm")
    ai3_out = predict_show_time(module, input_data, name + " ai3 smm")

    assert (isinstance(torch_out, torch.Tensor))
    compare_tensors(ai3_out, torch_out, "")
