import torch
import ai3
from typing import Optional, Callable
from test import compare_tensors


def run():
    print('MODELS SWAPPING CONV2D')
    from . import (vgg16, alexnet, convnext, densenet, efficientnet, googlenet,
                   inception, shufflenetv2, mobilenet, mnasnet, squeezenet, vision_transformer,
                   swin_transformer, maxvit, regnet, resnet)
    vgg16.run()
    alexnet.run()
    convnext.run()
    densenet.run()
    efficientnet.run()
    googlenet.run()
    inception.run()
    regnet.run()
    resnet.run()
    mobilenet.run()
    mnasnet.run()
    shufflenetv2.run()
    squeezenet.run()
    vision_transformer.run()
    swin_transformer.run()
    maxvit.run()


def swap_and_compare(orig: torch.nn.Module, input: torch.Tensor, mes: str, selector_func: Optional[Callable] = None):
    with torch.inference_mode():
        torch_out = orig(input)

        ai3.swap_conv2d(orig, selector_func)
        ai3_out = orig(input)

    assert (isinstance(torch_out, torch.Tensor))
    compare_tensors(ai3_out, torch_out, mes)
