import torch
import ai3
from bench import predict_show_time
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


def swap_conv2d_and_time(orig: torch.nn.Module, input: torch.Tensor):
    torch_out = predict_show_time(orig, input, "pytorch")

    ai3.swap_conv2d(orig)
    assert (isinstance(torch_out, torch.Tensor))
    ai3_out = predict_show_time(orig, input, "ai3")
    compare_tensors(ai3_out, torch_out, "")
