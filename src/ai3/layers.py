from ai3 import core, utils
import torch
from torch import fx, nn
from typing import (
    Union,
    Sequence,
    Optional,
    List
)
from abc import ABC


def issequential(name: str) -> bool:
    return '.' in name

def getmodule(module: nn.Module, name: str) -> nn.Module:
    if issequential(name):
        names = name.split('.', 1)
        return getmodule(getattr(module, names[0]), names[1])
    else:
        return getattr(module, name)

class Layer(ABC):
    def __init__(self, core):
        self.core = core
        ...

def get_layers(module: nn.Module, dtype) -> List[Layer]:
    gm: fx.GraphModule = fx.symbolic_trace(module)
    layers = []
    for node in gm.graph.nodes:
        if node.op == 'placeholder' or node.op == 'output':
            pass
        elif node.op == 'call_function':
            if node.target == torch.flatten:
                start_dim = 0
                end_dim = -1
                if len(node.args) > 1:
                    start_dim = node.args[1]
                if len(node.args) > 2:
                    end_dim = node.args[2]
                if 'start_dim' in node.kwargs:
                    start_dim = node.kwargs['start_dim']
                if 'end_dim' in node.kwargs:
                    end_dim = node.kwargs['end_dim']
                layers.append(Flatten(dtype, start_dim, end_dim))
            elif node.target == torch.relu:
                layers.append(ReLU(dtype))
            else:
                utils.bail(f"unsupported function: {node.target}")
        elif node.op == 'call_module':
            mod = getmodule(module, node.target)
            if not isinstance(mod, nn.Dropout):
                swapped = swap(mod, dtype)
                if not swapped:
                    utils.bail(f"unsupported module: {mod}")
                layers.append(swapped)
        else:
            utils.bail(f"unsupported call: {node.op}")

    return layers

def swap(module: nn.Module, dtype) -> Optional[Layer]:
    if isinstance(module, nn.Conv2d):
        return Conv2D(dtype, module.weight, module.bias, module.stride,
                      module.padding, module.dilation, module.padding_mode,
                      module.groups)
    elif isinstance(module, nn.Linear):
        return Linear(dtype, module.weight, module.bias)
    elif isinstance(module, nn.MaxPool2d):
        return MaxPool2D(dtype, module.kernel_size, module.stride,
                         module.padding, module.dilation, module.ceil_mode)
    elif isinstance(module, nn.AvgPool2d):
        return AvgPool2D(dtype, module.kernel_size, module.stride,
                         module.padding, module.ceil_mode,
                         module.count_include_pad,
                         module.divisor_override)
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return AdaptiveAvgPool2D(dtype, module.output_size)
    elif isinstance(module, nn.ReLU):
        return ReLU(dtype)
    elif isinstance(module, nn.Flatten):
        return Flatten(dtype, start_dim=module.start_dim, end_dim=module.end_dim)
    return None

class Conv2D(Layer):
    def __init__(self, dtype, weight, bias,
                 stride: Union[int, Sequence[int]],
                 padding: Union[str, Union[int, Sequence[int]]],
                 dilation: Union[int, Sequence[int]], padding_mode: str, groups: int):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        padding = utils.make_padding_2d(
            padding, stride, dilation, weight.size())
        assert(padding_mode in ['zeros', 'reflect', 'replicate', 'circular'])
        pad_mode = {
            'zeros': core.PaddingMode.zeros,
            'reflect': core.PaddingMode.reflect,
            'replicate': core.PaddingMode.replicate,
            'circular': core.PaddingMode.circular
        }[padding_mode]

        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = utils.get_correct_from_type(dtype, core.Conv2D_float,
                                                core.Conv2D_double)(weight_addr,
                                                                    weight_shape,
                                                                    bias_addr,
                                                                    padding,
                                                                    stride,
                                                                    dilation,
                                                                    pad_mode,
                                                                    groups)

    def forward(self, input) -> Union[core.Tensor_float, core.Tensor_double]:
        return self.core.forward(utils.get_address(input), utils.get_shape(input))


class Linear(Layer):
    def __init__(self, dtype, weight, bias):
        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = utils.get_correct_from_type(
            dtype, core.Linear_float, core.Linear_double)(weight_addr, weight_shape, bias_addr)


class ReLU(Layer):
    def __init__(self, dtype):
        self.core = utils.get_correct_from_type(
            dtype, core.ReLU_float, core.ReLU_double)()


class MaxPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]],
                 dilation: Union[int, Sequence[int]],
                 ceil_mode: bool):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)
        ceil_mode = ceil_mode

        self.core = utils.get_correct_from_type(dtype, core.MaxPool2D_float, core.MaxPool2D_double)(kernel_shape,
                                                                                                    padding, stride, dilation, ceil_mode)


class AvgPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]],
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int]):
        stride = utils.make_2d(stride)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)

        self.core = utils.get_correct_from_type(dtype, core.AvgPool2D_float, core.AvgPool2D_double)(kernel_shape,
                                                                                                    padding, stride, ceil_mode, count_include_pad, divisor_override)


class AdaptiveAvgPool2D(Layer):
    def __init__(self, dtype, output_shape: Optional[Union[int, Sequence[Optional[int]]]]):
        if isinstance(output_shape, list):
            assert (len(output_shape) == 2)
        if isinstance(output_shape, int):
            output_shape = utils.make_2d(output_shape)
        self.core = utils.get_correct_from_type(
            dtype, core.AdaptiveAvgPool2D_float, core.AdaptiveAvgPool2D_double)(output_shape)


class Flatten(Layer):
    def __init__(self, dtype, start_dim: int, end_dim: int):
        self.core = utils.get_correct_from_type(
            dtype, core.Flatten_float, core.Flatten_double)(start_dim, end_dim)
