import ai3
from ai3 import layers, utils
from typing import Mapping, Optional, List, Sequence, Union, Callable
from collections import defaultdict
import torch
from torch import nn, fx

def mod_to_op(mod: nn.Module) -> str:
    if isinstance(mod, nn.Conv2d):
        return "conv2d"
    elif isinstance(mod, nn.Linear):
        return "linear"
    elif isinstance(mod, nn.MaxPool2d):
        return "maxpool2d"
    elif isinstance(mod, nn.AvgPool2d):
        return "avgpool2d"
    elif isinstance(mod, nn.AdaptiveAvgPool2d):
        return 'adaptiveavgpool2d'
    elif isinstance(mod, nn.ReLU):
        return 'relu'
    elif isinstance(mod, nn.Flatten):
        return 'flatten'
    utils.bail(f"unsupported module {mod}")


def iscontainer(name: str) -> bool:
    return '.' in name


def getmodule(module: nn.Module, name) -> nn.Module:
    if iscontainer(name):
        names = name.split('.', 1)
        return getmodule(getattr(module, names[0]), names[1])
    else:
        return getattr(module, name)


def setmodule(module: nn.Module, name, new: nn. Module) -> nn.Module:
    if iscontainer(name):
        names = name.split('.', 1)
        setmodule(getattr(module, names[0]), names[1], new)
    else:
        setattr(module, name, new)
    return module


class Conv2D(nn.Module):
    def __init__(self, internal: layers.Conv2D):
        super(Conv2D, self).__init__()
        self.internal = internal

    def forward(self, x: torch.Tensor):
        out = ai3.Tensor(self.internal.forward(x))
        return out.to(torch.Tensor)


def swap_conv2d(module: nn.Module, dtype, algo: Union[str, Sequence[str], Callable]) -> nn.Module:
    gm: fx.GraphModule = fx.symbolic_trace(module)
    layer_counters = defaultdict(int)
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            mod = getmodule(module, node.target)
            if isinstance(mod, nn.Conv2d):
                swapped = swap_layer(
                    mod, dtype, {'conv2d': algo}, layer_counters)
                assert (isinstance(swapped, layers.Conv2D))
                module = setmodule(module, node.target, Conv2D(swapped))
    return module


def get_algo_inc_counter(orig: nn.Module,  algos: Mapping[str, Union[str, Sequence[str], Callable]], layer_counters: defaultdict[str, int]) -> str:
    op = mod_to_op(orig)
    if callable(algos[op]):
        algo = algos[op](orig)
    elif isinstance(algos[op], list):
        algo = algos[op][layer_counters[op]]
        layer_counters[op] += 1
    else:
        algo = algos[op]
    assert isinstance(algo, str)
    return algo


def get_layers(module: nn.Module, dtype, algos: Mapping[str, Union[str, Sequence[str], Callable]]) -> List[layers.Layer]:
    gm: fx.GraphModule = fx.symbolic_trace(module)
    forwards = []

    layer_counters = defaultdict(int)

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
                forwards.append(layers.Flatten(
                    dtype, start_dim, end_dim,
                    get_algo_inc_counter(torch.nn.Flatten(start_dim, end_dim), algos, layer_counters)))
            elif node.target == torch.relu:
                forwards.append(layers.ReLU(
                    dtype, get_algo_inc_counter(torch.nn.ReLU(), algos, layer_counters)))
            else:
                utils.bail(f"unsupported function: {node.target}")
        elif node.op == 'call_module':
            mod = getmodule(module, node.target)
            if not isinstance(mod, nn.Dropout):
                swapped = swap_layer(mod, dtype, algos, layer_counters)
                if not swapped:
                    utils.bail(f"unsupported module: {mod}")
                forwards.append(swapped)
        else:
            utils.bail(f"unsupported call: {node.op}")

    return forwards


def swap_layer(module: nn.Module, dtype, algos: Mapping[str, Union[str, Sequence[str], Callable]],
               layer_counters: defaultdict[str, int]) -> Optional[layers.Layer]:
    if isinstance(module, nn.Conv2d):
        return layers.Conv2D(dtype, module.weight, module.bias, module.stride,
                             module.padding, module.dilation, module.padding_mode,
                             module.groups, get_algo_inc_counter(module, algos, layer_counters))
    elif isinstance(module, nn.Linear):
        return layers.Linear(dtype, module.weight, module.bias, get_algo_inc_counter(module, algos, layer_counters))
    elif isinstance(module, nn.MaxPool2d):
        return layers.MaxPool2D(dtype, module.kernel_size, module.stride,
                                module.padding, module.dilation, module.ceil_mode, get_algo_inc_counter(module, algos, layer_counters))
    elif isinstance(module, nn.AvgPool2d):
        return layers.AvgPool2D(dtype, module.kernel_size, module.stride,
                                module.padding, module.ceil_mode,
                                module.count_include_pad,
                                module.divisor_override, get_algo_inc_counter(module, algos, layer_counters))
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return layers.AdaptiveAvgPool2D(dtype, module.output_size, get_algo_inc_counter(module, algos, layer_counters))
    elif isinstance(module, nn.ReLU):
        return layers.ReLU(dtype, get_algo_inc_counter(module, algos, layer_counters))
    elif isinstance(module, nn.Flatten):
        return layers.Flatten(dtype, module.start_dim, module.end_dim, get_algo_inc_counter(module, algos, layer_counters))
    return None
