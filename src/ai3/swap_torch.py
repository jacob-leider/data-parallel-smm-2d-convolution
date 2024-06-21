import ai3
from ai3 import layers, utils
from typing import Optional, List
import torch
from torch import nn, fx

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

def swap_conv2d(module: nn.Module, dtype, algo: str) -> nn.Module:
    gm: fx.GraphModule = fx.symbolic_trace(module)
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            mod = getmodule(module, node.target)
            if isinstance(mod, nn.Conv2d):
                swapped = swap_layer(mod, dtype, {'conv2d': algo})
                assert(isinstance(swapped, layers.Conv2D))
                module = setmodule(module, node.target, Conv2D(swapped))
    return module


def get_layers(module: nn.Module, dtype, algos: dict[str,str]) -> List[layers.Layer]:
    gm: fx.GraphModule = fx.symbolic_trace(module)
    forwards = []
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
                forwards.append(layers.Flatten(dtype, start_dim, end_dim, algos['flatten']))
            elif node.target == torch.relu:
                forwards.append(layers.ReLU(dtype, algos['relu']))
            else:
                utils.bail(f"unsupported function: {node.target}")
        elif node.op == 'call_module':
            mod = getmodule(module, node.target)
            if not isinstance(mod, nn.Dropout):
                swapped = swap_layer(mod, dtype, algos)
                if not swapped:
                    utils.bail(f"unsupported module: {mod}")
                forwards.append(swapped)
        else:
            utils.bail(f"unsupported call: {node.op}")

    return forwards


# TODO add a dict here for algorithms to use at specific layers
# implement it for all the other layers
def swap_layer(module: nn.Module, dtype, algos: dict[str,str]) -> Optional[layers.Layer]:
    if isinstance(module, nn.Conv2d):
        return layers.Conv2D(dtype, module.weight, module.bias, module.stride,
                      module.padding, module.dilation, module.padding_mode,
                      module.groups, algos["conv2d"])
    elif isinstance(module, nn.Linear):
        return layers.Linear(dtype, module.weight, module.bias, algos['linear'])
    elif isinstance(module, nn.MaxPool2d):
        return layers.MaxPool2D(dtype, module.kernel_size, module.stride,
                         module.padding, module.dilation, module.ceil_mode, algos['maxpool2d'])
    elif isinstance(module, nn.AvgPool2d):
        return layers.AvgPool2D(dtype, module.kernel_size, module.stride,
                         module.padding, module.ceil_mode,
                         module.count_include_pad,
                         module.divisor_override, algos['avgpool2d'])
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return layers.AdaptiveAvgPool2D(dtype, module.output_size, algos['adaptiveavgpool2d'])
    elif isinstance(module, nn.ReLU):
        return layers.ReLU(dtype, algos['relu'])
    elif isinstance(module, nn.Flatten):
        return layers.Flatten(dtype, module.start_dim, module.end_dim, algos['flatten'])
    return None

