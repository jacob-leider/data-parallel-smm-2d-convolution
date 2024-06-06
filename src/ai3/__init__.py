# TODO benchmarks should benchmark each layer to start
# right now limitation is that the model to optimize must be written
# with each part of its forwarding declared in order as a field of
# the model class

import torch
from torch import nn, fx
from typing import Optional, Sequence, List
from ai3 import model

KN2ROW = 'kn2row'
TORCH = 'torch'
SUPPORTED_OBJECTIVES = ['energy', 'latency', 'memory']
SUPPORTED_ALGORITHMS = [KN2ROW, TORCH]


# TODO support for guess also being a dict mapping layer type to algorithm
def module_algorithms(holder: nn.Module, objective: str = 'latency', guess: Optional[Sequence[str]] = None) -> Sequence[str]:
    assert objective in SUPPORTED_OBJECTIVES
    assert guess is None or len(guess) == len(list(holder.children()))

    algos = []
    for i, mod in enumerate(holder.children()):
        if isinstance(mod, nn.Sequential):
            if guess is not None:
                assert isinstance(guess[i], list)
                algos.append(module_algorithms(mod, objective, guess[i]))
            algos.append(module_algorithms(mod, objective, None))
        else:
            if guess is not None:
                # TODO some processing on whether to use the guess
                assert guess[i] in SUPPORTED_ALGORITHMS
            if isinstance(mod, nn.Conv2d):
                algos.append(KN2ROW)
            else:
                algos.append(TORCH)
    return algos


# TODO function: optimize(objective='memory'/'energy'/'latency',
#                         guess=[users guess for best algorithms to use for each layer],
#                         given=[each layers algorithm will be set to the algorithm in this list] # only one of given or guess can be used at a time
#                         model=model to optimize layers for) -> New Model With Better Layers

def issequential(name: str) -> bool:
    return '.' in name

def getmodule(module: nn.Module, name:str) -> nn.Module:
    if issequential(name):
        names = name.split('.', 1)
        return getmodule(getattr(module, names[0]), names[1])
    else:
        return getattr(module, name)

def get_layers(module: nn.Module, dtype) -> List:
    gm: fx.GraphModule = fx.symbolic_trace(module)
    layers = []
    import torch
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
                layers.append(model.Flatten(dtype, start_dim, end_dim))
            elif node.target == torch.relu:
                layers.append(model.ReLU(dtype))
        elif node.op == 'call_module':
            mod = getmodule(module, node.target)
            if isinstance(mod, nn.Conv2d):
                assert (mod.padding_mode == 'zeros')
                layers.append(model.Conv2D(dtype, mod.weight,
                                           mod.bias, mod.stride,
                                           mod.padding, mod.dilation))
            elif isinstance(mod, nn.Linear):
                layers.append(model.Linear(dtype, mod.weight, mod.bias))
            elif isinstance(mod, nn.MaxPool2d):
                layers.append(model.MaxPool2D(dtype, mod.kernel_size, mod.stride,
                                              mod.padding, mod.dilation, mod.ceil_mode))
            elif isinstance(mod, nn.AvgPool2d):
                layers.append(model.AvgPool2D(dtype, mod.kernel_size, mod.stride,
                                              mod.padding, mod.ceil_mode,
                                              mod.count_include_pad,
                                              mod.divisor_override))
            elif isinstance(mod, nn.AdaptiveAvgPool2d):
                layers.append(model.AdaptiveAvgPool2D(dtype, mod.output_size))
            elif isinstance(mod, nn.ReLU):
                layers.append(model.ReLU(dtype))
            elif isinstance(mod, nn.Flatten):
                layers.append(model.Flatten(dtype, start_dim=mod.start_dim, end_dim=mod.end_dim))
            elif isinstance(mod, nn.Dropout):
                pass
            else:
                assert False, f"unsupported module: {mod}"
        else:
            assert False, f"unsupported call: {node.op}"

    return layers

def optimize(module: nn.Module) -> model.Model:
    dtype = torch.get_default_dtype()
    return model.Model(dtype, get_layers(module, dtype))
