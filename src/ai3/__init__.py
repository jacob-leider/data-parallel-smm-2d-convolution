# TODO benchmarks should benchmark each layer to start
# right now limitation is that the model to optimize must be written
# with each part of its forwarding declared in order as a field of
# the model class

from torch import nn
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

def get_layers(module: nn.Module, layers, dtype) -> List:
    for _, mod in module.named_children():
        if isinstance(mod, nn.Sequential):
            layers = get_layers(mod, layers, dtype)
        elif isinstance(mod, nn.Conv2d):
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
        elif isinstance(mod, nn.Dropout):
            pass
        else:
            assert False, f"unsupported module: f{mod}"

    return layers


def optimize(module: nn.Module) -> model.Model:
    dtype = None
    for param in module.parameters():
        dtype = param.dtype
        break
    return model.Model(dtype, get_layers(module, [], dtype))
