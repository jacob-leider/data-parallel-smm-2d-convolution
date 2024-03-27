# TODO for backprop: https://pytorch.org/tutorials/advanced/cpp_extension.html
# TODO benchmarks
from torch import nn
from ai3 import layers
from typing import Optional, Sequence

KN2ROW = 'kn2row'
TORCH = 'torch'
SUPPORTED_OBJECTIVES = ['energy', 'latency', 'memory']
SUPPORTED_ALGORITHMS = [KN2ROW, TORCH]

# TODO support for guess also being a dict mapping layer type to algorithm
# TODO some check to make sure objective is either latency, memory, energy


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


def optimize(module: nn.Module) -> nn.Module:
    for name, mod in module.named_children():
        if isinstance(mod, nn.Sequential):
            setattr(module, name, optimize(mod))
        if isinstance(mod, nn.Conv2d):
            # make sure non-defaults are all supported, remove this as they get supported
            assert (mod.output_padding == 0 or mod.output_padding == (
                0, 0)) and mod.padding_mode == 'zeros' and not mod.transposed

            ai3_layer = layers.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                                      bias=mod.bias is not None, stride=mod.stride,
                                      padding=mod.padding,
                                      dilation=mod.dilation)
            assert ai3_layer.weight.shape == mod.weight.shape
            ai3_layer.weight = mod.weight
            if mod.bias is not None and ai3_layer.bias is not None:
                assert ai3_layer.bias.shape == mod.bias.shape
                ai3_layer.bias = mod.bias
            setattr(module, name, ai3_layer)

    return module
