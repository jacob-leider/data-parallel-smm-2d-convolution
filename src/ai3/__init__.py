import torch
from torch import nn
from typing import Optional, Sequence
from ai3 import layers
from ai3 import swap_torch
from ai3 import utils, core

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


class Model():
    def __init__(self, dtype, layers: Sequence[layers.Layer]):
        cores = [layer.core for layer in layers]
        self.core = utils.get_correct_from_type(
            dtype, core.Model_float, core.Model_double)(cores)

    def predict(self, input, out_type=None):
        out = self.core.predict(utils.get_address(
            input), utils.get_shape(input))
        out = utils.tensor_to_type(out, out_type)
        return out

# TODO function: optimize(objective='memory'/'energy'/'latency',
#                         guess=[users guess for best algorithms to use for each layer],
#                         given=[each layers algorithm will be set to the algorithm in this list] # only one of given or guess can be used at a time
#                         model=model to optimize layers for) -> New Model With Better Layers


def swap_backend(module: nn.Module) -> Model:
    dtype = torch.get_default_dtype()
    return Model(dtype, swap_torch.get_layers(module, dtype))


def swap_conv2d(module: nn.Module):
    swap_torch.swap_conv2d(module, torch.get_default_dtype())
