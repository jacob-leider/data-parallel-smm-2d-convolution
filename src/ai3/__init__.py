# * Python *
# TODO for easier swap_module https://github.com/pytorch/examples/blob/main/fx/replace_op.py
# TODO can actually pass the function to the swap_torch.Conv2D which can then call it with the input size known

# * CPP *
# TODO organize the CPP files by operation
# TODO probably better to make the vector for stride and dilation, just be the two ints
# TODO groups and padding modes for convolution implementations
# TODO adaptiveavgpool2d where output dim isn't multiple of input dim


# * FUTURE *
# TODO onnx format support
# TODO tensorflow format support


import torch
from typing import Mapping, Optional, Sequence, Union, Callable
from ai3 import core, utils, layers, swap_torch
from ai3.tensor import Tensor

DEFAULT_ALGOS: Mapping[str, str] = {key: "default" for key in [
    "conv2d", "linear", "relu", "maxpool2d", "avgpool2d", "adaptiveavgpool2d", "flatten"]}


class Model():
    def __init__(self, dtype, layers: Sequence[layers.Layer]):
        cores = [layer.core for layer in layers]
        (model, self.dtype) = utils.get_item_and_type(
            dtype, core.Model_float, core.Model_double)
        self.core = model(cores)

    def __call__(self, input):
        return self.predict(input, type(input))

    def predict(self, input, out_type=None):
        out = self.core.predict(utils.get_address(
            input), utils.get_shape(input))
        out = Tensor(out)
        return out.to(out_type)


def swap_backend(module: torch.nn.Module, algos: Optional[Mapping[str, Union[str, Sequence[str], Callable]]] = None, dtype=None) -> Model:
    if algos:
        algos = {**DEFAULT_ALGOS, **algos}
    else:
        algos = DEFAULT_ALGOS
    if not dtype:
        dtype = torch.get_default_dtype()
    return Model(dtype, swap_torch.get_swapped_backend_layers(module, dtype, algos))


def swap_conv2d(module: torch.nn.Module, algos: Optional[Union[str, Sequence[str], Callable]] = None):
    if not algos:
        algos = DEFAULT_ALGOS["conv2d"]
    swap_torch.swap_conv2d(module, torch.get_default_dtype(), algos)
