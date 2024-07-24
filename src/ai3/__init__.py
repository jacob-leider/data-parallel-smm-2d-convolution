# * CPP *
# TODO groups and padding modes for convolution implementations
# TODO adaptiveavgpool2d where output dim isn't multiple of input dim

# * PUBLISHING *
# TODO after PyTorch 2.4 release 7-24-24 add that version to pyproject
# TODO documentation for both the framework and CPP library
# TODO add homepage information, docs, tests to pyproject.toml

# * FUTURE *
# TODO custom backpropagation algorithms
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
        model = utils.get_item(
            dtype, core.Model_float, core.Model_double)
        self.core = model(cores)

    def __call__(self, input):
        return self.predict(input, type(input))

    def predict(self, input, out_type=None):
        out = self.core.predict(utils.get_address(
            input), utils.get_shape(input))
        out = Tensor(out)
        return out.to(out_type)


def swap_backend(module: torch.nn.Module,
                 algos: Optional[Mapping[str, Union[str, Sequence[str],
                                                    Callable]]] = None,
                 sample_input_shape: Optional[Sequence[int]] = None, *,
                 dtype=None) -> Model:
    if algos:
        algos = {**DEFAULT_ALGOS, **algos}
    else:
        algos = DEFAULT_ALGOS
    if not dtype:
        dtype = torch.get_default_dtype()
    utils.check_callable_params_with_shape(
        algos, sample_input_shape)
    return Model(dtype, swap_torch.swap_backend_layers(
        module, dtype, algos, sample_input_shape))


def swap_conv2d(
        module: torch.nn.Module,
        algos: Optional[Union[str, Sequence[str],
                              Callable]] = None,
        sample_input_shape: Optional[Sequence[int]] = None):
    if not algos:
        algos = DEFAULT_ALGOS["conv2d"]
    utils.check_callable_params_with_shape(
        {'conv2d': algos}, sample_input_shape)
    swap_torch.swap_conv2d(
        module, algos, sample_input_shape)
