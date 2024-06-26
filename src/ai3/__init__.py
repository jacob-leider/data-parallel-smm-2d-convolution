# TODO try some way to do the normal pip install without compiling cpp code
# then taking users cpp code and compiling the SO with that. This would stop
# requiring building from source, this would be easier to do with CMAKE
# see if we can include <ai3.hpp> and if that gets handled in cmake
# TODO onnx support, should be pretty easy to also iterate
# through the onnx layers and hyperparametrs, could also use the pytorch way
# of loading .onnx, .onnx -> nn.Module -> ai3.optimize -> ai3.Model
# once onnx support we can have two files which are the only places torch and onnx are imported

import torch
from torch import nn
from typing import Optional, Sequence, Union
from ai3 import layers, swap_torch, utils, core

DEFAULT_ALGOS = {key: "default" for key in [
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


class Tensor():
    def __init__(self, tens: Union[core.Tensor_float, core.Tensor_double]):
        self.core = tens
        if isinstance(tens, core.Tensor_float):
            self.typestr = utils.FLOAT32_STR
        else:
            self.typestr = utils.FLOAT64_STR

    def to(self, out_type):
        if out_type is torch.Tensor:
            return self.torch()
        elif out_type is None:
            return self
        utils.bail(f"unsupported type to transfer tensor to {out_type}")

    def numpy(self):
        import numpy
        dtype = {
            utils.FLOAT32_STR: numpy.float32,
            utils.FLOAT64_STR: numpy.float64
        }[self.typestr]
        utils.bail_if(dtype is None,
                      f"type, {self.typestr} is neither float32 or float64")
        data = numpy.frombuffer(self.core, dtype=dtype)
        return data.reshape(self.core.shape)

    def torch(self):
        return torch.frombuffer(self.core,
                                dtype=torch.__dict__[self.typestr]).view(self.core.shape)


def swap_backend(module: nn.Module, algos: Optional[dict[str, str]] = None) -> Model:
    if algos:
        algos = DEFAULT_ALGOS | algos
    else:
        algos = DEFAULT_ALGOS
    dtype = torch.get_default_dtype()
    return Model(dtype, swap_torch.get_layers(module, dtype, algos))


def swap_conv2d(module: nn.Module, algo: Optional[str] = None):
    if not algo:
        algo = DEFAULT_ALGOS["conv2d"]
    swap_torch.swap_conv2d(module, torch.get_default_dtype(), algo)
