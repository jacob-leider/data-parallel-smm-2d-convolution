from ai3 import core
from ai3 import utils
from ai3.utils import get_dtype_str
from typing import (
    Union,
    Sequence
)

# TODO organize these bad datatype errors maybe create a function that
# takes a datatype gets the string and then picks from the correct one
# see where get_dtype_str is called and the similarities, will also need to do generics
# for all functions

class Model():
    def __init__(self, dtype, layers):
        dtype = get_dtype_str(dtype)
        cores = [layer.core for layer in layers]
        if dtype == 'float':
            self.core = core.Model_float(cores)
        elif dtype == 'double':
            self.core = core.Model_double(cores)
        else:
            assert False and 'bad input type'

    def predict(self, input):
        return self.core.predict(utils.get_address(input), utils.get_shape(input))

class Conv2D():
    def __init__(self, dtype, weight, bias, *,
                 stride: Union[int, Sequence[int]],
                 padding: Union[str, Union[int, Sequence[int]]],
                 dilation: Union[int, Sequence[int]]):
        dtype = get_dtype_str(dtype)
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        padding = utils.make_padding_2d(padding, stride, dilation, weight.size())

        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        if dtype == "float":
            self.core = core.Conv2D_float(weight_addr, weight_shape, bias_addr,
                                     padding, stride, dilation)
        elif dtype == "double":
            self.core = core.Conv2D_double(weight_addr, weight_shape, bias_addr,
                                     padding, stride, dilation)

