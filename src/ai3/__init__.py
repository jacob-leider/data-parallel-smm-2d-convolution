"""Provides the easy-to-use fine-grain algorithmic control over a `PyTorch <https://pytorch.org/>`_ *DNN*

The framework currently features two methods for algorithmic swapping. :func:`swap_backend`
which swaps every module type of a *DNN* returning an object completely managed
by the framework and :func:`swap_conv2d` which swaps convolution operations out of the
existing *DNN*.
"""

import torch
from typing import Mapping, Optional, Sequence, TypeAlias
from ai3 import core, utils, layers, swap_torch
from ai3.tensor import Tensor

AlgorithmicSelector: TypeAlias = utils.AlgorithmicSelector
"""The object that performs the algorithmic selection for an associated operation.

There are three different types of algorithmic selectors.

* **str:** contains the name of the algorithm that will be used for all modules
    of the associated type
* **Sequence[str]:** list of algorithm names, as modules are encountered in a forward pass,
    they are replaced with an implementation of the algorithm in the list
    with the same index as that module has relative to the other modules
    of the associated type
* **Callable:** function which is given the module being swapped and returns the name of the
    algorithm to use, the function can optionally take the input size for this
    module provided that a sample input shape was passed to the swapping function.

Example:
    Function to perform algorithmic selection.

    >>> def conv2d_selector(orig: torch.nn.Conv2d, input_shape) -> str:
    ...     out_channels = orig.weight.shape[0]
    ...     if (out_channels < 50 and
    ...         input_shape[0] < 50 and
    ...         input_shape[1] > 150 and
    ...         input_shape[2] > 150):
    ...         return 'direct'
    ...     return 'smm'
    ...
    >>> input_data = torch.randn(10, 3, 224, 224)
    >>> orig = ConvNet()
    >>> torch_out = orig(input_data)
    >>> ai3.swap_conv2d(orig, conv2d_selector, (3, 224, 224))
    >>> sc_out = orig(input_data)
    >>> torch.allclose(torch_out, sc_out, atol=1e-6)
    True
"""


DEFAULT_ALGOS: Mapping[str, str] = {key: "default" for key in [
    "conv2d", "linear", "relu", "maxpool2d", "avgpool2d", "adaptiveavgpool2d", "flatten"]}

SUPPORTED_ALGORITHMS = utils.SUPPORTED_ALGORITHMS
"""The supported operations and their supported algorithms.

See :ref:`supported-operations` for supported acceleration platform by algorithm.
"""


class Model():
    """The model which performs the operations using the user specified
    algorithms."""

    def __init__(self, dtype, layers: Sequence[layers.Layer]):
        cores = [layer.core for layer in layers]
        model = utils.get_item(
            dtype, core.Model_float, core.Model_double)
        self.core = model(cores)

    def __call__(self, input):
        """Perform a prediction on the input data."""
        return self.predict(input, type(input))

    def predict(self, input, out_type=None):
        """Returns the output after passing input through the layers.

        Args:
            input : input tensor to perform operations on
            out_type : the desired type of the output tensor
        Returns:
            output after performing operations
        """
        out = self.core.predict(utils.get_address(
            input), utils.get_shape(input))
        out = Tensor(out)
        return out.to(out_type)


def swap_conv2d(
        module: torch.nn.Module,
        algos: Optional[AlgorithmicSelector] = None,
        sample_input_shape: Optional[Sequence[int]] = None):
    """
    Swaps, in-place, conv2d operations out of the existing *DNN* for an implementation of
    the user specified algorithm. After swapping, the same *DNN* can still be trained
    and compiled. If no :type:`AlgorithmicSelector` is given then the default
    algorithm decided by the framework are used.

    Args:
        module : the module containing the conv2d operations to be swapped out in-place
        algos (Optional[:type:`AlgorithmicSelector`]) :
            algorithmic selector for the *conv2d* operations
        sample_input_shape : the input shape to the *DNN* which is passed to the
                             function algorithmic selector if present

    Example:
        Swaps the first *conv2d* operation for an implementation of direct convolution
        and the second *conv2d* operation for an implementation of *SMM* convolution

        >>> input_data = torch.randn(10, 3, 224, 224)
        >>> orig = ConvNet()
        >>> orig_out = orig(input_data)
        >>> ai3.swap_conv2d(orig, ['direct', 'smm'])
        >>> sc_out = orig(input_data)
        >>> torch.allclose(orig_out, sc_out, atol=1e-6)
        True
    """
    if not algos:
        algos = DEFAULT_ALGOS["conv2d"]
    utils.check_callable_params_with_shape(
        {'conv2d': algos}, sample_input_shape)
    swap_torch.swap_conv2d(
        module, algos, sample_input_shape)


def swap_backend(module: torch.nn.Module,
                 algos: Optional[Mapping[str, AlgorithmicSelector]] = None,
                 sample_input_shape: Optional[Sequence[int]] = None, *,
                 dtype=None) -> Model:
    """
    Swaps every module in an exsiting *DNN* for an implementation
    of the user specified algorithm returning
    a :class:`Model` completly managed by the framework.

    Algorithmic selection is performed by passing a mapping from strings
    containing names of the operations to swap to a :type:`AlgorithmicSelector`.
    If no :type:`AlgorithmicSelector` is passed for a given operation then the default
    algorithm decided by the framework are used.

    Args:
        module : the module to have its operations replaced
        algos (Optional[Mapping[str, AlgorithmicSelector]]) :
            mapping from operation type to a :type:`AlgorithmicSelector`
        sample_input_shape : the input shape to the *DNN* which is passed to the
                             function algorithmic selector if present
    Returns:
        A :class:`Model` which uses the algorithms specified and
        yields equivalent outputs as the original *DNN*

    Example:
    Swaps the first *conv2d* operation for an implementation of direct convolution
    and the second *conv2d* operation for an implementation of *SMM* convolution

        >>> def auto_selector(orig: torch.nn.Conv2d, input_shape) -> str:
        ...     out_channels = orig.weight.shape[0]
        ...     if (out_channels < 50 and
        ...         input_shape[1] < 50 and
        ...         input_shape[2] > 150 and
        ...         input_shape[3] > 150):
        ...         return 'direct'
        ...     return 'smm'
        ...
        >>> input_data = torch.randn(1, 3, 224, 224)
        >>> vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        >>> vgg16 = vgg16.eval()
        >>> with torch.inference_mode():
        ...     torch_out = vgg16(input_data)
        ...     model: ai3.Model = ai3.swap_backend(vgg16, {"conv2d": auto_selector,
        ...                                                 "maxpool2d": "default"},
        ...                                         sample_input_shape=(1, 3, 224, 224))
        ...     sb_out = model(input_data)
        ...     torch.allclose(torch_out, sb_out, atol=1e-4)
        True
    """

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


