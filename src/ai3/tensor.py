from ai3 import _core, utils, errors
from typing import Union


class Tensor():
    """Simple type which implements the
    `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_"""

    def __init__(self, tens: Union[_core.Tensor_float, _core.Tensor_double]):
        self.core = tens
        if isinstance(tens, _core.Tensor_float):
            self.typestr = utils.FLOAT32_STR
        else:
            self.typestr = utils.FLOAT64_STR

    def to(self, out_type):
        """
        Transform this Tensor to another type using
        the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.

        Args:
            out_type : type to transform this object to

        Returns:
            An object of *out_type* with the shape and contents of the original
        """
        if out_type is None:
            return self
        if not isinstance(out_type, str):
            out_type = f"{out_type.__module__}.{out_type.__name__}"

        if out_type == "torch.Tensor":
            return self.torch()
        elif out_type == "numpy.ndarray":
            return self.numpy()
        errors.bail(
            f"unsupported type to transfer tensor to {out_type}")

    def numpy(self):
        """
        Transform this Tensor to a *numpy.ndarray* using
        the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.

        Returns:
            *numpy.ndarray* with the shape and contents of the original
        """
        import numpy
        dtype = {
            utils.FLOAT32_STR: numpy.float32,
            utils.FLOAT64_STR: numpy.float64
        }[self.typestr]
        errors.bail_if(dtype is None,
                       f"type, {self.typestr} is neither float32 or float64")
        data: numpy.ndarray = numpy.frombuffer(
            self.core, dtype=dtype)
        return data.reshape(self.core.shape)

    def torch(self):
        """
        Transform this Tensor to a `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_ using
        the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.


        Returns:
            `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_ with the shape and contents of the original
        """
        import torch
        return torch.frombuffer(
            self.core, dtype=torch.__dict__[self.typestr]).view(
            self.core.shape)
