from ai3 import layers, utils
import torch
from torch import nn


class Conv2D(nn.Module):
    def __init__(self, internal: layers.Conv2D):
        super(Conv2D, self).__init__()
        self.internal = internal

    def forward(self, x: torch.Tensor):
        out = self.internal.forward(x)
        return utils.tensor_to_type(out, torch.Tensor)
