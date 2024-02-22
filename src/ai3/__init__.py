# TODO should setup so we allocate the outputs in Python and give it as an attribute
# of the class, need to make sure the attribute doesn't already exists or just call it
# _ai3_layer_i_output or something
# then use the .storage() thing that Nat said in slack
import torch
from torch import nn
from ai3 import functions


class Module(nn.Module):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return functions.linear(x, self.weights, self.bias)


def optimize(model) -> nn.Module:
    assert (isinstance(model, nn.Module))

    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            setattr(model, name, Linear(layer.in_features, layer.out_features))

    return model
