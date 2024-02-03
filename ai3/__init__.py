import torch.nn as nn
import torch
import ai3_functions


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
        return ai3_functions.linear(x, self.weights, self.bias)


def optimize(model) -> nn.Module:
    assert (isinstance(model, nn.Module))

    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            setattr(model, name, Linear(layer.in_features, layer.out_features))

    return model
