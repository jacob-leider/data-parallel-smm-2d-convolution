import torch
from bench import predict_show_time
from torch import nn
import ai3
from tests import compare_tensors


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return x


def run():
    print("Flatten")
    input = torch.randn(10, 100, 20, 30, 40, 50)
    orig = Flatten()
    optim = ai3.swap_backend(orig)
    orig_out = predict_show_time(orig, input, "pytorch")
    assert (isinstance(orig_out, torch.Tensor))
    optim_out = predict_show_time(optim, input, "ai3")
    # TODO remove this
    # compare_tensors(optim_out, orig_out.detach().numpy(), "")

if __name__ == "__main__":
    run()