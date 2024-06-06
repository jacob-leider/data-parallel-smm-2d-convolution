import torch
from bench import predict_show_time
from torch import nn
import ai3
from tests import compare_tensors

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x

def run():
    print("Conv2D")
    input = torch.randn(100, 3, 300, 300)
    orig = Conv2D(3, 4, (5,5))
    optim = ai3.optimize(orig)
    orig_out = predict_show_time(orig, input, "pytorch")
    assert(isinstance(orig_out, torch.Tensor))
    optim_out = predict_show_time(optim, input, "ai3")
    compare_tensors(optim_out, orig_out.detach().numpy())

if __name__ == "__main__":
    run()
