import torch
from torch import nn
import ai3
import time


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def run():
    print('SIMPLE CREATED')
    input_data = torch.randn(100, 3, 224, 224)
    torch_model = SimpleConvNet()
    tstart = time.time()
    torch_out = torch_model(input_data)
    tend = time.time()
    assert (isinstance(torch_out, torch.Tensor))
    ai3_model = ai3.swap_backend(torch_model, {"conv2d": "default"})
    astart = time.time()
    ai3_out = ai3_model(input_data)
    aend = time.time()
    print(f"Time torch: {tend-tstart}")
    print(f"Time ai3: {aend-astart}")
    assert (isinstance(ai3_out, torch.Tensor))
    assert torch.allclose(torch_out, ai3_out, atol=1e-4)


if __name__ == "__main__":
    run()
