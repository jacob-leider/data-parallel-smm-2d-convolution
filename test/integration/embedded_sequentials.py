import torch
from torch import nn
import ai3
from test import compare_tensors


class EmbeddedSequentials(torch.nn.Module):
    def __init__(self):
        super(EmbeddedSequentials, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
                )))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
                )))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def _run(data, mes):
    model = EmbeddedSequentials()
    ai3_model = ai3.swap_backend(model)
    target = model(data)
    output = ai3_model.predict(data)
    compare_tensors(output, target.detach().numpy(), mes)


def run():
    print("SIMPLE CREATED CONV NET Embedded Sequentials")
    _run(torch.randn(3, 224, 224), "simple conv2d")
    _run(torch.randn(5, 3, 224, 224), "simple conv2d multi sample")


if __name__ == "__main__":
    run()
