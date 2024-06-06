import torch
from torch import nn
import ai3
from tests import compare_tensors

class SimpleConvNet(torch.nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def run():
    print("SIMPLE CREATED CONV NET")
    model = SimpleConvNet()

    input_data = torch.randn(3, 224, 224)
    target = model(input_data)
    ai3_model = ai3.optimize(model)
    output = ai3_model.predict(input_data)
    compare_tensors(output, target.detach().numpy(), "simple conv2d")
    input_multi_sample = torch.randn(5, 3, 224, 224)
    target = model(input_multi_sample)
    output = ai3_model.predict(input_multi_sample)
    compare_tensors(output, target.detach().numpy(), "simple conv2d multi sample")


if __name__ == "__main__":
    run()
