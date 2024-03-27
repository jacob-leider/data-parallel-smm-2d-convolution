import torch
import ai3
from tests import compare_tensors


class SimpleConvNet(torch.nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run():
    model = SimpleConvNet()

    input_data = torch.randn(1, 3, 224, 224)
    output_original = model(input_data)
    ai3.optimize(model)
    output = model(input_data)
    compare_tensors(output.detach().numpy(),
                    output_original.detach().numpy(), "simple conv2d")


if __name__ == "__main__":
    run()
