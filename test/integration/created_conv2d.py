import torch
import ai3
from test import compare_tensors
from example.simple_conv2d import SimpleConvNet


def _run(data, mes):
    model = SimpleConvNet()
    ai3_model = ai3.swap_backend(model)
    target = model(data)
    output = ai3_model.predict(data)
    compare_tensors(output, target.detach().numpy(), mes)


def run():
    print("SIMPLE CREATED CONV NET")
    _run(torch.randn(3, 224, 224), "simple conv2d")
    _run(torch.randn(5, 3, 224, 224), "simple conv2d multi sample")


if __name__ == "__main__":
    run()
