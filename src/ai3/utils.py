from torch import nn
from typing import Generator


def layers(model: nn.Module) -> Generator:
    return model.named_modules()
