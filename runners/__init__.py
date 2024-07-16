GROUPED_CONVOLUTION = False  # noqa
BATCH = 2  # noqa

from ai3 import errors
from .models import *


def run_on(runner, name):
    if len(name) == 0:
        run_all(runner)
    else:
        model_func = globals().get(name)
        if model_func:
            model_func(runner)
        else:
            errors.bail(f'Invalid model {name}')


def run_all(runner):
    alexnet(runner)
    convnext(runner)
    densenet(runner)
    efficientnet(runner)
    googlenet(runner)
    inception(runner)
    maxvit(runner)
    mnasnet(runner)
    mobilenet(runner)
    regnet(runner)
    resnet(runner)
    shufflenet(runner)
    simple_created(runner)
    squeezenet(runner)
    swintransformer(runner)
    vgg16(runner)
    visiontransformer(runner)
