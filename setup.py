# TODO put all the hyperparameter stuff done in the tests in the layer creation
# TODO after beginning benchmarking on the vgg16, then try separating the outer three for loops to a function which takes
# a lambda to do the other processing
# TODO for onnx backend could use Torches frontend for onnx which would make everything easier. I really don't want to make torch a required
# dependency though
# TODO could checkout KAN networks and provide support for their forwarding, don't think it should actually be bad to just do inferencing,
# and could make it pretty cool

# TODO after changing model, do vgg16 and inception compare the performance, impact on energy,
# BME people have UNET model but their images is too large and run out of memory (good scientific use case)
# - TODO bench against intel extension for pytorch and CUDNN have to check if it has Python support
# - TODO then do the transformer model

# TODO not sure how to set up dependencies or this file, need either a Pytorch model or a .onnx file but not both
# - we are not always building a PyTorch extension
# optional flag when installing, something like pip install --frontend=torch/onnx
# TODO onnx support, should be pretty easy to also iterate
# through the onnx layers and hyperparametrs. In both create a class
# which has a list of the functions with all the hparams. This is then passed
# to C++ which creates and returns the model, then forwarding is done fully
# in C++

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

DIR = 'src'
PKG = 'ai3'

def form_path(file_name: str):
    return os.path.join(DIR, PKG, 'csrc', file_name + '.cpp')

def sources():
    return [form_path('ai3')]

if __name__ == "__main__":
    setup(
        name='ai3',
        packages=find_packages(where=DIR),
        package_dir={"": DIR},
        ext_modules=[
            Pybind11Extension(
                name=PKG + '.core',
                sources=sources())
        ],
        cmdclass={
            'build_ext': build_ext
        },
    )
