# TODO July 7th deadline
# TODO once we have a way to just swap out the layers, try to
# run torch.compile on it

# TODO in the torch.compile see if we can inject our convolution algorithms as
# customized by the user into it,
# TODO an easy bench suite to bench on the same 140ish models that PyTorch benchmarks on

# TODO onnx support, should be pretty easy to also iterate
# through the onnx layers and hyperparametrs, could also use the pytorch way
# of loading .onnx, .onnx -> nn.Module -> ai3.optimize -> ai3.Model
# TODO not sure how to set up dependencies or this file, need either a Pytorch model or a .onnx file but not both
# - we are not always building a PyTorch extension
# optional flag when installing, something like pip install --frontend=torch/onnx
# TODO could checkout KAN networks and provide support for their forwarding, don't think it should actually be bad to just do inferencing,
# and could make it pretty cool

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
