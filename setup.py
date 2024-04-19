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
