# TODO not sure how to set up dependencies or this file, need either a Pytorch model or a .onnx file but not both
# - we are not always building a PyTorch extension
# optional flag when installing, something like pip install --frontend=torch/onnx
# TODO try the model in slack paper with transformers
# TODO more algorithms to implement
# TODO for onnx support, should be pretty easy to also iterate
# through the onnx layers and hyperparametrs. In both create a class
# which has a list of the functions with all the hparams. This is then passed
# to C++ which creates and returns the model, then forwarding is done fully
# in C++
# - im2col
# torch.cuda.is_available()
# torch.backends.mps.is_available()
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

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
        CppExtension(
            name=PKG + '.core',
            sources=sources())
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
