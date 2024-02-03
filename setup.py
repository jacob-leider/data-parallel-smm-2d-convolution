from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import glob

setup(
    name='ai3_functions',
    ext_modules=[
        CppExtension(
            name='ai3_functions',
            sources=glob.glob('ai3/csrc/*.cpp'))
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
