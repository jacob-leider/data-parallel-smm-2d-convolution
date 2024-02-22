from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import glob

# TODO setup different sources depending on the platform

setup(
    name='ai3',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            name='ai3.core',
            sources=glob.glob('src/ai3/csrc/*.cpp'))
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
