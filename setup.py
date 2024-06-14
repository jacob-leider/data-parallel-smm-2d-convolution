# TODO July 7th deadline
# TODO write adaptiveavgpool to use a separate function so that users can override
# TODO bench suite on all the imagenet winners
# TODO try some way to do the normal pip install without compiling cpp code then taking users cpp code and
# compiling the SO with that. This would stop requiring building from source

# TODO would be best to split up samples then each process also has its way of
# accelerating instead of doing all the samples in each layer
# TODO onnx support, should be pretty easy to also iterate
# through the onnx layers and hyperparametrs, could also use the pytorch way
# of loading .onnx, .onnx -> nn.Module -> ai3.optimize -> ai3.Model
# TODO not sure how to set up dependencies or this file, need either a Pytorch model or a .onnx file but not both
# optional flag when installing, something like pip install --frontend=torch/onnx
# TODO move the acpp flags I set by default to cmake, try acpp in CMAKE first then do icpx in cmake on remote
# - also use the setup.cfg thing
# - in CMAKE use release flags for compiler optimizations

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import shutil

DIR = 'src'
PKG = 'ai3'
PKG_PATH = os.path.join(DIR, PKG)
CSRC = 'csrc'
CSRC_PATH = os.path.join(PKG_PATH, CSRC)
PKG_CSRC = PKG + '.' + CSRC


def form_path(file_name: str):
    return os.path.join(CSRC_PATH, file_name + '.cpp')


def sources():
    return [form_path('ai3')]


class Builder(build_ext):
    def build_extensions(self) -> None:
        extra_compile_args = ['-Wall', '-Werror']
        extra_link_args = []
        icpx_path = shutil.which("icpx")
        acpp_path = shutil.which("acpp")
        icpx_path = None
        acpp_path = None
        if icpx_path or acpp_path:
            print('Using SYCL compiler')
            extra_compile_args.append('-DAI3_USE_SYCL')
            extra_link_args.append('-shared')
            if icpx_path:
                print('Using icpx')
                self.compiler.set_executable('compiler_so', icpx_path)
                self.compiler.set_executable('compiler_cxx', icpx_path)
                self.compiler.set_executable('linker_so', icpx_path)
                extra_compile_args.extend(['-fsycl', '-fpic'])
                extra_link_args.append('-lsycl')
            elif acpp_path:
                print('Using acpp')
                self.compiler.set_executable('compiler_so', acpp_path)
                self.compiler.set_executable('compiler_cxx', acpp_path)
                self.compiler.set_executable('linker_so', acpp_path)
                if not self.debug:
                    print('no debug')
                    extra_compile_args.append('-O3')
        for ext in self.extensions:
            ext.extra_compile_args.extend(extra_compile_args)
            ext.extra_link_args.extend(extra_link_args)
        return super().build_extensions()


if __name__ == "__main__":
    setup(
        name=PKG,
        packages=[PKG, PKG_CSRC],
        package_dir={PKG: PKG_PATH,
                     PKG_CSRC: CSRC_PATH},
        ext_modules=[
            Pybind11Extension(
                name=PKG + '.core',
                sources=sources())
        ],
        cmdclass={
            'build_ext': Builder
        },
    )
