import subprocess
import pybind11
import os
from setup import sources
import sys

FLAGS = [
        '-Wall',
        '-Wextra',
        '-Werror',
        '-std=c++17',
        '-I' +
        os.path.join(os.path.dirname(pybind11.__file__), 'include'),
        *subprocess.check_output(
            ['python3-config', '--include']).decode().strip().split()
        ]

def gen_clangd(file_path):
    with open(file_path, 'w') as f:
        f.write('CompileFlags:\n')
        f.write('  Add:\n')
        for flag in FLAGS:
            f.write(f'    - "{flag}"\n')

def build():
    cmd = ['clang++', '-shared', '-o', 'output.so'] + sources() + FLAGS
    print(f"running: {cmd}")
    subprocess.run(cmd)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'gen-clangd':
            gen_clangd('.clangd')
        else:
            print(f"invalid arguments: {sys.argv}")
    else:
        build()
