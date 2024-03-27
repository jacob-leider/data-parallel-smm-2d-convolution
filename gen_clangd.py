import subprocess
import torch
import os


def get_python_include():
    python_include = subprocess.check_output(
        ['python3-config', '--include']).decode().strip().split()
    return python_include


def create_clangd_file(file_path):
    flags = [
        '-Wall',
        '-Wextra',
        '-Werror',
        '-std=c++17',
        '-I' +
        os.path.join(os.path.dirname(torch.__file__), 'include'),
        '-I' + os.path.join(os.path.dirname(torch.__file__),
                            'include', 'torch', 'csrc', 'api', 'include'),
        *get_python_include()
    ]

    with open(file_path, 'w') as f:
        f.write('CompileFlags:\n')
        f.write('  Add:\n')
        for flag in flags:
            f.write(f'    - "{flag}"\n')


if __name__ == '__main__':
    create_clangd_file('.clangd')
