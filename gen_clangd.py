import subprocess
import pybind11
import os


def run(file_path):
    flags = [
            '-Wall',
            '-Wextra',
            '-Werror',
            '-std=c++17',
            '-I' +
            os.path.join(os.path.dirname(pybind11.__file__), 'include'),
            *subprocess.check_output(
                ['python3-config', '--include']).decode().strip().split()
            ]

    with open(file_path, 'w') as f:
        f.write('CompileFlags:\n')
        f.write('  Add:\n')
        for flag in flags:
            f.write(f'    - "{flag}"\n')


if __name__ == '__main__':
    run('.clangd')
