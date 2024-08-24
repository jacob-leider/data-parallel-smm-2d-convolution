import subprocess
import os
from pathlib import Path
import argparse

PY = 'python3 -m'
PIP = 'pip3'
C_FORMAT = 'clang-format'
PY_FORMAT = 'autopep8 --in-place --experimental'

CSRC_FILES = [
    str(f) for f in Path('.').rglob('*')
    if f.suffix in ['.cpp', '.hpp'] and 'venv' not in f.parts]
PY_FILES = [str(f) for f in Path('.').rglob('*.py') if 'venv' not in f.parts]

CONV2D_ALGOS_TO_USE = ['mps', 'metal']
"""The *conv2d* algorithms to use"""
USE_ALL_POSSIBLE_CONV = False
"""
Whether to automatically generate :data:`CONV2D_ALGOS_TO_USE` to contain all
possible algorithms
"""
if USE_ALL_POSSIBLE_CONV:
    assert len(CONV2D_ALGOS_TO_USE) == 0
    import ai3
    CONV2D_ALGOS_TO_USE.extend(['direct', 'smm'])
    if ai3.using_mps():
        CONV2D_ALGOS_TO_USE.append('mps')
    if ai3.using_cuda_tools():
        CONV2D_ALGOS_TO_USE.extend(['winograd', 'gemm',
                                    'implicit gemm', 'implicit precomp gemm',
                                    'guess'])


def run_command(command, cwd=None):
    print(f'Running: {command}')
    try:
        subprocess.run(command, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print('Error:', e)
        exit(1)


def run_sphinx(builder, out):
    run_command(f'sphinx-build -b {builder} docs docs/_build/{out}')


def gen_clangd(file_path):
    import pybind11
    flags = [
        '-Wall',
        '-Wextra',
        '-Werror',
        '-std=c++17',
        '-I' +
        os.path.join(os.getcwd(), 'src', 'ai3', 'csrc'),
        '-I' +
        os.path.join(os.path.dirname(pybind11.__file__), 'include'),
        *subprocess.check_output(
            ['python3-config', '--include']).decode().strip().split()
    ]
    with open(file_path, 'w') as f:
        def writeln(mes):
            f.write(f'{mes}\n')
        writeln('CompileFlags:')
        writeln('  Add:')
        for flag in flags:
            writeln(f'    - \'{flag}\'')


def build(editable: bool = False, verbose: bool = False, dev: bool = False):
    cxx_flags = ''
    cmd = f'{PIP} install'
    if editable:
        cmd += ' --editable'
    cmd += ' .'
    if dev:
        cmd += '[dev,doc]'
    if verbose:
        cmd += ' --verbose'
    if cxx_flags:
        cmd += f' --config-settings=cmake.define.CMAKE_CXX_FLAGS=\'{cxx_flags}\''
    run_command(cmd)


def starts_with_any(cmd, starts):
    return any(cmd.startswith(s) for s in starts)


def clean_start(cmd, starts):
    for s in starts:
        if cmd.startswith(s):
            return cmd[len(s) + 1:]
    assert False and 'cleaning with unmatched start'


def fix_cmd_run(cmd, starter):
    cmd = cmd.replace(f'{starter}.', f'{starter} ')
    run_command(f'{PY} {cmd}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run various development commands')
    parser.add_argument('commands', nargs='+')
    args = parser.parse_args()

    for cmd in args.commands:
        if cmd == 'clangd':
            gen_clangd('.clangd')
        elif cmd == 'install':
            build()
        elif cmd == 'install.e':
            build(editable=True)
        elif cmd == 'install.ev':
            build(editable=True, verbose=True)
        elif cmd == 'install.d':
            build(dev=True)
        elif cmd.startswith('example'):
            run_command(f'{PY} {cmd}')
        elif cmd == 'test':
            run_command(f'{PY} {cmd}')
        elif cmd == 'doctest':
            run_sphinx('doctest', 'doctest')
        elif cmd == 'docs':
            run_sphinx('html', '')
            run_command('make html', cwd='docs')
        elif cmd == 'readme':
            run_command(f'{PY} docs.gen_readme')
        elif cmd == 'format':
            run_command(f'{C_FORMAT} -i {' '.join(CSRC_FILES)}')
            run_command(f'{PY_FORMAT} {' '.join(PY_FILES)}')
        else:
            cmd_found = False
            for start in [
                'test.ops', 'test.swap_conv2d', 'test.swap_backend',
                'test.serialization.pickle', 'test.serialization.deepcopy',
                'bench.backward_step', 'bench.swap_conv2d',
                    'bench.swap_backend', 'bench.compile']:
                if cmd.startswith(start):
                    fix_cmd_run(cmd, start)
                    break
            else:
                run_command(f'{PY} {cmd}')
