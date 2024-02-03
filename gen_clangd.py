import subprocess
import torch
import os


def get_python_include():
    python_include = subprocess.check_output(
        ["python3-config", "--include"]).decode().strip().split()
    return python_include


def create_clangd_file(file_path):
    compile_flags = {
        "CompileFlags": {
            "Add": [
                "-Wall",
                "-Wextra",
                "-Werror",
                "-std=c++17",
                "-I" +
                os.path.join(os.path.dirname(torch.__file__), "include"),
                "-I" + os.path.join(os.path.dirname(torch.__file__),
                                    "include", "torch", "csrc", "api", "include"),
                *get_python_include()
            ]
        }
    }

    with open(file_path, 'w') as f:
        for key, values in compile_flags.items():
            f.write(f"{key}:\n")
            for sub_key, sub_values in values.items():
                f.write(f"  {sub_key}:\n")
                for value in sub_values:
                    f.write(f"    - \"{value}\"\n")


if __name__ == "__main__":
    create_clangd_file(".clangd")
