// TODO in future CPP code make a wrapper for torch and tensorflow which
// implement the same thing so we can easily change the backend
#include "kn2row_conv2d.hpp"
#include <Python.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kn2row_conv2d", &kn2row_conv2d_entry, "Linear forward");
}
