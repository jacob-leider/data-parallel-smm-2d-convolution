// TODO in future CPP code make a wrapper for torch and tensorflow which
// implement the same thing so we can easily change the backend
// #include "kn2row_conv2d.hpp"
#include <Python.h>
#include <torch/extension.h>

extern at::Tensor kn2row_conv2d_entry(
    const at::Storage &input_store, const at::IntArrayRef input_shape,
    const at::Storage &kernel_store, const at::IntArrayRef kernel_shape,
    const std::string &dtype, const std::optional<at::Storage> &bias,
    const std::vector<int> padding, const at::Tensor &output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kn2row_conv2d", &kn2row_conv2d_entry, "Linear forward");
}
