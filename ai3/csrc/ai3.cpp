// TODO in future CPP code make a wrapper for torch and tensorflow which
// implement the same thing so we can easily change the backend
#include "linear.hpp"
#include <Python.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear, "Linear forward");
}

int main() {
    std::cout << "hello";
    return 0;
}
