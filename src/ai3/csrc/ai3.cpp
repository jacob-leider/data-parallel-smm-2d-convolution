// TODO it is actually normal to put template implementations in header files
// so do that but we can't conditionall include header files
// TODO in future CPP code make a wrapper for torch and tensorflow which
// implement the same thing so we can easily change the backend
#include <torch/extension.h>

#define DECLARE_KN2ROW_CONV2D(T)                                               \
    at::Tensor kn2row_conv2d(                                                  \
        const T *input, const std::vector<int> input_shape, const T *kernel,   \
        const std::vector<int> kernel_shape,                                   \
        const std::optional<const T *> bias,                                   \
        const std::vector<int> output_shape, const std::vector<int> padding,   \
        const std::vector<int> stride, const std::vector<int> dilation);

DECLARE_KN2ROW_CONV2D(float);
DECLARE_KN2ROW_CONV2D(double);

#define CALL_KN2ROW_CONV2D(T)                                                  \
    return kn2row_conv2d(                                                      \
        static_cast<const T *>(input_store.data()), input_shape,               \
        static_cast<const T *>(kernel_store.data()), kernel_shape,             \
        bias.has_value() ? std::make_optional<const T *>(                      \
                               static_cast<const T *>(bias.value().data()))    \
                         : std::nullopt,                                       \
        output_shape, padding, stride, dilation);

// TODO don't think all these references are the same
// also make sure all the same parameters
at::Tensor kn2row_conv2d_entry(
    const at::Storage input_store, const std::vector<int> input_shape,
    const at::Storage kernel_store, const std::vector<int> kernel_shape,
    const std::optional<at::Storage> bias, const std::vector<int> output_shape,
    const std::string dtype, const std::vector<int> padding,
    const std::vector<int> stride, const std::vector<int> dilation) {
    if (dtype != "torch.float64" && dtype != "torch.float32") {
        std::cerr << "Unsupported data type." << dtype << std::endl;
        std::exit(1);
    }
    if (dtype == "torch.float32") {
        CALL_KN2ROW_CONV2D(float);
    }
    CALL_KN2ROW_CONV2D(double);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kn2row_conv2d", &kn2row_conv2d_entry, "2d convolution using kn2row");
}
