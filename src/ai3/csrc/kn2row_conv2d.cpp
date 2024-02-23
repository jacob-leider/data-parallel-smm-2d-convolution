#include "kn2row_conv2d.hpp"
#include <cstdlib>

// TODO other parameters from the PyTorch impl
// groups
// dilation
// stride
// TODO need to do valid, full, same padding
template <typename dtype>
at::Tensor
kn2row_conv2d(const dtype *input, const at::IntArrayRef input_shape,
              const dtype *kernel, const at::IntArrayRef weight_shape,
              const std::optional<const dtype *> bias,
              const std::vector<int> padding, const at::Tensor output) {
    const int input_channels = input_shape[1];
    const int input_height = input_shape[2];
    const int input_width = input_shape[3];
    const int output_channels = weight_shape[0];
    const int kernel_height = weight_shape[2];
    const int kernel_width = weight_shape[3];

    const int output_height = output.size(2);
    const int output_width = output.size(3);

    const int kernel_size = kernel_height * kernel_width;

    for (int out_c = 0; out_c < output_channels; ++out_c) {
        for (int out_h = 0; out_h < output_height; ++out_h) {
            for (int out_w = 0; out_w < output_width; ++out_w) {
                dtype sum = 0;
                for (int in_c = 0; in_c < input_channels; ++in_c) {
                    for (int kern_h = 0; kern_h < kernel_height; ++kern_h) {
                        for (int kern_w = 0; kern_w < kernel_width; ++kern_w) {
                            // should put some stuff on padding
                            // and stride when we do that
                            int h_offset = out_h - padding[0] + kern_h;
                            int w_offset = out_w - padding[1] + kern_w;

                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                sum +=
                                    input[in_c * input_height * input_width +
                                          h_offset * input_width + w_offset] *
                                    kernel[out_c * input_channels *
                                               kernel_size +
                                           in_c * kernel_size +
                                           kern_h * kernel_width + kern_w];
                            }
                        }
                    }
                }
                if (bias.has_value()) {
                    sum += bias.value()[out_c];
                }
                output[0][out_c][out_h][out_w] = sum;
            }
        }
    }
    return output;
}

template <typename dtype>
at::Tensor kn2row_conv2d_caster(const at::Storage &input_store,
                                const at::IntArrayRef input_shape,
                                const at::Storage &kernel_store,
                                const at::IntArrayRef kernel_shape,
                                const std::optional<at::Storage> &bias,
                                const std::vector<int> padding,
                                const at::Tensor &output) {
    return kn2row_conv2d(
        static_cast<const dtype *>(input_store.data()), input_shape,
        static_cast<const dtype *>(kernel_store.data()), kernel_shape,
        bias.has_value() ? std::make_optional<const dtype *>(
                               static_cast<const dtype *>(bias.value().data()))
                         : std::nullopt,
        padding, output);
}

at::Tensor kn2row_conv2d_entry(
    const at::Storage &input_store, const at::IntArrayRef input_shape,
    const at::Storage &kernel_store, const at::IntArrayRef kernel_shape,
    const std::string &dtype, const std::optional<at::Storage> &bias,
    const std::vector<int> padding, const at::Tensor &output) {
    if (dtype != "torch.float64" && dtype != "torch.float32") {
        std::cerr << "Unsupported data type." << dtype << std::endl;
        std::exit(1);
    }
    auto caster = kn2row_conv2d_caster<float>;
    if (dtype == "torch.float64") {
        caster = kn2row_conv2d_caster<double>;
    }
    return caster(input_store, input_shape, kernel_store, kernel_shape, bias,
                  padding, output);
}
