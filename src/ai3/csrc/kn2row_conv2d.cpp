#include <cstdlib>
#include <torch/extension.h>

template <typename dtype>
at::Tensor
kn2row_conv2d(const dtype *input, const std::vector<int> input_shape,
              const dtype *kernel, const std::vector<int> weight_shape,
              const std::optional<const dtype *> bias,
              const std::vector<int> padding, const std::vector<int> stride,
              const std::vector<int> dilation, const at::Tensor output) {
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
                    for (int kern_r = 0; kern_r < kernel_height; ++kern_r) {
                        for (int kern_c = 0; kern_c < kernel_width; ++kern_c) {
                            int h_offset = out_h * stride[0] - padding[0] +
                                           kern_r * dilation[0];
                            int w_offset = out_w * stride[1] - padding[1] +
                                           kern_c * dilation[1];

                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                sum +=
                                    input[in_c * input_height * input_width +
                                          h_offset * input_width + w_offset] *
                                    kernel[out_c * input_channels *
                                               kernel_size +
                                           in_c * kernel_size +
                                           kern_r * kernel_width + kern_c];
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
at::Tensor kn2row_conv2d_caster(
    const at::Storage &input_store, const std::vector<int> input_shape,
    const at::Storage &kernel_store, const std::vector<int> kernel_shape,
    const std::optional<at::Storage> &bias, const std::vector<int> padding,
    const std::vector<int> stride, const std::vector<int> dilation,
    const at::Tensor &output) {
    return kn2row_conv2d(
        static_cast<const dtype *>(input_store.data()), input_shape,
        static_cast<const dtype *>(kernel_store.data()), kernel_shape,
        bias.has_value() ? std::make_optional<const dtype *>(
                               static_cast<const dtype *>(bias.value().data()))
                         : std::nullopt,
        padding, stride, dilation, output);
}

at::Tensor kn2row_conv2d_entry(
    const at::Storage &input_store, const std::vector<int> input_shape,
    const at::Storage &kernel_store, const std::vector<int> kernel_shape,
    const std::string &dtype, const std::optional<at::Storage> &bias,
    const std::vector<int> padding, const std::vector<int> stride,
    const std::vector<int> dilation, const at::Tensor &output) {
    if (dtype != "torch.float64" && dtype != "torch.float32") {
        std::cerr << "Unsupported data type." << dtype << std::endl;
        std::exit(1);
    }
    auto caster = kn2row_conv2d_caster<float>;
    if (dtype == "torch.float64") {
        caster = kn2row_conv2d_caster<double>;
    }
    return caster(input_store, input_shape, kernel_store, kernel_shape, bias,
                  padding, stride, dilation, output);
}
