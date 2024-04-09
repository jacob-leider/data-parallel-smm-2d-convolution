#include "utils.hpp"
#include <torch/extension.h>

template <typename dtype>
at::Tensor run(const dtype *input, const std::vector<int> input_shape,
               const dtype *kernel, const std::vector<int> kernel_shape,
               const std::optional<const dtype *> bias,
               const std::vector<int> output_shape,
               const std::vector<int> padding, const std::vector<int> stride,
               const std::vector<int> dilation) {
    const int input_channels = input_shape[1];
    const int input_height = input_shape[2];
    const int input_width = input_shape[3];
    const int kernel_height = kernel_shape[2];
    const int kernel_width = kernel_shape[3];

    const int output_channels = kernel_shape[0];
    const int output_height = output_shape[2];
    const int output_width = output_shape[3];
    const int kernel_size = kernel_height * kernel_width;

    dtype *output = new dtype[output_channels * output_height * output_width];

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
                output[out_c * output_height * output_width +
                       out_h * output_width + out_w] = sum;
            }
        }
    }
    return torch::from_blob(output,
                            {1, output_channels, output_height, output_width});
}

IMPL_ENTRY_FOR_DOUBLE_FLOAT
