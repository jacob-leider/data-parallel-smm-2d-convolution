#include "kn2row_conv2d.hpp"
#include <cstdlib>

using namespace std;

// TODO other parameters from the PyTorch impl
// groups
// dilation
// bias
// TODO need to do valid, full, same padding
template <typename dtype>
at::Tensor kn2row_conv2d(const at::Storage &input_store,
                         const at::IntArrayRef input_shape,
                         const at::Storage &weight_store,
                         const at::IntArrayRef weight_shape,
                         const c10::optional<at::Storage> &bias_store,
                         const at::Tensor output) {
    // TODO this whole function is written more like c then c++ see if c++ will
    // clean it up
    const dtype *kernel = static_cast<const dtype *>(weight_store.data());
    const dtype *input = static_cast<const dtype *>(input_store.data());
    std::optional<const dtype *> bias;
    if (bias_store.has_value()) {
        bias = static_cast<const dtype *>(bias_store.value().data());
    }

    int input_channels = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    int output_channels = weight_shape[0];
    int kernel_height = weight_shape[2];
    int kernel_width = weight_shape[3];

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    int kernel_size = kernel_height * kernel_width;

    for (int out_c = 0; out_c < output_channels; ++out_c) {
        for (int out_h = 0; out_h < output_height; ++out_h) {
            for (int out_w = 0; out_w < output_width; ++out_w) {
                dtype sum = 0;
                for (int c_in = 0; c_in < input_channels; ++c_in) {
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            // should put some stuff on padding
                            // and stride when we do that
                            int h_offset = out_h + kh;
                            int w_offset = out_w + kw;

                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                sum +=
                                    input[c_in * input_height * input_width +
                                          h_offset * input_width + w_offset] *
                                    kernel[out_c * input_channels *
                                               kernel_size +
                                           c_in * kernel_size +
                                           kh * kernel_width + kw];
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

at::Tensor kn2row_conv2d_entry(const at::Storage &input,
                               const at::IntArrayRef input_shape,
                               const at::Storage &weight,
                               const at::IntArrayRef weight_shape,
                               const std::string &dtype,
                               const c10::optional<at::Storage> &bias,
                               const at::Tensor &output) {
    if (dtype == "torch.float32") {
        return kn2row_conv2d<float>(input, input_shape, weight, weight_shape,
                                    bias, output);
    } else if (dtype == "torch.float64") {
        return kn2row_conv2d<double>(input, input_shape, weight, weight_shape,
                                     bias, output);
    }
    std::cerr << "Unsupported data type." << dtype << std::endl;
    std::exit(1);
}
