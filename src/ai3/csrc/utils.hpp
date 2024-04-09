#pragma once

// TODO actually use this type instead of int in most places
using uint = unsigned int;

#define IMPL_ENTRY_FOR_DOUBLE_FLOAT                                            \
    at::Tensor kn2row_conv2d(                                                  \
        const float *input, const std::vector<int> input_shape,                \
        const float *kernel, const std::vector<int> kernel_shape,              \
        const std::optional<const float *> bias,                               \
        const std::vector<int> output_shape, const std::vector<int> padding,   \
        const std::vector<int> stride, const std::vector<int> dilation) {      \
        return run(input, input_shape, kernel, kernel_shape, bias,             \
                   output_shape, padding, stride, dilation);                   \
    }                                                                          \
    at::Tensor kn2row_conv2d(                                                  \
        const double *input, const std::vector<int> input_shape,               \
        const double *kernel, const std::vector<int> kernel_shape,             \
        const std::optional<const double *> bias,                              \
        const std::vector<int> output_shape, const std::vector<int> padding,   \
        const std::vector<int> stride, const std::vector<int> dilation) {      \
        return run(input, input_shape, kernel, kernel_shape, bias,             \
                   output_shape, padding, stride, dilation);                   \
    }
