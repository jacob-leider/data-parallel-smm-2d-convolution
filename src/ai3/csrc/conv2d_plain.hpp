#pragma once

#include "errors.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>

// TODO groups and padding modes
template <typename dtype>
Tensor<dtype> conv2d(const Tensor<dtype> &input, const Tensor<dtype> &kernel,
                     const std::optional<const Tensor<dtype>> &bias,
                     const std::vector<int> &padding,
                     const std::vector<int> &stride,
                     const std::vector<int> &dilation,
                     const PaddingMode padding_mode, int groups) {
    bail_if(padding_mode != Zeros, "padding mode must be zeroes");
    bail_if(groups != 1, "groups must be 1");

    const int input_channels = dims::input_channels(input.shape);
    const int input_height = dims::height(input.shape);
    const int input_width = dims::width(input.shape);

    const int kernel_height = dims::height(kernel.shape);
    const int kernel_width = dims::width(kernel.shape);

    const int output_channels = dims::out_channels(kernel.shape);

    const int output_height = dims::output_dim<dtype>(
        input_height, kernel_height, padding[0], dilation[0], stride[0], false);
    const int output_width = dims::output_dim<dtype>(
        input_width, kernel_width, padding[1], dilation[1], stride[1], false);

    int num_samples;
    Tensor<dtype> output;
    if (dims::has_dim_for_batch_size(input.shape, dims::input::CONV2D)) {
        num_samples = 1;
        output = Tensor<dtype>({output_channels, output_height, output_width});
    } else {
        num_samples = dims::batch_size(input.shape, dims::input::POOL2D);
        output = Tensor<dtype>(
            {num_samples, output_channels, output_height, output_width});
    }

    for (int samp = 0; samp < num_samples; samp++) {
        for (int out_c = 0; out_c < output_channels; out_c++) {
            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    dtype res = 0;
                    for (int in_c = 0; in_c < input_channels; ++in_c) {
                        for (int ker_h = 0; ker_h < kernel_height; ++ker_h) {
                            for (int ker_w = 0; ker_w < kernel_width; ++ker_w) {
                                int h_offset = out_h * stride[0] - padding[0] +
                                               ker_h * dilation[0];
                                int w_offset = out_w * stride[1] - padding[1] +
                                               ker_w * dilation[1];

                                if (h_offset >= 0 && h_offset < input_height &&
                                    w_offset >= 0 && w_offset < input_width) {
                                    res += input.data[to_linear(
                                               samp, in_c, h_offset, w_offset,
                                               input_channels, input_height,
                                               input_width)] *
                                           kernel.data[to_linear(
                                               out_c, in_c, ker_h, ker_w,
                                               input_channels, kernel_height,
                                               kernel_width)];
                                }
                            }
                        }
                    }
                    if (bias.has_value()) {
                        res += bias.value().data[out_c];
                    }
                    output.data[to_linear(samp, out_c, out_h, out_w,
                                          output_channels, output_height,
                                          output_width)] = res;
                }
            }
        }
    }

    return output;
}
