#pragma once

#include "ai3.hpp"
#include "utils.hpp"
#include <cstddef>
#include <iostream>
#include <optional>

template <typename dtype>
Tensor<dtype> direct_conv2d(Tensor<dtype> input, const Tensor<dtype> &kernel,
                            const std::optional<const Tensor<dtype>> &bias,
                            const uint padding_h, const uint padding_w,
                            const uint stride_h, const uint stride_w,
                            const uint dilation_h, const uint dilation_w,
                            const PaddingMode padding_mode, uint groups) {
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();

    const uint kernel_height = kernel.height();
    const uint kernel_width = kernel.width();

    const uint output_channels = kernel.out_channels();

    const uint output_height = output_hw_for_2d<dtype>(
        input_height, kernel_height, padding_h, dilation_h, stride_h, false);
    const uint output_width = output_hw_for_2d<dtype>(
        input_width, kernel_width, padding_w, dilation_w, stride_w, false);

    uint num_samples;
    Tensor<dtype> output;
    if (input.batched(input_dims::CONV2D)) {
        num_samples = input.batch_size(input_dims::CONV2D);
        output = Tensor<dtype>(
            {num_samples, output_channels, output_height, output_width});
    } else {
        num_samples = 1;
        output = Tensor<dtype>({output_channels, output_height, output_width});
    }

    const bool has_bias = bias.has_value();

    for (uint samp = 0; samp < num_samples; samp++) {
        for (uint out_c = 0; out_c < output_channels; out_c++) {
            for (uint out_h = 0; out_h < output_height; out_h++) {
                for (uint out_w = 0; out_w < output_width; out_w++) {
                    dtype res = 0;
                    for (uint in_c = 0; in_c < input_channels; ++in_c) {
                        for (uint ker_h = 0; ker_h < kernel_height; ++ker_h) {
                            for (uint ker_w = 0; ker_w < kernel_width;
                                 ++ker_w) {
                                uint h_offset = out_h * stride_h - padding_h +
                                                ker_h * dilation_h;
                                uint w_offset = out_w * stride_w - padding_w +
                                                ker_w * dilation_w;

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
                    if (has_bias) {
                        res += bias->data[out_c];
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
