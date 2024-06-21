#pragma once

#include "ai3.hpp"
#include <cassert>
#include <iostream>
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> _linear(const Tensor<dtype> &input, const Tensor<dtype> &weight,
                      const std::optional<const Tensor<dtype>> &bias) {
    errs::bail_if(
        dims::width(input.shape) != dims::width(weight.shape),
        "Invalid matrix multiplication: input width=", dims::width(input.shape),
        " weight width=", dims::width(weight.shape));

    const int in_features = dims::width(input.shape);
    const int out_features = dims::height(weight.shape);

    Tensor<dtype> output;
    int num_samples;
    if (dims::has_dim_for_batch_size(input.shape, dims::input::LINEAR)) {
        num_samples = 1;
        output = Tensor<dtype>({out_features});
    } else {
        num_samples = dims::batch_size(input.shape, dims::input::LINEAR);
        output = Tensor<dtype>({num_samples, out_features});
    }

    const bool has_bias = bias.has_value();
    for (int s = 0; s < num_samples; s++) {
        for (int i = 0; i < out_features; i++) {
            dtype res = 0;
            for (int j = 0; j < in_features; ++j) {
                res += weight.data[to_linear(i, j, in_features)] *
                       input.data[to_linear(s, j, in_features)];
            }
            if (has_bias) {
                res += bias->data[i];
            }
            output.data[to_linear(s, i, out_features)] = res;
        }
    }
    return output;
}
