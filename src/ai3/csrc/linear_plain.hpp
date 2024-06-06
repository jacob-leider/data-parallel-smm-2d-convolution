#pragma once

#include "errors.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <cassert>
#include <iostream>
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> linear(const Tensor<dtype> &input, const Tensor<dtype> &weight,
                     const std::optional<const Tensor<dtype>> &bias) {
    bail_if(
        dims::width(input.shape) != dims::width(weight.shape),
        "Invalid matrix multiplication: input width=", dims::width(input.shape),
        " weight width=", dims::width(weight.shape));

    const int in_features = dims::width(input.shape);
    const int out_features = dims::height(weight.shape);

    int num_samples = dims::num_samples(input.shape, dims::input::LINEAR);
    Tensor<dtype> output = Tensor<dtype>({num_samples, out_features});

    for (int s = 0; s < num_samples; s++) {
        for (int i = 0; i < out_features; i++) {
            dtype res = 0;
            for (int j = 0; j < in_features; ++j) {
                dtype in = input.at(s, j);
                res += weight.at(i, j) * in;
            }
            if (bias.has_value()) {
                res += bias.value().at(i);
            }
            output.at(s, i) = res;
        }
    }

    return output;
}
