#pragma once

#include "tensor.hpp"
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> linear(const Tensor<dtype> &input, const Tensor<dtype> &weight,
                     const std::optional<const Tensor<dtype>> &bias) {
    assert(weight.shape[1] == input.shape[0]);

    const int in_features = input.shape[0];
    const int out_features = weight.shape[0];

    Tensor<dtype> output = Tensor<dtype>({out_features});

    for (int i = 0; i < out_features; i++) {
        dtype res = 0;
        for (int j = 0; j < in_features; ++j) {
            res += weight.at(i, j) * input.at(j);
        }
        if (bias.has_value()) {
            res += bias.value().at(i);
        }
        output.at(i) = res;
    }

    return output;
}
