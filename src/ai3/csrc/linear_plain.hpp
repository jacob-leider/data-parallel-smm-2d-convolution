#ifndef LINEAR
#define LINEAR

#include "errors.hpp"
#include "tensor.hpp"
#include "utils.hpp"
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
#endif
