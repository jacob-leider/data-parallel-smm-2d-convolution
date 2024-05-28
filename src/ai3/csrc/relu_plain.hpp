#pragma once

#include "tensor.hpp"
#include <optional>
#include <vector>

template <typename dtype> Tensor<dtype> relu(const Tensor<dtype> &input) {
    Tensor<dtype> output(input.shape);
    auto output_iter = output.data.begin();

    for (const auto &value : input.data) {
        *output_iter++ = std::max(value, static_cast<dtype>(0));
    }

    return output;
}
