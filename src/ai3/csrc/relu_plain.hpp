#pragma once

#include "tensor.hpp"
#include <optional>
#include <vector>

template <typename dtype> Tensor<dtype> relu(Tensor<dtype> input) {
    for (int i = 0; i < Tensor<dtype>::total_elem(input.shape); i++) {
        input.data[i] = std::max(input.data[i], static_cast<dtype>(0));
    }
    return input;
}
