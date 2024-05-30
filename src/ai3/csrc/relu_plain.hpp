#ifndef RELU
#define RELU

#include "tensor.hpp"
#include <optional>
#include <vector>

template <typename dtype> Tensor<dtype> relu(const Tensor<dtype> &input) {
    Tensor<dtype> output = input;
    for (dtype &value : output.data) {
        value = std::max(value, static_cast<dtype>(0));
    }
    return output;
}
#endif
