#pragma once

#include "ai3.hpp"
#include <optional>

template <typename dtype> Tensor _relu(Tensor input) {
    int total_elements = input.count();
    dtype *in_data = data_as<dtype>(input.data);
    for (int i = 0; i < total_elements; i++) {
        in_data[i] = (in_data[i] > 0) ? in_data[i] : 0;
    }
    return input;
}
