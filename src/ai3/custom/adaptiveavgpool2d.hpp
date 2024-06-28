#pragma once

#include <ai3.hpp>

// #define ADAPTIVE_CUSTOM
template <typename dtype>
Tensor<dtype> _adaptiveavgpool2d(
    Tensor<dtype> input,
    const std::optional<std::vector<std::optional<uint>>> output_shape);
