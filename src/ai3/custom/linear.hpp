#pragma once
#include <ai3.hpp>

// #define LINEAR_CUSTOM
template <typename dtype>
Tensor<dtype> linear(Tensor<dtype> input, const Tensor<dtype> &weight,
                     const std::optional<const Tensor<dtype>> &bias);
