#pragma once

#include "ai3.hpp"

// #define FLATTEN_CUSTOM
template <typename dtype>
Tensor<dtype> flatten(Tensor<dtype> input, const uint start_dim,
                      int end_dim_orig);
