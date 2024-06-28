#pragma once

#include <ai3.hpp>

// #define MAXPOOL2D_CUSTOM
template <typename dtype>
Tensor<dtype>
maxpool2d(Tensor<dtype> input, const std::vector<uint> kernel_shape,
          const std::vector<uint> &padding, const std::vector<uint> &stride,
          const std::vector<uint> &dilation, const bool ceil_mode);
