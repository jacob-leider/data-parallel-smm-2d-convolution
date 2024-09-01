#pragma once

#include "ai3.hpp"
#include <optional>

template <typename dtype>
Tensor mps_conv2d(Tensor input, const Tensor &kernel,
                  const std::optional<const Tensor> &bias, const uint padding_h,
                  const uint padding_w, const uint stride_h,
                  const uint stride_w, const uint dilation_h,
                  const uint dilation_w, const PaddingMode padding_mode,
                  uint groups);
