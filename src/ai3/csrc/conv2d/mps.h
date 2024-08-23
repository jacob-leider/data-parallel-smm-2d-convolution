#pragma once

#include "ai3.hpp"
#include <optional>

Tensor<double> mps_conv2d(Tensor<double> input, const Tensor<double> &kernel,
                          const std::optional<const Tensor<double>> &bias,
                          const uint padding_h, const uint padding_w,
                          const uint stride_h, const uint stride_w,
                          const uint dilation_h, const uint dilation_w,
                          const PaddingMode padding_mode, uint groups);
Tensor<float> mps_conv2d(Tensor<float> input, const Tensor<float> &kernel,
                         const std::optional<const Tensor<float>> &bias,
                         const uint padding_h, const uint padding_w,
                         const uint stride_h, const uint stride_w,
                         const uint dilation_h, const uint dilation_w,
                         const PaddingMode padding_mode, uint groups);
