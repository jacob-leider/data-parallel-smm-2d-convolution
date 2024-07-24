#pragma once
#include <ai3.hpp>

const bool DEFAULT_TO_CUSTOM_CONV2D = false;

template <typename dtype>
Tensor<dtype> conv2d(Tensor<dtype> input, const Tensor<dtype> &kernel,
                     const std::optional<const Tensor<dtype>> &bias,
                     const uint padding_h, const uint padding_w,
                     const uint stride_h, const uint stride_w,
                     const uint dilation_h, const uint dilation_w,
                     const PaddingMode padding_mode, uint groups) {
    errs::no_user_def("conv2d");
}
