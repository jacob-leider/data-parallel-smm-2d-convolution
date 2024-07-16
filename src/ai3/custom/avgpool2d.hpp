#pragma once

#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_AVGPOOL2D = false;

template <typename dtype>
Tensor<dtype> avgpool2d(Tensor<dtype> input, const uint kernel_h,
                        const uint kernel_w, const uint padding_h,
                        const uint padding_w, const uint stride_h,
                        const uint stride_w, const bool ceil_mode,
                        const bool count_include_pad,
                        const std::optional<int> divisor_override) {
    errs::no_user_def("avgpool2d");
}
