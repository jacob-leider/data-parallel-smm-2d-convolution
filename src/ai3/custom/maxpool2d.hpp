#pragma once

#include <ai3.hpp>

const bool DEFAULT_TO_CUSTOM_MAXPOOL2D = false;

template <typename dtype>
Tensor<dtype> maxpool2d(Tensor<dtype> input, const uint kernel_h,
                        const uint kernel_w, const uint padding_h,
                        const uint padding_w, const uint stride_h,
                        const uint stride_w, const uint dilation_h,
                        const uint dilation_w, const bool ceil_mode) {
    errs::no_user_def("maxpool2d");
}
