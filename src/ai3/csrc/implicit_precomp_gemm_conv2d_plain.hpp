#pragma once

#include "ai3.hpp"

template <typename dtype>
Tensor<dtype> implicit_precomp_gemm_conv2d(
    Tensor<dtype> input, const Tensor<dtype> &kernel,
    const std::optional<const Tensor<dtype>> &bias, const uint padding_h,
    const uint padding_w, const uint stride_h, const uint stride_w,
    const uint dilation_h, const uint dilation_w,
    const PaddingMode padding_mode, uint groups, Context &ctx) {
    (void)input;
    (void)kernel;
    (void)bias;
    (void)padding_h;
    (void)padding_w;
    (void)stride_h;
    (void)stride_w;
    (void)dilation_h;
    (void)dilation_w;
    (void)padding_mode;
    (void)groups;
    (void)ctx;
    errs::bail("implicit precomp gemm not implemented outside of cuDNN");
}
