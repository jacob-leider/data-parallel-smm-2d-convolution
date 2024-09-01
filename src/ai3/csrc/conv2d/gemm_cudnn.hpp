#pragma once

#include "ai3.hpp"
#include "exec_cudnn.hpp"

template <typename dtype>
Tensor gemm_conv2d(Tensor input, const Tensor &kernel,
                   const std::optional<const Tensor> &bias,
                   const uint padding_h, const uint padding_w,
                   const uint stride_h, const uint stride_w,
                   const uint dilation_h, const uint dilation_w,
                   const PaddingMode padding_mode, const uint groups) {
    return conv_bias_forward_with_algo<dtype>(
        std::move(input), kernel, bias, padding_h, padding_w, stride_h,
        stride_w, dilation_h, dilation_w, padding_mode, groups,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
}
