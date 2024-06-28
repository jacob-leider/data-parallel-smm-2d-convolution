#pragma once
#include "ai3.hpp"
#include <iostream>

template <typename dtype>
Tensor<dtype> conv2d(const Tensor<dtype> &input, const Tensor<dtype> &kernel,
                     const std::optional<const Tensor<dtype>> &bias,
                     const std::vector<uint> &padding,
                     const std::vector<uint> &stride,
                     const std::vector<uint> &dilation,
                     const PaddingMode padding_mode, uint groups) {
    (void)input;
    (void)kernel;
    (void)bias;
    (void)padding;
    (void)stride;
    (void)dilation;
    (void)padding_mode;
    (void)groups;
    std::cout << "user override" << std::endl;
    std::exit(1);
}
