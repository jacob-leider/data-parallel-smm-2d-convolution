// To use this impl instead of the default put
// #define CONV2D_USER `path to this file` at top of `user_defined.hpp`
#pragma once
#include "ai3.hpp"
#include <iostream>

template <typename dtype>
Tensor<dtype> conv2d(const Tensor<dtype> &input, const Tensor<dtype> &kernel,
                     const std::optional<const Tensor<dtype>> &bias,
                     const std::vector<int> &padding,
                     const std::vector<int> &stride,
                     const std::vector<int> &dilation,
                     const PaddingMode padding_mode, int groups) {
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
