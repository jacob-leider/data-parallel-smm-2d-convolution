#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{Linear}
 */
const bool DEFAULT_TO_CUSTOM_LINEAR = false;

/**
 * @CUSTOM_OP{Linear,linear}
 */
template <typename dtype>
Tensor<dtype> linear(Tensor<dtype> input, const Tensor<dtype> &weight,
                     const std::optional<const Tensor<dtype>> &bias) {
    errs::no_user_def("linear");
}
