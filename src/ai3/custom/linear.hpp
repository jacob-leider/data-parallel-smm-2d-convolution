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
Tensor linear(Tensor input, const Tensor &weight,
              const std::optional<const Tensor> &bias) {
    errs::no_user_def("linear");
}
