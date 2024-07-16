#pragma once

#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_ADAPTIVEAVGPOOL2D = false;

template <typename dtype>
Tensor<dtype> adaptiveavgpool2d(Tensor<dtype> input,
                                const std::optional<uint> output_h,
                                const std::optional<uint> output_w) {
    errs::no_user_def("adaptiveavgpool2d");
}
