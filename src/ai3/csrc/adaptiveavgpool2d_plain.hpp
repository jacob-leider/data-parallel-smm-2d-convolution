#pragma once

#include "ai3.hpp"
#include "avgpool2d_plain.hpp"
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> _adaptiveavgpool2d(Tensor<dtype> input,
                                 const std::optional<uint> output_h,
                                 const std::optional<uint> output_w) {
    uint input_height = input.height();
    uint input_width = input.width();
    const uint output_height = output_h.value_or(input_height);
    const uint output_width = output_w.value_or(input_width);
    errs::bail_if(
        input_height % output_height != 0 || input_width % output_width != 0,
        "Adaptive average pooling not implemented for cases where "
        "input size is not a multiple of output size given input shape=(",
        input_height, ", ", input_width, ") and output shape=(", output_height,
        ", ", output_width, ")");
    const uint stride_h = input_height / output_height;
    const uint stride_w = input_width / output_width;

    const uint kernel_h = input_height - ((output_height - 1) * stride_h);
    const uint kernel_w = input_width - ((output_width - 1) * stride_w);

    return _avgpool2d<dtype>(std::move(input), kernel_h, kernel_w, 0, 0,
                             stride_h, stride_w, false, false, std::nullopt);
}
