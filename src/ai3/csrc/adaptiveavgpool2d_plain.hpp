#pragma once

#include "ai3.hpp"
#include "avgpool2d_plain.hpp"
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> adaptiveavgpool2d(
    const Tensor<dtype> &input,
    const std::optional<std::vector<std::optional<int>>> output_shape) {
    int input_height = dims::height(input.shape);
    int input_width = dims::width(input.shape);
    const std::vector<std::optional<int>> opt_in_shape = {input_height,
                                                          input_width};
    const int output_height =
        output_shape.value_or(opt_in_shape)[0].value_or(input_height);
    const int output_width =
        output_shape.value_or(opt_in_shape)[1].value_or(input_width);
    errs::bail_if(
        input_height % output_height != 0 || input_width % output_width != 0,
        "Adaptive average pooling not implemented for cases where "
        "input size is not a multiple of output size given input shape=(",
        input_height, ", ", input_width, ") and output shape=(", output_height,
        ", ", output_width, ")");
    std::vector<int> stride = {input_height / output_height,
                               input_width / output_width};
    std::vector<int> kernel_shape = {
        input_height - ((output_height - 1) * stride[0]),
        input_width - ((output_width - 1) * stride[1])};
    return avgpool2d<dtype>(input, kernel_shape, {0, 0}, stride, false, false,
                            std::nullopt);
}
