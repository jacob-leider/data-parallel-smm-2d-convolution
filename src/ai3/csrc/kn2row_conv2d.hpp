#pragma once

#include "ATen/core/ATen_fwd.h"
#include <torch/extension.h>

at::Tensor kn2row_conv2d_entry(
    const at::Storage &input_store, const at::IntArrayRef input_shape,
    const at::Storage &kernel_store, const at::IntArrayRef kernel_shape,
    const std::string &dtype, const std::optional<at::Storage> &bias,
    const std::vector<int> padding, const at::Tensor &output);
