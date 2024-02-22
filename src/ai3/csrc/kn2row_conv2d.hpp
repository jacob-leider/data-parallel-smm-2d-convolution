#pragma once

#include "ATen/core/ATen_fwd.h"
#include <torch/extension.h>

at::Tensor kn2row_conv2d_entry(const at::Storage &input,
                               const at::IntArrayRef input_shape,
                               const at::Storage &weight,
                               const at::IntArrayRef weight_shape,
                               const std::string &dtype,
                               const c10::optional<at::Storage> &bias,
                               const at::Tensor &output);
