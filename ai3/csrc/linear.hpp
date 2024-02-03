#pragma once

#include <torch/extension.h>

torch::Tensor linear(torch::Tensor input, torch::Tensor weight,
                     torch::Tensor bias);
