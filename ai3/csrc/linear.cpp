#include "linear.hpp"

torch::Tensor linear(torch::Tensor input, torch::Tensor weight,
                     torch::Tensor bias) {
    if (input.dim() == 1) {
        input = input.unsqueeze(0);
    }

    return torch::addmm(bias, input, weight.transpose(0, 1)).squeeze();
}
