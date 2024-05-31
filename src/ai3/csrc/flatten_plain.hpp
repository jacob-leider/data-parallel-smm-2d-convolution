#ifndef FLATTEN
#define FLATTEN

#include "errors.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <iostream>
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> flatten(const Tensor<dtype> &input, const int start_dim,
                      int end_dim) {
    bail_if(end_dim != -1 && start_dim > end_dim,
            "start dimension > end dimension in flattening function");
    if (end_dim == -1) {
        end_dim = input.shape.size() - 1;
    }
    int new_num_dim = input.shape.size() - (end_dim - start_dim);
    bail_if(
        new_num_dim < 0,
        "tensor would have a negative number of dimensions after flattening");

    std::vector<int> new_shape(new_num_dim);
    std::copy(input.shape.begin(), input.shape.begin() + start_dim,
              new_shape.begin());

    new_shape[start_dim] = std::accumulate(input.shape.begin() + start_dim,
                                           input.shape.begin() + end_dim + 1, 1,
                                           std::multiplies<int>());
    std::copy(input.shape.begin() + end_dim + 1, input.shape.end(),
              new_shape.begin() + start_dim + 1);

    return Tensor<dtype>(input.data.data(), new_shape);
}
#endif
