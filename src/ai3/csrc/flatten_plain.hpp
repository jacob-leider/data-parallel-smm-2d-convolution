#pragma once

#include "ai3.hpp"
#include <iostream>
#include <vector>

template <typename dtype>
Tensor<dtype> _flatten(Tensor<dtype> input, const int start_dim, int end_dim) {
    errs::bail_if(end_dim != -1 && start_dim > end_dim,
                  "start dimension > end dimension in flattening function");
    if (end_dim == -1) {
        end_dim = input.shape.size() - 1;
    }
    const int new_num_dim = input.shape.size() - (end_dim - start_dim);
    errs::bail_if(
        new_num_dim < 0,
        "tensor would have a negative number of dimensions after flattening");

    std::vector<int> new_shape(new_num_dim);
    int flat = 1;
    int shift = 0;
    for (int dim = 0; dim < int(input.shape.size()); dim++) {
        if (dim < start_dim || dim > end_dim) {
            new_shape[dim - shift] = input.shape[dim];
        } else {
            flat *= input.shape[dim];
            if (dim == end_dim) {
                new_shape[start_dim] = flat;
                shift = end_dim - start_dim;
            }
        }
    }
    input.shape = new_shape;
    return input;
}
