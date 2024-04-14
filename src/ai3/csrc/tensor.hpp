#pragma once

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

// std::accumulate(begin(vars), end(vars), 1.0, std::multiplies<double>());
// TODO can we define accesses here
template <typename dtype> struct Tensor {
    Tensor(const std::vector<dtype> &d, const std::vector<int> &s)
        : data(d), shape(s) {}
    Tensor(const dtype *d, const std::vector<int> &s)
        : data(d, d + std::accumulate(std::begin(s), std::end(s), 1,
                                      std::multiplies<int>())),
          shape(s) {}
    ~Tensor() = default;

    std::vector<dtype> data;
    std::vector<int> shape;

    const dtype &operator[](int index) const { return data[index]; }
};

template <typename dtype>
Tensor<dtype> formTensorFrom(const intptr_t data_address,
                             const std::vector<int> shape) {
    return Tensor<dtype>(static_cast<const dtype *>((void *)data_address),
                         shape);
}
