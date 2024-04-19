#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

template <typename dtype> class Tensor {
  public:
    Tensor(const dtype *d, const std::vector<int> &s)
        : data(d, d + total_elem(s)), shape(s) {}
    Tensor(const intptr_t data_address, const std::vector<int> &s)
        : Tensor(static_cast<const dtype *>((void *)data_address), s) {}

    Tensor(const std::vector<int> &s) : Tensor(new dtype[total_elem(s)], s) {}

    ~Tensor() = default;

    const dtype &operator[](int index) const { return data[index]; }
    dtype &operator[](int index) { return data[index]; }

    template <typename... Indices>
    inline const dtype &at(Indices... indices) const {
        return data[linear_index(indices...)];
    }

    template <typename... Indices> inline dtype &at(Indices... indices) {
        return data[linear_index(indices...)];
    }

    std::vector<dtype> data;
    std::vector<int> shape;

  private:
    static int total_elem(const std::vector<int> &shape) {
        return std::accumulate(std::begin(shape), std::end(shape), 1,
                               std::multiplies<int>());
    }

    template <typename... Indices> int linear_index(Indices... indices) const {
        int idx = 0;
        int multiplier = 1;
        int indexArr[] = {indices...};
        for (int i = shape.size() - 1; i >= 0; --i) {
            idx += indexArr[i] * multiplier;
            multiplier *= shape[i];
        }
        return idx;
    }
};
