#pragma once

#include <memory>
#include <numeric>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename dtype> class Tensor {
  public:
    Tensor(const intptr_t data_address, const std::vector<int> &s,
           bool input_data = false)
        : shape(s), owned(!input_data) {
        if (owned) {
            data = new dtype[total_elem(s)];
            std::memcpy(data, reinterpret_cast<const dtype *>(data_address),
                        total_elem(s) * sizeof(dtype));
        } else {
            data = reinterpret_cast<dtype *>(data_address);
        }
    }

    Tensor(const std::vector<int> &s)
        : data(new dtype[total_elem(s)]), shape(s), owned(true) {}

    static std::optional<Tensor>
    from_optional(const std::optional<intptr_t> &data_address,
                  const std::vector<int> &s, bool input_data = false) {
        if (data_address.has_value()) {
            return Tensor<dtype>((*data_address), s, input_data);
        } else {
            return std::nullopt;
        }
    }

    Tensor() = default;

    ~Tensor() {
        if (owned) {
            delete[] data;
        }
    };

    Tensor(Tensor &&other) noexcept { *this = std::move(other); }

    Tensor &operator=(Tensor &&other) noexcept {
        if (this != &other) {
            data = other.data;
            shape = std::move(other.shape);
            owned = other.owned;
            other.data = nullptr;
        }
        return *this;
    }

    py::buffer_info buffer() {
        std::vector<int> stride(shape.size());
        stride[shape.size() - 1] = sizeof(dtype);
        for (int i = shape.size() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        return py::buffer_info(data, sizeof(dtype),
                               py::format_descriptor<dtype>::format(),
                               shape.size(), shape, stride);
    }

    static int total_elem(const std::vector<int> &shape) {
        return std::accumulate(shape.begin(), shape.end(), 1,
                               std::multiplies<int>());
    }

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    dtype *data;
    std::vector<int> shape;
    bool owned;
};
