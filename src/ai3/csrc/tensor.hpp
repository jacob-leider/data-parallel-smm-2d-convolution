#pragma once

#include <cassert>
#include <numeric>
#include <optional>

template <typename dtype> class Tensor {
  public:
    Tensor(const dtype *d, const std::vector<int> &s)
        : data(d, d + total_elem(s)), shape(s) {}
    Tensor(const intptr_t data_address, const std::vector<int> &s)
        : Tensor(static_cast<const dtype *>((void *)data_address), s) {}

    Tensor(const std::vector<int> &s) : Tensor(new dtype[total_elem(s)], s) {}

    ~Tensor() = default;

    static std::optional<Tensor>
    from_optional(const std::optional<intptr_t> &data_address,
                  const std::vector<int> &s) {
        return data_address.has_value()
                   ? std::make_optional(Tensor<dtype>(data_address.value(), s))
                   : std::nullopt;
    }

    template <typename... Indices>
    inline const dtype &at(Indices... indices) const {
        return data[linear_index(indices...)];
    }

    template <typename... Indices> inline dtype &at(Indices... indices) {
        return data[linear_index(indices...)];
    }

    static inline int num_data(const std::vector<int> &shape) {
        return shape[shape.size() - 4];
    }
    static inline int kern_out_channels(const std::vector<int> &shape) {
        return shape[shape.size() - 4];
    }
    static inline int input_channels(const std::vector<int> &shape) {
        return shape[shape.size() - 3];
    }
    static inline int height(const std::vector<int> &shape) {
        return shape[shape.size() - 2];
    }
    static inline int width(const std::vector<int> &shape) {
        return shape[shape.size() - 1];
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
