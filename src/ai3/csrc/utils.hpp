#pragma once

#include "errors.hpp"
#include <iterator>
#include <optional>
#include <vector>

namespace dims {
template <typename dtype>
inline int output_dim(const int input, const int kernel, const int padding,
                      const std::optional<int> dilation, const int stride,
                      const bool ceil_mode) {
    const int top =
        input + (2 * padding) - (dilation.value_or(1) * (kernel - 1)) - 1;
    const int bot = stride;

    const int poss =
        (ceil_mode ? static_cast<int>(std::ceil(static_cast<dtype>(top) / bot))
                   : top / bot) +
        1;

    if (!ceil_mode) {
        return poss;
    }

    return (poss - 1) * stride >= input + padding ? poss - 1 : poss;
}

inline bool add_dim_for_samples(const std::vector<int> &shape,
                                int data_dim = -1) {
    if (data_dim == -1) {
        return false;
    }
    return shape.size() != unsigned(data_dim + 1);
}

inline int num_samples(const std::vector<int> &shape, const int input_dims) {
    return shape[shape.size() - 1 - input_dims];
}

inline int out_channels(const std::vector<int> &shape) {
    return shape[shape.size() - 4];
}

inline int input_channels(const std::vector<int> &shape) {
    return shape[shape.size() - 3];
}
inline int height(const std::vector<int> &shape) {
    return shape[shape.size() - 2];
}
inline int width(const std::vector<int> &shape) {
    return shape[shape.size() - 1];
}
namespace input {
const int LINEAR = 1;
const int POOL2D = 3;
const int CONV2D = 3;
const int ACTIVATION = -1;
const int FLATTEN = -1;
}; // namespace input
} // namespace dims

template <typename dtype>
void loop_over_out(int num_samples, int output_channels, int output_height,
                   int output_width,
                   std::function<void(int, int, int, int)> inner_processing) {
    for (int samp = 0; samp < num_samples; samp++) {
        for (int out_c = 0; out_c < output_channels; out_c++) {
            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    inner_processing(samp, out_c, out_h, out_w);
                }
            }
        }
    }
}
