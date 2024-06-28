#pragma once

using uint = unsigned int;
#include <cmath>
#include <optional>
#include <sstream>
#include <vector>

#ifndef GROUP_SIZE_GUESS
#define GROUP_SIZE_GUESS 256
#endif
#ifndef SAMPLES_PER_KERNEL
#define SAMPLES_PER_KERNEL 10
#endif

enum PaddingMode { Zeros, Reflect, Replicate, Circular };

inline uint to_linear(uint i, uint j, uint k, uint l, uint m, uint J, uint K,
                      uint L, uint M) {
    return (((i * J + j) * K + k) * L + l) * M + m;
}
inline uint to_linear(uint i, uint j, uint k, uint l, uint m, uint n, uint J,
                      uint K, uint L, uint M, uint N) {
    return ((((i * J + j) * K + k) * L + l) * M + m) * N + n;
}
inline uint to_linear(uint i, uint j, uint k, uint l, uint J, uint K, uint L) {
    return ((i * J + j) * K + k) * L + l;
}
inline uint to_linear(uint i, uint j, uint k, uint J, uint K) {
    return (i * J + j) * K + k;
}
inline uint to_linear(uint i, uint j, uint J) { return i * J + j; }

namespace errs {
template <typename... Args> [[noreturn]] void bail(Args... args) {
    std::stringstream ss;
    (ss << ... << args);
    throw std::runtime_error(ss.str());
}

template <typename... Args> void bail_if(bool check, Args... args) {
    if (check) {
        bail(args...);
    }
}
} // namespace errs

namespace dims {
template <typename dtype>
inline uint output_dim(const uint input, const uint kernel, const uint padding,
                       const std::optional<uint> dilation, const uint stride,
                       const bool ceil_mode) {
    const uint top =
        input + (2 * padding) - (dilation.value_or(1) * (kernel - 1)) - 1;
    const uint bot = stride;

    const uint poss =
        (ceil_mode ? static_cast<uint>(std::ceil(static_cast<dtype>(top) / bot))
                   : top / bot) +
        1;

    if (!ceil_mode) {
        return poss;
    }

    return (poss - 1) * stride >= input + padding ? poss - 1 : poss;
}

inline bool has_dim_for_batch_size(const std::vector<uint> &shape,
                                   int data_dim = -1) {
    if (data_dim == -1) {
        return false;
    }
    return shape.size() != unsigned(data_dim + 1);
}

inline uint batch_size(const std::vector<uint> &shape, const uint input_dims) {
    return shape[shape.size() - 1 - input_dims];
}

inline uint out_channels(const std::vector<uint> &shape) {
    return shape[shape.size() - 4];
}

inline uint input_channels(const std::vector<uint> &shape) {
    return shape[shape.size() - 3];
}
inline uint height(const std::vector<uint> &shape) {
    return shape[shape.size() - 2];
}
inline uint width(const std::vector<uint> &shape) {
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
