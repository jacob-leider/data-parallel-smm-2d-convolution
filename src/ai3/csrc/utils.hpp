#ifndef UTILS
#define UTILS

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

inline int out_channels(const std::vector<int> &shape) {
    return shape[shape.size() - 4];
}
inline int num_data(const std::vector<int> &shape) {
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
} // namespace dims
#endif
