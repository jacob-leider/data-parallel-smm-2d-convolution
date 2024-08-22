// see if can do call once instead of static members
#include "ai3.hpp"
#include "algo_paths.hpp"
#include "pybind11/detail/common.h"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

inline bool is_default(std::string algo) { return algo == "default"; }
inline bool is_custom(std::string algo) { return algo == "default"; }

template <typename dtype> class Layer {
  public:
    virtual Tensor<dtype> _forward(Tensor<dtype> input) = 0;
    virtual ~Layer() = default;
};

template <typename dtype> class MaxPool2D : virtual public Layer<dtype> {
  public:
    MaxPool2D(const uint kernel_h, const uint kernel_w, const uint padding_h,
              const uint padding_w, const uint stride_h, const uint stride_w,
              const uint dilation_h, const uint dilation_w,
              const bool ceil_mode, const std::string algorithm)
        : kernel_h(kernel_h), kernel_w(kernel_w), padding_h(padding_h),
          padding_w(padding_w), stride_h(stride_h), stride_w(stride_w),
          dilation_h(dilation_h), dilation_w(dilation_w), ceil_mode(ceil_mode),
          algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (is_default(algorithm)) {
            if constexpr (DEFAULT_TO_CUSTOM_MAXPOOL2D) {
                return maxpool2d<dtype>(
                    std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                    stride_h, stride_w, dilation_h, dilation_w, ceil_mode);
            } else {
                return _maxpool2d<dtype>(
                    std::move(input), kernel_h, kernel_w, padding_h, padding_w,
                    stride_h, stride_w, dilation_h, dilation_w, ceil_mode);
            }
        } else if (is_custom(algorithm)) {
            return maxpool2d<dtype>(std::move(input), kernel_h, kernel_w,
                                    padding_h, padding_w, stride_h, stride_w,
                                    dilation_h, dilation_w, ceil_mode);
        } else if (algorithm == "direct") {
            return _maxpool2d<dtype>(std::move(input), kernel_h, kernel_w,
                                     padding_h, padding_w, stride_h, stride_w,
                                     dilation_h, dilation_w, ceil_mode);
        }

        errs::invalid_algo("maxpool2d", algorithm);
    }
    ~MaxPool2D() = default;

  private:
    const uint kernel_h;
    const uint kernel_w;
    const uint padding_h;
    const uint padding_w;
    const uint stride_h;
    const uint stride_w;
    const uint dilation_h;
    const uint dilation_w;
    const std::string algorithm;
    const bool ceil_mode;
};

template <typename dtype> class AvgPool2D : virtual public Layer<dtype> {
  public:
    AvgPool2D(const uint kernel_h, const uint kernel_w, const uint padding_h,
              const uint padding_w, const uint stride_h, const uint stride_w,
              const bool ceil_mode, const bool count_include_pad,
              const std::optional<uint> divisor_override,
              const std::string algorithm)
        : kernel_h(kernel_h), kernel_w(kernel_w), padding_h(padding_h),
          padding_w(padding_w), stride_h(stride_h), stride_w(stride_w),
          ceil_mode(ceil_mode), count_include_pad(count_include_pad),
          divisor_override(divisor_override), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (is_default(algorithm)) {
            if constexpr (DEFAULT_TO_CUSTOM_AVGPOOL2D) {
                return avgpool2d<dtype>(std::move(input), kernel_h, kernel_w,
                                        padding_h, padding_w, stride_h,
                                        stride_w, ceil_mode, count_include_pad,
                                        divisor_override);
            } else {
                return _avgpool2d<dtype>(std::move(input), kernel_h, kernel_w,
                                         padding_h, padding_w, stride_h,
                                         stride_w, ceil_mode, count_include_pad,
                                         divisor_override);
            }
        } else if (is_custom(algorithm)) {
            return avgpool2d<dtype>(std::move(input), kernel_h, kernel_w,
                                    padding_h, padding_w, stride_h, stride_w,
                                    ceil_mode, count_include_pad,
                                    divisor_override);
        } else if (algorithm == "direct") {
            return _avgpool2d<dtype>(std::move(input), kernel_h, kernel_w,
                                     padding_h, padding_w, stride_h, stride_w,
                                     ceil_mode, count_include_pad,
                                     divisor_override);
        }
        errs::invalid_algo("avgpool2d", algorithm);
    }
    ~AvgPool2D() = default;

  private:
    const uint kernel_h;
    const uint kernel_w;
    const uint padding_h;
    const uint padding_w;
    const uint stride_h;
    const uint stride_w;
    const bool ceil_mode;
    const bool count_include_pad;
    const std::optional<uint> divisor_override;
    const std::string algorithm;
};

template <typename dtype>
class AdaptiveAvgPool2D : virtual public Layer<dtype> {
  public:
    AdaptiveAvgPool2D(const std::optional<uint> output_h,
                      const std::optional<uint> output_w,
                      const std::string algorithm)
        : output_h(output_h), output_w(output_w), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (is_default(algorithm)) {
            if constexpr (DEFAULT_TO_CUSTOM_ADAPTIVEAVGPOOL2D) {
                return adaptiveavgpool2d(std::move(input), output_h, output_w);
            } else {
                return _adaptiveavgpool2d(std::move(input), output_h, output_w);
            }
        } else if (is_custom(algorithm)) {
            return adaptiveavgpool2d(std::move(input), output_h, output_w);
        } else if (algorithm == "direct") {
            return _adaptiveavgpool2d(std::move(input), output_h, output_w);
        }
        errs::invalid_algo("adaptiveavgpool2d", algorithm);
    }
    ~AdaptiveAvgPool2D() = default;

  private:
    const std::optional<uint> output_h;
    const std::optional<uint> output_w;
    const std::string algorithm;
};

template <typename dtype> class ReLU : virtual public Layer<dtype> {
  public:
    ReLU(const std::string algorithm) : algorithm(algorithm){};

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (is_default(algorithm)) {
            if constexpr (DEFAULT_TO_CUSTOM_RELU) {
                return relu<dtype>(std::move(input));
            } else {
                return _relu<dtype>(std::move(input));
            }
        } else if (is_custom(algorithm)) {
            return relu<dtype>(std::move(input));
        } else if (algorithm == "direct") {
            return _relu(std::move(input));
        }

        errs::invalid_algo("relu", algorithm);
    }
    ~ReLU() = default;

  private:
    const std::string algorithm;
};

template <typename dtype> class Linear : virtual public Layer<dtype> {
  public:
    Linear(const intptr_t weight_address, const std::vector<uint> weight_shape,
           const std::optional<intptr_t> bias_addr, const std::string algorithm)
        : weight(weight_address, weight_shape),
          bias(Tensor<dtype>::from_optional(bias_addr, {weight_shape[0]})),
          algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (is_default(algorithm)) {
            if constexpr (DEFAULT_TO_CUSTOM_LINEAR) {
                return linear<dtype>(std::move(input), weight, bias);
            } else {
                return _linear<dtype>(std::move(input), weight, bias, ctx);
            }
        } else if (is_custom(algorithm)) {
            return linear<dtype>(std::move(input), weight, bias);
        } else if (algorithm == "gemm") {
            return _linear<dtype>(std::move(input), weight, bias, ctx);
        }

        errs::invalid_algo("linear", algorithm);
    }

    ~Linear() = default;

  private:
    static Context ctx;
    const Tensor<dtype> weight;
    const std::optional<const Tensor<dtype>> bias;
    const std::string algorithm;
};
template <typename dtype> Context Linear<dtype>::ctx = Context();

template <typename dtype> class Flatten : virtual public Layer<dtype> {
  public:
    Flatten(const uint start_dim, const int end_dim,
            const std::string algorithm)
        : start_dim(start_dim), end_dim(end_dim), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (is_default(algorithm)) {
            if constexpr (DEFAULT_TO_CUSTOM_FLATTEN) {
                return flatten<dtype>(std::move(input), start_dim, end_dim);
            } else {
                return _flatten<dtype>(std::move(input), start_dim, end_dim);
            }
        } else if (is_custom(algorithm)) {
            return flatten<dtype>(std::move(input), start_dim, end_dim);
        } else if (algorithm == "direct") {
            return _flatten<dtype>(std::move(input), start_dim, end_dim);
        }

        errs::invalid_algo("flatten", algorithm);
    }
    ~Flatten() = default;

  private:
    const uint start_dim;
    const int end_dim;
    const std::string algorithm;
};

template <typename dtype> class Conv2D : virtual public Layer<dtype> {
  public:
    Conv2D(const intptr_t weight_address, const std::vector<uint> weight_shape,
           const std::optional<intptr_t> bias_addr, const uint padding_h,
           const uint padding_w, const uint stride_h, const uint stride_w,
           const uint dilation_h, const uint dilation_w,
           const PaddingMode padding_mode, const uint groups,
           const std::string algorithm, bool own_params = true)
        : weight(own_params ? Tensor<dtype>(weight_address, weight_shape)
                            : Tensor<dtype>::form_tensor(weight_address,
                                                         weight_shape)),
          bias(Tensor<dtype>::from_optional(bias_addr, {weight_shape[0]},
                                            own_params)),
          padding_h(padding_h), padding_w(padding_w), stride_h(stride_h),
          stride_w(stride_w), dilation_h(dilation_h), dilation_w(dilation_w),
          padding_mode(padding_mode), groups(groups), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (is_default(algorithm)) {
            if constexpr (DEFAULT_TO_CUSTOM_CONV2D) {
                return conv2d<dtype>(std::move(input), weight, bias, padding_h,
                                     padding_w, stride_h, stride_w, dilation_h,
                                     dilation_w, padding_mode, groups);
            } else if constexpr (USING_CUDA_TOOLS) {
                return implicit_precomp_gemm_conv2d<dtype>(
                    std::move(input), weight, bias, padding_h, padding_w,
                    stride_h, stride_w, dilation_h, dilation_w, padding_mode,
                    groups, ctx);
            } else if constexpr (USING_MPS) {
                return metal_conv2d(std::move(input), weight, bias, padding_h,
                                    padding_w, stride_h, stride_w, dilation_h,
                                    dilation_w, padding_mode, groups);
            } else {
                return direct_conv2d<dtype>(std::move(input), weight, bias,
                                            padding_h, padding_w, stride_h,
                                            stride_w, dilation_h, dilation_w,
                                            padding_mode, groups);
            }
        } else if (is_custom(algorithm)) {
            return conv2d<dtype>(std::move(input), weight, bias, padding_h,
                                 padding_w, stride_h, stride_w, dilation_h,
                                 dilation_w, padding_mode, groups);
        } else if (algorithm == "metal") {
            return metal_conv2d(std::move(input), weight, bias, padding_h,
                                padding_w, stride_h, stride_w, dilation_h,
                                dilation_w, padding_mode, groups);
        } else if (algorithm == "direct") {
            return direct_conv2d<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups);
        } else if (algorithm == "smm") {
            return smm_conv2d<dtype>(std::move(input), weight, bias, padding_h,
                                     padding_w, stride_h, stride_w, dilation_h,
                                     dilation_w, padding_mode, groups);
        } else if (algorithm == "implicit precomp gemm") {
            return implicit_precomp_gemm_conv2d<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups, ctx);
        } else if (algorithm == "implicit gemm") {
            return implicit_gemm_conv2d<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups, ctx);
        } else if (algorithm == "gemm") {
            return gemm_conv2d<dtype>(std::move(input), weight, bias, padding_h,
                                      padding_w, stride_h, stride_w, dilation_h,
                                      dilation_w, padding_mode, groups, ctx);
        } else if (algorithm == "winograd") {
            return winograd_conv2d<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups, ctx);
        } else if (algorithm == "guess") {
            return guess_conv2d<dtype>(
                std::move(input), weight, bias, padding_h, padding_w, stride_h,
                stride_w, dilation_h, dilation_w, padding_mode, groups, ctx);
        }
        errs::invalid_algo("conv2d", algorithm);
    }

    Tensor<dtype> forward(const intptr_t input_address,
                          std::vector<uint> input_shape) {
        return _forward(Tensor<dtype>::form_tensor(input_address, input_shape));
    }

    ~Conv2D() = default;

  private:
    static Context ctx;
    const Tensor<dtype> weight;
    const std::optional<const Tensor<dtype>> bias;
    const uint padding_h;
    const uint padding_w;
    const uint stride_h;
    const uint stride_w;
    const uint dilation_h;
    const uint dilation_w;
    const PaddingMode padding_mode;
    const uint groups;
    const std::string algorithm;
};

template <typename dtype> Context Conv2D<dtype>::ctx = Context();

template <typename dtype>
Tensor<dtype> conv2d_with_algo(
    const intptr_t input_address, const std::vector<uint> input_shape,
    const intptr_t weight_address, const std::vector<uint> weight_shape,
    const std::optional<intptr_t> bias_addr, const uint padding_h,
    const uint padding_w, const uint stride_h, const uint stride_w,
    const uint dilation_h, const uint dilation_w, const uint padding_mode_uint,
    const uint groups, const std::string algorithm) {
    PaddingMode padding_mode = static_cast<PaddingMode>(padding_mode_uint);
    Conv2D<dtype> layer(weight_address, weight_shape, bias_addr, padding_h,
                        padding_w, stride_h, stride_w, dilation_h, dilation_w,
                        padding_mode, groups, algorithm, false);

    return layer.forward(input_address, input_shape);
}

template <typename dtype> class Model {
  public:
    Model(const std::vector<std::shared_ptr<Layer<dtype>>> layers)
        : layers(layers) {}

    Tensor<dtype> predict(const intptr_t input_address,
                          std::vector<uint> input_shape) {
        Tensor<dtype> output =
            Tensor<dtype>::form_tensor(input_address, input_shape);
        for (const std::shared_ptr<Layer<dtype>> &layer : layers) {
            output = layer->_forward(std::move(output));
        }
        return output;
    }

    ~Model() = default;

  private:
    const std::vector<std::shared_ptr<Layer<dtype>>> layers;
};

namespace py = pybind11;

template <typename dtype>
void define_layer_classes(py::module &m, std::string type_str) {
    py::class_<Tensor<dtype>>(m, ("Tensor_" + type_str).c_str(),
                              pybind11::buffer_protocol())
        .def_readonly("shape", &Tensor<dtype>::shape)
        .def_buffer(&Tensor<dtype>::buffer);

    py::class_<Model<dtype>>(m, ("Model_" + type_str).c_str())
        .def(py::init<const std::vector<std::shared_ptr<Layer<dtype>>>>())
        .def("predict", &Model<dtype>::predict);

    py::class_<Layer<dtype>, std::shared_ptr<Layer<dtype>>> _(
        m, ("Layer_" + type_str).c_str());

    py::class_<MaxPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<MaxPool2D<dtype>>>(
        m, ("MaxPool2D_" + type_str).c_str())
        .def(py::init<const uint, const uint, const uint, const uint,
                      const uint, const uint, const uint, const uint,
                      const bool, const std::string>());

    py::class_<Conv2D<dtype>, Layer<dtype>, std::shared_ptr<Conv2D<dtype>>>(
        m, ("Conv2D_" + type_str).c_str())
        .def(py::init<const intptr_t, const std::vector<uint>,
                      const std::optional<intptr_t>, const uint, const uint,
                      const uint, const uint, const uint, const uint,
                      const PaddingMode, const uint, const std::string>())
        .def("forward", &Conv2D<dtype>::forward);

    m.def(("conv2d_" + type_str).c_str(), &conv2d_with_algo<dtype>);

    py::class_<Linear<dtype>, Layer<dtype>, std::shared_ptr<Linear<dtype>>>(
        m, ("Linear_" + type_str).c_str())
        .def(py::init<const intptr_t, const std::vector<uint>,
                      const std::optional<intptr_t>, const std::string>());

    py::class_<ReLU<dtype>, Layer<dtype>, std::shared_ptr<ReLU<dtype>>>(
        m, ("ReLU_" + type_str).c_str())
        .def(py::init<const std::string>());

    py::class_<AvgPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<AvgPool2D<dtype>>>(
        m, ("AvgPool2D_" + type_str).c_str())
        .def(py::init<const uint, const uint, const uint, const uint,
                      const uint, const uint, const bool, const bool,
                      const std::optional<uint>, const std::string>());

    py::class_<AdaptiveAvgPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<AdaptiveAvgPool2D<dtype>>>(
        m, ("AdaptiveAvgPool2D_" + type_str).c_str())
        .def(py::init<const std::optional<uint>, const std::optional<uint>,
                      const std::string>());

    py::class_<Flatten<dtype>, Layer<dtype>, std::shared_ptr<Flatten<dtype>>>(
        m, ("Flatten_" + type_str).c_str())
        .def(py::init<const uint, const int, const std::string>());
}

PYBIND11_MODULE(core, m) {
    define_layer_classes<float>(m, "float");
    define_layer_classes<double>(m, "double");
    py::enum_<PaddingMode>(m, "PaddingMode")
        .value("zeros", PaddingMode::Zeros)
        .value("reflect", PaddingMode::Reflect)
        .value("replicate", PaddingMode::Replicate)
        .value("circular", PaddingMode::Circular)
        .export_values();
    m.def("output_hw_for_2d", &output_hw_for_2d_no_ceil);
    m.def("using_metal", [] { return USING_MPS; });
    m.def("using_sycl", [] { return USING_SYCL; });
    m.def("using_cuda_tools", [] { return USING_CUDA_TOOLS; });
}
