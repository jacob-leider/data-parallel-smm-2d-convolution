#define CONV2D_SYCL
#include "ai3.hpp"
#include "algo_paths.hpp"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#define DEFAULT(algo) algo == "default"
#define AI3(algo) algo == "ai3"
#define USER(algo) algo == "user"

template <typename dtype> class Layer {
  public:
    virtual Tensor<dtype> _forward(Tensor<dtype> input) = 0;
    virtual ~Layer() = default;
};

template <typename dtype> class MaxPool2D : virtual public Layer<dtype> {
  public:
    MaxPool2D(const std::vector<uint> kernel_shape,
              const std::vector<uint> padding, const std::vector<uint> stride,
              const std::vector<uint> dilation, const bool ceil_mode,
              const std::string algorithm)
        : kernel_shape(kernel_shape), padding(padding), stride(stride),
          dilation(dilation), ceil_mode(ceil_mode), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (DEFAULT(algorithm)) {
#ifdef MAXPOOL2D_USER
            return maxpool2d<dtype>(input, kernel_shape, padding, stride,
                                    dilation, ceil_mode);
#else
            return _maxpool2d<dtype>(input, kernel_shape, padding, stride,
                                     dilation, ceil_mode);
#endif
        } else if (USER(algorithm)) {
#ifdef LINEAR_USER
            return maxpool2d<dtype>(input, kernel_shape, padding, stride,
                                    dilation, ceil_mode);
#else
            errs::bail(
                "trying to use user maxpool2d when no implementation exists");
#endif
        } else if (AI3(algorithm)) {
            return _maxpool2d<dtype>(input, kernel_shape, padding, stride,
                                     dilation, ceil_mode);
        }
        errs::bail("invalid maxpool2d algorithm: ", algorithm);
    }
    ~MaxPool2D() = default;

  private:
    const std::vector<uint> kernel_shape;
    const std::vector<uint> padding;
    const std::vector<uint> stride;
    const std::vector<uint> dilation;
    const std::string algorithm;
    const bool ceil_mode;
};

template <typename dtype> class AvgPool2D : virtual public Layer<dtype> {
  public:
    AvgPool2D(const std::vector<uint> kernel_shape,
              const std::vector<uint> padding, const std::vector<uint> stride,
              const bool ceil_mode, const bool count_include_pad,
              const std::optional<uint> divisor_override,
              const std::string algorithm)
        : kernel_shape(kernel_shape), padding(padding), stride(stride),
          ceil_mode(ceil_mode), count_include_pad(count_include_pad),
          divisor_override(divisor_override), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (DEFAULT(algorithm)) {
#ifdef AVGPOOL2D_USER
            return avgpool2d<dtype>(input, kernel_shape, padding, stride,
                                    ceil_mode, count_include_pad,
                                    divisor_override);
#else
            return _avgpool2d<dtype>(input, kernel_shape, padding, stride,
                                     ceil_mode, count_include_pad,
                                     divisor_override);
#endif
        } else if (USER(algorithm)) {
#ifdef AVGPOOL2D_USER
            return avgpool2d<dtype>(input, kernel_shape, padding, stride,
                                    ceil_mode, count_include_pad,
                                    divisor_override);
#else
            errs::bail("trying to use user avgpool2d when no "
                       "implementation exists");
#endif
        } else if (AI3(algorithm)) {
            return _avgpool2d<dtype>(input, kernel_shape, padding, stride,
                                     ceil_mode, count_include_pad,
                                     divisor_override);
        }
        errs::bail("invalid avgpool2d algorithm: ", algorithm);
    }
    ~AvgPool2D() = default;

  private:
    const std::vector<uint> kernel_shape;
    const std::vector<uint> padding;
    const std::vector<uint> stride;
    const bool ceil_mode;
    const bool count_include_pad;
    const std::optional<uint> divisor_override;
    const std::string algorithm;
};

template <typename dtype>
class AdaptiveAvgPool2D : virtual public Layer<dtype> {
  public:
    AdaptiveAvgPool2D(
        std::optional<std::vector<std::optional<uint>>> output_shape,
        const std::string algorithm)
        : output_shape(output_shape), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (DEFAULT(algorithm)) {
#ifdef AVGPOOL2D_USER
            return adaptiveavgpool2d(input, output_shape);
#else
            return _adaptiveavgpool2d(input, output_shape);
#endif
        } else if (USER(algorithm)) {
#ifdef AVGPOOL2D_USER
            return adaptiveavgpool2d(input, output_shape);
#else
            errs::bail("trying to use user adaptiveavgpool2d when no "
                       "implementation exists");
#endif
        } else if (AI3(algorithm)) {
            return _adaptiveavgpool2d(input, output_shape);
        }
        errs::bail("invalid adaptiveavgpool2d algorithm: ", algorithm);
    }
    ~AdaptiveAvgPool2D() = default;

  private:
    const std::optional<std::vector<std::optional<uint>>> output_shape;
    const std::string algorithm;
};

template <typename dtype> class ReLU : virtual public Layer<dtype> {
  public:
    ReLU(const std::string algorithm) : algorithm(algorithm){};

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (DEFAULT(algorithm)) {
#ifdef LINEAR_USER
            return relu<dtype>(std::move(input));
#else
            return _relu<dtype>(std::move(input));
#endif
        } else if (USER(algorithm)) {
#ifdef LINEAR_USER
            return relu<dtype>(std::move(input));
#else
            errs::bail(
                "trying to use user flatten when no implementation exists");
#endif
        } else if (AI3(algorithm)) {
            return _relu<dtype>(std::move(input));
        }
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
        if (DEFAULT(algorithm)) {
#ifdef LINEAR_USER
            return linear<dtype>(input, weight, bias);
#else
            return _linear<dtype>(input, weight, bias);
#endif
        } else if (USER(algorithm)) {
#ifdef LINEAR_USER
            return linear<dtype>(input, weight, bias);
#else
            errs::bail(
                "trying to use user flatten when no implementation exists");
#endif
        } else if (AI3(algorithm)) {
            return _linear<dtype>(input, weight, bias);
        }
        errs::bail("invalid linear algorithm: ", algorithm);
    }

    ~Linear() = default;

  private:
    const Tensor<dtype> weight;
    const std::optional<const Tensor<dtype>> bias;
    const std::string algorithm;
};

template <typename dtype> class Flatten : virtual public Layer<dtype> {
  public:
    Flatten(const uint start_dim, const int end_dim,
            const std::string algorithm)
        : start_dim(start_dim), end_dim(end_dim), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (DEFAULT(algorithm)) {
#ifdef FLATTEN_USER
            return flatten<dtype>(std::move(input), start_dim, end_dim);
#else
            return _flatten<dtype>(std::move(input), start_dim, end_dim);
#endif
        } else if (USER(algorithm)) {
#ifdef FLATTEN_USER
            return flatten<dtype>(std::move(input), start_dim, end_dim);
#else
            errs::bail(
                "trying to use user linear when no implementation exists");
#endif
        } else if (AI3(algorithm)) {
            return _flatten<dtype>(std::move(input), start_dim, end_dim);
        }
        errs::bail("invalid flatten algorithm: ", algorithm);
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
           const std::optional<intptr_t> bias_addr,
           const std::vector<uint> padding, const std::vector<uint> stride,
           const std::vector<uint> dilation, const PaddingMode padding_mode,
           const uint groups, const std::string algorithm)
        : weight(weight_address, weight_shape),
          bias(Tensor<dtype>::from_optional(bias_addr, {weight_shape[0]})),
          padding(padding), stride(stride), dilation(dilation),
          padding_mode(padding_mode), groups(groups), algorithm(algorithm) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        if (DEFAULT(algorithm)) {
#ifdef CONV2D_USER
            return conv2d<dtype>(input, weight, bias, padding, stride, dilation,
                                 padding_mode, groups);
#else
            return direct_conv2d<dtype>(input, weight, bias, padding, stride,
                                        dilation, padding_mode, groups);
#endif
        } else if (USER(algorithm)) {
#ifdef CONV2D_USER
            return conv2d<dtype>(input, weight, bias, padding, stride, dilation,
                                 padding_mode, groups);
#else
            errs::bail(
                "trying to use user conv2d when no implementation exists");
#endif
        } else if (algorithm == "direct" || AI3(algorithm)) {
            return direct_conv2d<dtype>(input, weight, bias, padding, stride,
                                        dilation, padding_mode, groups);
        } else if (algorithm == "smm") {
            return smm_conv2d<dtype>(input, weight, bias, padding, stride,
                                     dilation, padding_mode, groups);
        }
        errs::bail("invalid convolution algorithm: ", algorithm);
    }

    Tensor<dtype> forward(const intptr_t input_address,
                          std::vector<uint> input_shape) {

        return _forward(Tensor<dtype>(input_address, input_shape));
    }

    ~Conv2D() = default;

  private:
    const Tensor<dtype> weight;
    const std::optional<const Tensor<dtype>> bias;
    const std::vector<uint> padding;
    const std::vector<uint> stride;
    const std::vector<uint> dilation;
    const PaddingMode padding_mode;
    const uint groups;
    const std::string algorithm;
};

template <typename dtype> class Model {
  public:
    Model(const std::vector<std::shared_ptr<Layer<dtype>>> layers)
        : layers(layers) {}

    Tensor<dtype> predict(const intptr_t input_address,
                          std::vector<uint> input_shape) {
        Tensor<dtype> output(input_address, input_shape, false);
        for (const std::shared_ptr<Layer<dtype>> &layer : layers) {
            output = layer->_forward(std::move(output));
        }
        return output;
    }

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
        .def(py::init<const std::vector<uint>, const std::vector<uint>,
                      const std::vector<uint>, const std::vector<uint>,
                      const bool, const std::string>());

    py::class_<Conv2D<dtype>, Layer<dtype>, std::shared_ptr<Conv2D<dtype>>>(
        m, ("Conv2D_" + type_str).c_str())
        .def(py::init<const intptr_t, const std::vector<uint>,
                      const std::optional<intptr_t>, const std::vector<uint>,
                      const std::vector<uint>, const std::vector<uint>,
                      const PaddingMode, const uint, const std::string>())
        .def("forward", &Conv2D<dtype>::forward);

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
        .def(py::init<const std::vector<uint>, const std::vector<uint>,
                      const std::vector<uint>, const bool, const bool,
                      const std::optional<uint>, const std::string>());

    py::class_<AdaptiveAvgPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<AdaptiveAvgPool2D<dtype>>>(
        m, ("AdaptiveAvgPool2D_" + type_str).c_str())
        .def(py::init<std::optional<std::vector<std::optional<uint>>>,
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
}
