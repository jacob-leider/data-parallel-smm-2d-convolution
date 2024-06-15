#include "user_defined.hpp"
#define CONV2D_SYCL
#include "ai3.hpp"
#include ADAPTIVEAVGPOOL2D_PATH
#include AVGPOOL2D_PATH
#include CONV2D_PATH
#include MAXPOOL2D_PATH
#include LINEAR_PATH
#include RELU_PATH
#include FLATTEN_PATH
#include <cstdint>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

template <typename dtype> class Layer {
  public:
    virtual Tensor<dtype> _forward(Tensor<dtype> input) = 0;
    virtual ~Layer() = default;
};

template <typename dtype> class MaxPool2D : virtual public Layer<dtype> {
  public:
    MaxPool2D(const std::vector<int> kernel_shape,
              const std::vector<int> padding, const std::vector<int> stride,
              const std::vector<int> dilation, const bool ceil_mode)
        : kernel_shape(kernel_shape), padding(padding), stride(stride),
          dilation(dilation), ceil_mode(ceil_mode) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        return maxpool2d<dtype>(input, kernel_shape, padding, stride, dilation,
                                ceil_mode);
    }
    ~MaxPool2D() = default;

  private:
    const std::vector<int> kernel_shape;
    const std::vector<int> padding;
    const std::vector<int> stride;
    const std::vector<int> dilation;
    const bool ceil_mode;
};

template <typename dtype> class AvgPool2D : virtual public Layer<dtype> {
  public:
    AvgPool2D(const std::vector<int> kernel_shape,
              const std::vector<int> padding, const std::vector<int> stride,
              const bool ceil_mode, const bool count_include_pad,
              const std::optional<int> divisor_override)
        : kernel_shape(kernel_shape), padding(padding), stride(stride),
          ceil_mode(ceil_mode), count_include_pad(count_include_pad),
          divisor_override(divisor_override) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        return avgpool2d<dtype>(input, kernel_shape, padding, stride, ceil_mode,
                                count_include_pad, divisor_override);
    }
    ~AvgPool2D() = default;

  private:
    const std::vector<int> kernel_shape;
    const std::vector<int> padding;
    const std::vector<int> stride;
    const bool ceil_mode;
    const bool count_include_pad;
    const std::optional<int> divisor_override;
};

template <typename dtype>
class AdaptiveAvgPool2D : virtual public Layer<dtype> {
  public:
    AdaptiveAvgPool2D(
        std::optional<std::vector<std::optional<int>>> output_shape)
        : output_shape(output_shape) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        return adaptiveavgpool2d(input, output_shape);
    }
    ~AdaptiveAvgPool2D() = default;

  private:
    const std::optional<std::vector<std::optional<int>>> output_shape;
};

template <typename dtype> class ReLU : virtual public Layer<dtype> {
  public:
    ReLU(){};

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        return relu<dtype>(std::move(input));
    }
    ~ReLU() = default;
};

template <typename dtype> class Linear : virtual public Layer<dtype> {
  public:
    Linear(const intptr_t weight_address, const std::vector<int> weight_shape,
           const std::optional<intptr_t> bias_addr)
        : weight(weight_address, weight_shape),
          bias(Tensor<dtype>::from_optional(bias_addr, {weight_shape[0]})) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        return linear<dtype>(input, weight, bias);
    }

    ~Linear() = default;

  private:
    const Tensor<dtype> weight;
    const std::optional<const Tensor<dtype>> bias;
};

template <typename dtype> class Flatten : virtual public Layer<dtype> {
  public:
    Flatten(const int start_dim, const int end_dim)
        : start_dim(start_dim), end_dim(end_dim) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        return flatten<dtype>(std::move(input), start_dim, end_dim);
    }
    ~Flatten() = default;

  private:
    const int start_dim;
    const int end_dim;
};

template <typename dtype> class Conv2D : virtual public Layer<dtype> {
  public:
    Conv2D(const intptr_t weight_address, const std::vector<int> weight_shape,
           const std::optional<intptr_t> bias_addr,
           const std::vector<int> padding, const std::vector<int> stride,
           const std::vector<int> dilation, const PaddingMode padding_mode,
           int groups)
        : weight(weight_address, weight_shape),
          bias(Tensor<dtype>::from_optional(bias_addr, {weight_shape[0]})),
          padding(padding), stride(stride), dilation(dilation),
          padding_mode(padding_mode), groups(groups) {}

    Tensor<dtype> _forward(Tensor<dtype> input) override {
        return conv2d<dtype>(input, weight, bias, padding, stride, dilation,
                             padding_mode, groups);
    }

    Tensor<dtype> forward(const intptr_t input_address,
                          std::vector<int> input_shape) {

        return _forward(Tensor<dtype>(input_address, input_shape));
    }

    ~Conv2D() = default;

  private:
    const Tensor<dtype> weight;
    const std::optional<const Tensor<dtype>> bias;
    const std::vector<int> padding;
    const std::vector<int> stride;
    const std::vector<int> dilation;
    const PaddingMode padding_mode;
    const int groups;
};

template <typename dtype> class Model {
  public:
    Model(const std::vector<std::shared_ptr<Layer<dtype>>> layers)
        : layers(layers) {}

    Tensor<dtype> predict(const intptr_t input_address,
                          std::vector<int> input_shape) {
        Tensor<dtype> output(input_address, input_shape, true);
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
        .def(py::init<const std::vector<int>, const std::vector<int>,
                      const std::vector<int>, const std::vector<int>,
                      const bool>());

    py::class_<Conv2D<dtype>, Layer<dtype>, std::shared_ptr<Conv2D<dtype>>>(
        m, ("Conv2D_" + type_str).c_str())
        .def(py::init<const intptr_t, const std::vector<int>,
                      const std::optional<intptr_t>, const std::vector<int>,
                      const std::vector<int>, const std::vector<int>,
                      const PaddingMode, const int>())
        .def("forward", &Conv2D<dtype>::forward);

    py::class_<Linear<dtype>, Layer<dtype>, std::shared_ptr<Linear<dtype>>>(
        m, ("Linear_" + type_str).c_str())
        .def(py::init<const intptr_t, const std::vector<int>,
                      const std::optional<intptr_t>>());

    py::class_<ReLU<dtype>, Layer<dtype>, std::shared_ptr<ReLU<dtype>>>(
        m, ("ReLU_" + type_str).c_str())
        .def(py::init());

    py::class_<AvgPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<AvgPool2D<dtype>>>(
        m, ("AvgPool2D_" + type_str).c_str())
        .def(py::init<const std::vector<int>, const std::vector<int>,
                      const std::vector<int>, const bool, const bool,
                      const std::optional<int>>());

    py::class_<AdaptiveAvgPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<AdaptiveAvgPool2D<dtype>>>(
        m, ("AdaptiveAvgPool2D_" + type_str).c_str())
        .def(py::init<std::optional<std::vector<std::optional<int>>>>());

    py::class_<Flatten<dtype>, Layer<dtype>, std::shared_ptr<Flatten<dtype>>>(
        m, ("Flatten_" + type_str).c_str())
        .def(py::init<const int, const int>());
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
