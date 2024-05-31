// TODO support for the first index of input and output tensors being greater
// than one signifying multiple inputs and outputs (predicting on a list)
// this should be done for conv2d, maxpool, linear
#include "avgpool2d_plain.hpp"
#include "flatten_plain.hpp"
#include "kn2rowconv2d_plain.hpp"
#include "linear_plain.hpp"
#include "maxpool2d_plain.hpp"
#include "relu_plain.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename dtype> class Layer {
  public:
    virtual Tensor<dtype> forward(const Tensor<dtype> &input);
    virtual ~Layer() = default;
};

template <typename dtype> class MaxPool2D : virtual public Layer<dtype> {
  public:
    MaxPool2D(const std::vector<int> kernel_shape,
              const std::vector<int> padding, const std::vector<int> stride,
              const std::vector<int> dilation, const bool ceil_mode)
        : kernel_shape(kernel_shape), padding(padding), stride(stride),
          dilation(dilation), ceil_mode(ceil_mode) {}

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return maxpool2d<dtype>(input, this->kernel_shape, this->padding,
                                this->stride, this->dilation, this->ceil_mode);
    }
    ~MaxPool2D() = default;

  private:
    std::vector<int> kernel_shape;
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    bool ceil_mode;
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

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return avgpool2d<dtype>(
            input, this->kernel_shape, this->padding, this->stride,
            this->ceil_mode, this->count_include_pad, this->divisor_override);
    }
    ~AvgPool2D() = default;

  private:
    std::vector<int> kernel_shape;
    std::vector<int> padding;
    std::vector<int> stride;
    bool ceil_mode;
    bool count_include_pad;
    std::optional<int> divisor_override;
};

template <typename dtype>
class AdaptiveAvgPool2D : virtual public Layer<dtype> {
  public:
    AdaptiveAvgPool2D(
        std::optional<std::vector<std::optional<int>>> output_shape)
        : output_shape(output_shape) {}

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        int input_height = dims::height(input.shape);
        int input_width = dims::width(input.shape);
        const std::vector<std::optional<int>> opt_in_shape = {input_height,
                                                              input_width};
        const int output_height =
            this->output_shape.value_or(opt_in_shape)[0].value_or(input_height);
        const int output_width =
            this->output_shape.value_or(opt_in_shape)[1].value_or(input_width);
        bail_if(
            input_height % output_height != 0 ||
                input_width % output_width != 0,
            "Adaptive average pooling not implemented for cases where "
            "input size is not a multiple of output size given input shape=(",
            input_height, ", ", input_width, ") and output shape=(",
            output_height, ", ", output_width, ")");
        std::vector<int> stride = {input_height / output_height,
                                   input_width / output_width};
        std::vector<int> kernel_shape = {
            input_height - ((output_height - 1) * stride[0]),
            input_width - ((output_width - 1) * stride[1])};
        return avgpool2d<dtype>(input, kernel_shape, {0, 0}, stride, false,
                                false, std::nullopt);
    }
    ~AdaptiveAvgPool2D() = default;

  private:
    std::optional<std::vector<std::optional<int>>> output_shape;
};

template <typename dtype> class ReLU : virtual public Layer<dtype> {
  public:
    ReLU(){};

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return relu<dtype>(input);
    }
    ~ReLU() = default;
};

template <typename dtype> class Linear : virtual public Layer<dtype> {
  public:
    Linear(const intptr_t weight_address, const std::vector<int> weight_shape,
           const std::optional<intptr_t> bias_addr)
        : weight(weight_address, weight_shape),
          bias(Tensor<dtype>::from_optional(bias_addr, {weight_shape[0]})) {}

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return linear<dtype>(input, this->weight, this->bias);
    }
    ~Linear() = default;

  private:
    Tensor<dtype> weight;
    std::optional<Tensor<dtype>> bias;
};

template <typename dtype> class Flatten : virtual public Layer<dtype> {
  public:
    Flatten(const int start_dim, const int end_dim)
        : start_dim(start_dim), end_dim(end_dim) {}

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return flatten<dtype>(input, this->start_dim, this->end_dim);
    }
    ~Flatten() = default;

  private:
    int start_dim;
    int end_dim;
};

// TODO do groups
template <typename dtype> class Conv2D : virtual public Layer<dtype> {
  public:
    Conv2D(const intptr_t weight_address, const std::vector<int> weight_shape,
           const std::optional<intptr_t> bias_addr,
           const std::vector<int> padding, const std::vector<int> stride,
           const std::vector<int> dilation)
        : weight(weight_address, weight_shape),
          bias(Tensor<dtype>::from_optional(bias_addr, {weight_shape[0]})),
          padding(padding), stride(stride), dilation(dilation) {}

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return kn2row_conv2d<dtype>(input, this->weight, this->bias,
                                    this->padding, this->stride,
                                    this->dilation);
    }
    ~Conv2D() = default;

  private:
    Tensor<dtype> weight;
    std::optional<Tensor<dtype>> bias;
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
};

template <typename dtype> class Model {
  public:
    Model(const std::vector<std::shared_ptr<Layer<dtype>>> layers)
        : layers(layers) {}

    Tensor<dtype> predict(const intptr_t input_address,
                          std::vector<int> input_shape) {
        Tensor<dtype> output(input_address, input_shape);
        for (const std::shared_ptr<Layer<dtype>> &layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

  private:
    std::vector<std::shared_ptr<Layer<dtype>>> layers;
};

namespace py = pybind11;

template <typename dtype>
void define_layer_classes(py::module &m, std::string type_str) {
    py::class_<Tensor<dtype>>(m, ("Tensor_" + type_str).c_str())
        .def_readonly("data", &Tensor<dtype>::data)
        .def_readonly("shape", &Tensor<dtype>::shape);

    py::class_<Model<dtype>>(m, ("Model_" + type_str).c_str())
        .def(py::init<const std::vector<std::shared_ptr<Layer<dtype>>>>())
        .def("predict", &Model<dtype>::predict);

    py::class_<Layer<dtype>, std::shared_ptr<Layer<dtype>>>(
        m, ("Layer_" + type_str).c_str())
        .def("forward", &Layer<dtype>::forward);

    py::class_<Conv2D<dtype>, Layer<dtype>, std::shared_ptr<Conv2D<dtype>>>(
        m, ("Conv2D_" + type_str).c_str())
        .def(py::init<const intptr_t, const std::vector<int>,
                      const std::optional<intptr_t>, const std::vector<int>,
                      const std::vector<int>, const std::vector<int>>());

    py::class_<MaxPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<MaxPool2D<dtype>>>(
        m, ("MaxPool2D_" + type_str).c_str())
        .def(py::init<const std::vector<int>, const std::vector<int>,
                      const std::vector<int>, const std::vector<int>,
                      const bool>());

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
}
