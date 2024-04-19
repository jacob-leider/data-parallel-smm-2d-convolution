// TODO support for the first index of input and output tensors being greater
// than one signifying multiple inputs and outputs (predicting on a list)
#include "kn2row_plain.hpp"
#include "tensor.hpp"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename dtype> class Layer {
  public:
    virtual Tensor<dtype> forward(const Tensor<dtype> &input) = 0;

    Tensor<dtype> weight;
    std::optional<Tensor<dtype>> bias;

  protected:
    Layer(const intptr_t weight_address, const std::vector<int> &weight_shape,
          const std::optional<intptr_t> bias_addr)
        : weight(Tensor<dtype>(weight_address, weight_shape)),
          bias(bias_addr.has_value()
                   ? std::make_optional(
                         Tensor<dtype>(bias_addr.value(), {weight_shape[0]}))
                   : std::nullopt) {}
};

// TODO do groups
template <typename dtype> class Conv2D : public Layer<dtype> {
  public:
    Conv2D(const intptr_t weight_address, const std::vector<int> weight_shape,
           const std::optional<intptr_t> bias_addr,
           const std::vector<int> padding, const std::vector<int> stride,
           const std::vector<int> dilation)
        : Layer<dtype>(weight_address, weight_shape, bias_addr),
          padding(padding), stride(stride), dilation(dilation) {}

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return kn2row_conv2d<dtype>(input, this->weight, this->bias,
                                    this->padding, this->stride,
                                    this->dilation);
    }
    ~Conv2D() = default;

  private:
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
        for (std::shared_ptr<Layer<dtype>> &layer : layers) {
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
    std::string tensor_name = std::string("Tensor_") + type_str;
    py::class_<Tensor<dtype>>(m, tensor_name.c_str())
        .def_readonly("data", &Tensor<dtype>::data)
        .def_readonly("shape", &Tensor<dtype>::shape);

    std::string layer_name = std::string("Layer_") + type_str;
    py::class_<Layer<dtype>, std::shared_ptr<Layer<dtype>>> layer(
        m, layer_name.c_str());

    std::string model_name = std::string("Model_") + type_str;
    py::class_<Model<dtype>> model(m, model_name.c_str());
    model.def(py::init<const std::vector<std::shared_ptr<Layer<dtype>>> &>())
        .def("predict", &Model<dtype>::predict);

    std::string conv2d_name = std::string("Conv2D_") + type_str;
    py::class_<Conv2D<dtype>, Layer<dtype>, std::shared_ptr<Conv2D<dtype>>>
        conv2d(m, conv2d_name.c_str());
    conv2d.def(py::init<const intptr_t, const std::vector<int>,
                        const std::optional<intptr_t>, const std::vector<int>,
                        const std::vector<int>, const std::vector<int>>());
}

PYBIND11_MODULE(core, m) {
    define_layer_classes<float>(m, "float");
    define_layer_classes<double>(m, "double");
}
