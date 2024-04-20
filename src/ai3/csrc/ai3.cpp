// TODO support for the first index of input and output tensors being greater
// than one signifying multiple inputs and outputs (predicting on a list)
#include "kn2row_plain.hpp"
#include "maxpool2d_plain.hpp"
#include "tensor.hpp"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename dtype> class Layer {
  public:
    virtual Tensor<dtype> forward(const Tensor<dtype> &input) = 0;
};

template <typename dtype> class MaxPool2D : public Layer<dtype> {
  public:
    MaxPool2D(const std::vector<int> kernel_shape,
              const std::vector<int> padding, const std::vector<int> stride,
              const std::vector<int> dilation)
        : kernel_shape(kernel_shape), padding(padding), stride(stride),
          dilation(dilation) {}

    Tensor<dtype> forward(const Tensor<dtype> &input) override {
        return maxpool2d<dtype>(input, this->kernel_shape, this->padding,
                                this->stride, this->dilation);
    }
    ~MaxPool2D() = default;

  private:
    std::vector<int> kernel_shape;
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
};

// TODO do groups
template <typename dtype> class Conv2D : public Layer<dtype> {
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

    // TODO simplify this naming, don't know if the third thing is necessary
    std::string conv2d_name = std::string("Conv2D_") + type_str;
    py::class_<Conv2D<dtype>, Layer<dtype>, std::shared_ptr<Conv2D<dtype>>>
        conv2d(m, conv2d_name.c_str());
    conv2d.def(py::init<const intptr_t, const std::vector<int>,
                        const std::optional<intptr_t>, const std::vector<int>,
                        const std::vector<int>, const std::vector<int>>());

    std::string maxpool_2d_name = std::string("MaxPool2D_") + type_str;
    py::class_<MaxPool2D<dtype>, Layer<dtype>,
               std::shared_ptr<MaxPool2D<dtype>>>
        maxpool2d(m, maxpool_2d_name.c_str());
    maxpool2d.def(py::init<const std::vector<int>, const std::vector<int>,
                           const std::vector<int>, const std::vector<int>>());
}

PYBIND11_MODULE(core, m) {
    define_layer_classes<float>(m, "float");
    define_layer_classes<double>(m, "double");
}
