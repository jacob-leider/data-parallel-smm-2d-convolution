#include "kn2row_plain.hpp"
#include "tensor.hpp"
// TODO don't want this from pytorch want it on the python path
// TODO remove the pytorch path from the .clangd and the setup.py script
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename dtype> class Layer {
  public:
    virtual Tensor<dtype> forward(const Tensor<dtype> &input) = 0;
    virtual ~Layer() = default;

    Tensor<dtype> weight;
    std::optional<Tensor<dtype>> bias;

  protected:
    Layer(const intptr_t weight_address, const std::vector<int> &weight_shape,
          const std::optional<intptr_t> bias_addr)
        : weight(formTensorFrom<dtype>(weight_address, weight_shape)),
          bias(bias_addr.has_value()
                   ? std::make_optional(formTensorFrom<dtype>(
                         bias_addr.value(), {1, weight_shape[0]}))
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

// TODO better to instantiate each layer on its own in python then pass that
// list into this function probably is better but don't make that Layer
// class public to users or maybe do make it public I don't know
template <typename dtype> class Model {
  public:
    Model(const std::vector<std::shared_ptr<Layer<dtype>>> layers)
        : layers(layers) {}

    Tensor<dtype> predict(const intptr_t input_address,
                          std::vector<int> input_shape) {
        Tensor<dtype> output =
            formTensorFrom<dtype>(input_address, input_shape);
        for (const auto &layer : layers) {
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

// TODO want to do all this casting in python so we don't need pytorch
// headers in our cpp
// TODO need to change this types so they work with onnx and torch frontend
// TODO need these to actually be attributes of a C++ class which owns a
// forward function
// TODO reorder this so dtype is either first or last
// at::Tensor kn2row_conv2d_entry(
//     const intptr_t input_address, const std::vector<int> input_shape,
//     const intptr_t kernel_address, const std::vector<int> kernel_shape,
//     const std::optional<intptr_t> bias, const std::vector<int>
//     output_shape, const std::string dtype, const std::vector<int>
//     padding, const std::vector<int> stride, const std::vector<int>
//     dilation) { if (dtype != "torch.float64" && dtype != "torch.float32")
//     {
//         std::cerr << "Unsupported data type." << dtype << std::endl;
//         std::exit(1);
//     }
//
//     const void *input_store = (void *)input_address;
//     const void *kernel_store = (void *)kernel_address;
//
//     if (dtype == "torch.float32") {
//         return kn2row_conv2d(
//             quick_cast<float>(input_store), input_shape,
//             quick_cast<float>(kernel_store), kernel_shape,
//             bias.has_value()
//                 ? std::make_optional<const float *>(
//                       static_cast<const float *>((void *)bias.value()))
//                 : std::nullopt,
//             output_shape, padding, stride, dilation);
//     } else {
//         return kn2row_conv2d(
//             quick_cast<double>(input_store), input_shape,
//             quick_cast<double>(kernel_store), kernel_shape,
//             bias.has_value()
//                 ? std::make_optional<const double *>(
//                       static_cast<const double *>((void *)bias.value()))
//                 : std::nullopt,
//             output_shape, padding, stride, dilation);
//     }
// }
//
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("kn2row_conv2d", &kn2row_conv2d_entry, "2d convolution using
//     kn2row");
// }
