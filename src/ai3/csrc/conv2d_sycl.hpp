#include "ai3.hpp"
#include <CL/sycl.hpp>
#include <optional>
using namespace cl;

template <typename dtype>
Tensor<dtype> conv2d(const Tensor<dtype> &input, const Tensor<dtype> &kernel,
                     const std::optional<const Tensor<dtype>> &bias,
                     const std::vector<int> &padding,
                     const std::vector<int> &stride,
                     const std::vector<int> &dilation,
                     const PaddingMode padding_mode, int groups) {
    errs::bail_if(padding_mode != Zeros, "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

    const int input_channels = dims::input_channels(input.shape);
    const int input_height = dims::height(input.shape);
    const int input_width = dims::width(input.shape);

    const int kernel_height = dims::height(kernel.shape);
    const int kernel_width = dims::width(kernel.shape);

    const int output_channels = dims::out_channels(kernel.shape);

    const int output_height = dims::output_dim<dtype>(
        input_height, kernel_height, padding[0], dilation[0], stride[0], false);
    const int output_width = dims::output_dim<dtype>(
        input_width, kernel_width, padding[1], dilation[1], stride[1], false);

    int num_samples;
    Tensor<dtype> output;
    if (dims::has_dim_for_batch_size(input.shape, dims::input::CONV2D)) {
        num_samples = 1;
        output = Tensor<dtype>({output_channels, output_height, output_width});
    } else {
        num_samples = dims::batch_size(input.shape, dims::input::POOL2D);
        output = Tensor<dtype>(
            {num_samples, output_channels, output_height, output_width});
    }

    int padding_h = padding[0];
    int padding_w = padding[1];
    int stride_h = stride[0];
    int stride_w = stride[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    sycl::queue queue(sycl::default_selector_v);

    sycl::buffer<dtype> input_buf(input.data, input.count());
    sycl::buffer<dtype> kernel_buf(kernel.data, kernel.count());

    const bool has_bias = bias.has_value();
    sycl::buffer<dtype> bias_buf =
        has_bias ? sycl::buffer<dtype>(bias->data, bias->count())
                 : sycl::buffer<dtype>(sycl::range<1>(0));
    sycl::buffer<dtype> output_buf(output.data, output.count());

    const int total_out = output.count();
    const int max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();

    int work_group;
    if (total_out < max_work_group_size) {
        work_group = total_out;
    } else {
        work_group = max_work_group_size;
        while (total_out % work_group != 0) {
            work_group--;
        }
    }

    const int num_batch = total_out / work_group;

    queue
        .submit([&](sycl::handler &h) {
            sycl::accessor ainput =
                sycl::accessor(input_buf, h, sycl::read_only);
            sycl::accessor akernel =
                sycl::accessor(kernel_buf, h, sycl::read_only);
            sycl::accessor abias = sycl::accessor(bias_buf, h, sycl::read_only);
            sycl::accessor aoutput =
                sycl::accessor(output_buf, h, sycl::write_only, sycl::no_init);
            h.parallel_for(
                sycl::nd_range(sycl::range(total_out),
                               sycl::range(total_out / num_batch)),
                [=](sycl::nd_item<1> item) {
                    int glob_id = item.get_global_id(0);

                    int idx = glob_id;
                    int out_w = idx % output_width;
                    idx /= output_width;
                    int out_h = idx % output_height;
                    idx /= output_height;
                    int out_c = idx % output_channels;
                    idx /= output_channels;
                    int samp = idx;
                    dtype sum = 0;
                    int h_offset_base = out_h * stride_h - padding_h;
                    int w_offset_base = out_w * stride_w - padding_w;
                    for (int in_c = 0; in_c < input_channels; ++in_c) {
                        for (int ker_h = 0; ker_h < kernel_height; ++ker_h) {
                            for (int ker_w = 0; ker_w < kernel_width; ++ker_w) {
                                int h_offset =
                                    h_offset_base + ker_h * dilation_h;
                                int w_offset =
                                    w_offset_base + ker_w * dilation_w;
                                if (h_offset >= 0 && h_offset < input_height &&
                                    w_offset >= 0 && w_offset < input_width) {
                                    sum += ainput[to_linear(
                                               samp, in_c, h_offset, w_offset,
                                               input_channels, input_height,
                                               input_width)] *
                                           akernel[to_linear(
                                               out_c, in_c, ker_h, ker_w,
                                               input_channels, kernel_height,
                                               kernel_width)];
                                }
                            }
                        }
                    }
                    if (has_bias) {
                        sum += abias[out_c];
                    }
                    aoutput[glob_id] = sum;
                });
        })
        .wait_and_throw();
    return output;
}
