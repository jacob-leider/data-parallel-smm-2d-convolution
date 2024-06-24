#pragma once

#include "ai3.hpp"
#include "utils.hpp"
#include <CL/sycl.hpp>
#include <optional>
#include <vector>
using namespace cl;

// TODO groups and padding modes
template <typename dtype>
Tensor<dtype>
smm_conv2d(const Tensor<dtype> &input, const Tensor<dtype> &kernel,
           const std::optional<const Tensor<dtype>> &bias,
           const std::vector<uint> &padding, const std::vector<uint> &stride,
           const std::vector<uint> &dilation, const PaddingMode padding_mode,
           int groups) {
    errs::bail_if(padding_mode != Zeros, "padding mode must be zeros");
    errs::bail_if(groups != 1, "groups must be 1");

    const uint input_channels = dims::input_channels(input.shape);
    const uint input_height = dims::height(input.shape);
    const uint input_width = dims::width(input.shape);

    const uint kernel_height = dims::height(kernel.shape);
    const uint kernel_width = dims::width(kernel.shape);

    const uint output_channels = dims::out_channels(kernel.shape);

    const uint output_height = dims::output_dim<dtype>(
        input_height, kernel_height, padding[0], dilation[0], stride[0], false);
    const uint output_width = dims::output_dim<dtype>(
        input_width, kernel_width, padding[1], dilation[1], stride[1], false);

    uint num_samples;
    Tensor<dtype> output;
    if (dims::has_dim_for_batch_size(input.shape, dims::input::CONV2D)) {
        num_samples = 1;
        output = Tensor<dtype>({output_channels, output_height, output_width});
    } else {
        num_samples = dims::batch_size(input.shape, dims::input::POOL2D);
        output = Tensor<dtype>(
            {num_samples, output_channels, output_height, output_width});
    }

    uint padding_h = padding[0];
    uint padding_w = padding[1];
    uint stride_h = stride[0];
    uint stride_w = stride[1];
    uint dilation_h = dilation[0];
    uint dilation_w = dilation[1];

    uint col_height = input_channels * kernel_height * kernel_width;
    uint col_width = output_height * output_width;
    Tensor<dtype> col({num_samples, col_height, col_width});
    // TODO can use sub buffers here
    sycl::buffer<dtype> col_buf(col.data, col.count());

    const bool has_bias = bias.has_value();
    sycl::buffer<dtype> bias_buf =
        has_bias ? sycl::buffer<dtype>(bias->data, bias->count())
                 : sycl::buffer<dtype>(sycl::range<1>(0));

    sycl::buffer<dtype> input_buf(input.data, input.count());
    sycl::buffer<dtype> kernel_buf(kernel.data, kernel.count());

    sycl::buffer<dtype> output_buf(output.data, output.count());

    sycl::queue queue(sycl::default_selector_v,
                      sycl::property::queue::in_order());

    const uint max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    const uint kernel_area = kernel_height * kernel_width;

    dtype scaler = std::cbrt(dtype(max_work_group_size) /
                             (num_samples * input_channels * kernel_area));

    uint each_sample = num_samples * scaler;
    uint each_channel = input_channels * scaler;
    uint each_kernel = kernel_area * scaler;
    if (each_sample == 0) {
        scaler = 1 / (scaler * num_samples);
        each_sample = 1;
        each_kernel /= scaler;
        if (each_kernel == 0) {
            each_kernel = 1;
        }
        each_channel /= scaler;
        if (each_channel == 0) {
            each_channel = 1;
        }
    }
    if (each_channel == 0) {
        scaler = 1 / (scaler * input_channels);
        each_channel = 1;
        each_kernel /= scaler;
        if (each_kernel == 0) {
            each_kernel = 1;
        }
        each_sample /= scaler;
        if (each_sample == 0) {
            each_sample = 1;
        }
    }
    if (each_kernel == 0) {
        scaler = 1 / (scaler * kernel_area);
        each_kernel = 1;
        each_channel /= scaler;
        if (each_channel == 0) {
            each_channel = 1;
        }
        each_sample /= scaler;
        if (each_sample == 0) {
            each_sample = 1;
        }
    }

    uint total_samples =
        ((num_samples + each_sample - 1) / each_sample) * each_sample;
    uint total_channels =
        ((input_channels + each_channel - 1) / each_channel) * each_channel;
    uint total_kernel =
        ((kernel_area + each_kernel - 1) / each_kernel) * each_kernel;

    queue.submit([&](sycl::handler &h) {
        sycl::accessor ainput = sycl::accessor(input_buf, h, sycl::read_only);
        sycl::accessor acol =
            sycl::accessor(col_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(
            sycl::nd_range(
                sycl::range(total_samples, total_channels, total_kernel),
                sycl::range(each_sample, each_channel, each_kernel)),
            [=](sycl::nd_item<3> item) {
                uint samp = item.get_global_id(0);
                uint in_c = item.get_global_id(1);
                uint ker = item.get_global_id(2);
                if (samp >= num_samples || in_c >= input_channels ||
                    ker >= kernel_area)
                    return;
                uint ker_w = ker % kernel_width;
                uint ker_h = ker / kernel_width;

                for (uint out_h = 0; out_h < output_height; ++out_h) {
                    for (uint out_w = 0; out_w < output_width; ++out_w) {
                        uint h_offset =
                            out_h * stride_h - padding_h + ker_h * dilation_h;
                        uint w_offset =
                            out_w * stride_w - padding_w + ker_w * dilation_w;
                        uint col_index = to_linear(
                            samp, in_c, ker, out_h, out_w, input_channels,
                            kernel_area, output_height, output_width);
                        if (h_offset >= 0 && h_offset < input_height &&
                            w_offset >= 0 && w_offset < input_width) {
                            acol[col_index] = ainput[to_linear(
                                samp, in_c, h_offset, w_offset, input_channels,
                                input_height, input_width)];
                        } else {
                            acol[col_index] = 0;
                        }
                    }
                }
            });
    });

    const uint output_area = output_height * output_width;

    scaler = std::cbrt(dtype(max_work_group_size) /
                       (num_samples * output_channels * output_area));

    each_sample = num_samples * scaler;
    each_channel = output_channels * scaler;
    uint each_output = output_area * scaler;
    if (each_sample == 0) {
        scaler = 1 / (scaler * num_samples);
        each_sample = 1;
        each_output /= scaler;
        if (each_output == 0) {
            each_output = 1;
        }
        each_channel /= scaler;
        if (each_channel == 0) {
            each_channel = 1;
        }
    }
    if (each_channel == 0) {
        scaler = 1 / (scaler * output_channels);
        each_channel = 1;
        each_output /= scaler;
        if (each_output == 0) {
            each_output = 1;
        }
        each_sample /= scaler;
        if (each_sample == 0) {
            each_sample = 1;
        }
    }
    if (each_output == 0) {
        scaler = 1 / (scaler * output_area);
        each_output = 1;
        each_channel /= scaler;
        if (each_channel == 0) {
            each_channel = 1;
        }
        each_sample /= scaler;
        if (each_sample == 0) {
            each_sample = 1;
        }
    }

    total_samples =
        ((num_samples + each_sample - 1) / each_sample) * each_sample;
    total_channels =
        ((output_channels + each_channel - 1) / each_channel) * each_channel;
    uint total_output =
        ((output_area + each_output - 1) / each_output) * each_output;

    queue.submit([&](sycl::handler &h) {
        sycl::accessor akernel = sycl::accessor(kernel_buf, h, sycl::read_only);
        sycl::accessor acol = sycl::accessor(col_buf, h, sycl::read_only);
        sycl::accessor abias = sycl::accessor(bias_buf, h, sycl::read_only);
        sycl::accessor aoutput =
            sycl::accessor(output_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(
            sycl::nd_range(
                sycl::range(total_samples, total_channels, total_output),
                sycl::range(1, 1, each_output)),
            [=](sycl::nd_item<3> item) {
                int samp = item.get_global_id(0);
                int out_c = item.get_global_id(1);
                int out_id = item.get_global_id(2);
                if (samp >= num_samples || out_c >= output_channels ||
                    out_id >= output_area)
                    return;
                dtype res = 0;
                for (uint in_c = 0; in_c < input_channels; in_c++) {
                    for (uint ker = 0; ker < kernel_area; ker++) {
                        res += acol[to_linear(samp, in_c, ker, out_id,
                                              input_channels, kernel_area,
                                              output_area)] *
                               akernel[to_linear(out_c, in_c, ker,
                                                 input_channels, kernel_area)];
                    }
                }
                if (has_bias) {
                    res += abias[out_c];
                }
                aoutput[to_linear(samp, out_c, out_id, output_channels,
                                  output_height * output_width)] = res;
            });
    });

    queue.wait_and_throw();

    return output;
}
