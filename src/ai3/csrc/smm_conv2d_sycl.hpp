#pragma once

#include "ai3.hpp"
#include "utils.hpp"
#include <CL/sycl.hpp>
#include <cstddef>
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

    sycl::queue queue(sycl::default_selector_v);

    const uint kernel_area = kernel_height * kernel_width;
    uint col_height = input_channels * kernel_area;
    uint col_width = output_height * output_width;
    dtype *col_data =
        sycl::malloc_device<dtype>(num_samples * col_height * col_width, queue);
    dtype *input_data = sycl::malloc_device<dtype>(input.count(), queue);
    queue.memcpy(input_data, input.data, input.count() * sizeof(dtype));

    dtype *kernel_data = sycl::malloc_device<dtype>(kernel.count(), queue);
    queue.memcpy(kernel_data, kernel.data, kernel.count() * sizeof(dtype));

    const bool has_bias = bias.has_value();
    dtype *bias_data = nullptr;
    if (has_bias) {
        bias_data = sycl::malloc_device<dtype>(bias->count(), queue);
        queue.memcpy(bias_data, bias->data, bias->count() * sizeof(dtype));
    }
    dtype *output_data = sycl::malloc_device<dtype>(output.count(), queue);
    queue.wait();

    const uint max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    uint work_group_size = max_work_group_size;
    if (GROUP_SIZE_GUESS < work_group_size) {
        work_group_size = GROUP_SIZE_GUESS;
    }
    if (kernel_area < work_group_size) {
        work_group_size = kernel_area;
    }

    const uint total_work_group_size =
        ((kernel_area + work_group_size - 1) / work_group_size) *
        work_group_size;

    queue
        .submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range(sycl::range(num_samples, input_channels,
                                           total_work_group_size),
                               sycl::range(1, 1, work_group_size)),
                [=](sycl::nd_item<3> item) {
                    uint samp = item.get_global_id(0);
                    uint in_c = item.get_global_id(1);
                    uint ker = item.get_global_id(2);
                    if (ker >= kernel_area)
                        return;
                    uint ker_w = ker % kernel_width;
                    uint ker_h = ker / kernel_width;

                    for (uint out_h = 0; out_h < output_height; ++out_h) {
                        for (uint out_w = 0; out_w < output_width; ++out_w) {
                            uint h_offset = out_h * stride_h - padding_h +
                                            ker_h * dilation_h;
                            uint w_offset = out_w * stride_w - padding_w +
                                            ker_w * dilation_w;
                            uint col_index = to_linear(
                                samp, in_c, ker, out_h, out_w, input_channels,
                                kernel_area, output_height, output_width);
                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                col_data[col_index] = input_data[to_linear(
                                    samp, in_c, h_offset, w_offset,
                                    input_channels, input_height, input_width)];
                            } else {
                                col_data[col_index] = 0;
                            }
                        }
                    }
                });
        })
        .wait_and_throw();

    const uint output_area = output_height * output_width;
    work_group_size = max_work_group_size;
    if (GROUP_SIZE_GUESS < work_group_size) {
        work_group_size = GROUP_SIZE_GUESS;
    }
    if (output_area < work_group_size) {
        work_group_size = output_area;
    }
    const uint output_size_per_channel_total =
        ((output_area + work_group_size - 1) / work_group_size) *
        work_group_size;

    queue
        .submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range(sycl::range(num_samples, output_channels,
                                           output_size_per_channel_total),
                               sycl::range(1, 1, work_group_size)),
                [=](sycl::nd_item<3> item) {
                    uint samp = item.get_global_id(0);
                    uint out_c = item.get_global_id(1);
                    uint out_id = item.get_global_id(2);
                    if (out_id >= output_area)
                        return;
                    dtype res = 0;
                    for (uint in_c = 0; in_c < input_channels; in_c++) {
                        for (uint ker = 0; ker < kernel_area; ker++) {
                            res += col_data[to_linear(
                                       samp, in_c, ker, out_id, input_channels,
                                       kernel_area, output_area)] *
                                   kernel_data[to_linear(out_c, in_c, ker,
                                                         input_channels,
                                                         kernel_area)];
                        }
                    }
                    if (has_bias) {
                        res += bias_data[out_c];
                    }
                    output_data[to_linear(samp, out_c, out_id, output_channels,
                                          output_height * output_width)] = res;
                });
        })
        .wait_and_throw();

    queue.memcpy(output.data, output_data, output.count() * sizeof(dtype));
    queue.wait();
    sycl::free(col_data, queue);
    sycl::free(input_data, queue);
    sycl::free(kernel_data, queue);
    if (bias_data != nullptr) {
        sycl::free(bias_data, queue);
    }
    sycl::free(output_data, queue);
    return output;
}
