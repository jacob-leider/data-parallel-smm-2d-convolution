#pragma once

#include "ai3.hpp"
#include "utils.hpp"
#include <CL/sycl.hpp>
#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>
using namespace cl;

// TODO groups and padding modes
template <typename dtype>
Tensor<dtype>
smm_conv2d(const Tensor<dtype> &input, const Tensor<dtype> &kernel,
           const std::optional<const Tensor<dtype>> &bias,
           const std::vector<int> &padding, const std::vector<int> &stride,
           const std::vector<int> &dilation, const PaddingMode padding_mode,
           int groups) {
    errs::bail_if(padding_mode != Zeros, "padding mode must be zeros");
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

    int col_height = input_channels * kernel_height * kernel_width;
    int col_width = output_height * output_width;
    Tensor<dtype> col({num_samples, col_height, col_width});
    sycl::buffer<dtype> col_buf(col.data, col.count());

    const bool has_bias = bias.has_value();
    sycl::buffer<dtype> bias_buf =
        has_bias ? sycl::buffer<dtype>(bias->data, bias->count())
                 : sycl::buffer<dtype>(sycl::range<1>(0));

    sycl::buffer<dtype> input_buf(input.data, input.count());
    sycl::buffer<dtype> kernel_buf(kernel.data, kernel.count());

    sycl::buffer<dtype> output_buf(output.data, output.count());

    sycl::queue queue(sycl::default_selector_v);

    const int kernel_area = kernel_height * kernel_width;
    const int max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    int work_group_size = max_work_group_size;
    if (GROUP_SIZE_GUESS < work_group_size) {
        work_group_size = GROUP_SIZE_GUESS;
    }
    if (kernel_area < work_group_size) {
        work_group_size = kernel_area;
    }

    const int total_work_group_size =
        ((kernel_area + work_group_size - 1) / work_group_size) *
        work_group_size;

    queue.submit([&](sycl::handler &h) {
        sycl::accessor ainput = sycl::accessor(input_buf, h, sycl::read_only);
        sycl::accessor acol =
            sycl::accessor(col_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(
            sycl::nd_range(
                sycl::range(num_samples, input_channels, total_work_group_size),
                sycl::range(1, 1, work_group_size)),
            [=](sycl::nd_item<3> item) {
                int samp = item.get_global_id(0);
                int in_c = item.get_global_id(1);
                int ker = item.get_global_id(2);
                if (ker >= kernel_area)
                    return;
                int ker_w = ker % kernel_width;
                int ker_h = ker / kernel_width;

                for (int out_h = 0; out_h < output_height; ++out_h) {
                    for (int out_w = 0; out_w < output_width; ++out_w) {
                        int h_offset =
                            out_h * stride_h - padding_h + ker_h * dilation_h;
                        int w_offset =
                            out_w * stride_w - padding_w + ker_w * dilation_w;
                        int col_index = to_linear(samp, in_c, ker, out_h, out_w,
                                                  input_channels,
                                                  kernel_height * kernel_width,
                                                  output_height, output_width);
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

    const int output_area = output_height * output_width;
    work_group_size = max_work_group_size;
    if (GROUP_SIZE_GUESS < work_group_size) {
        work_group_size = GROUP_SIZE_GUESS;
    }
    if (output_area < work_group_size) {
        work_group_size = output_area;
    }
    const int output_size_per_channel_total =
        ((output_area + work_group_size - 1) / work_group_size) *
        work_group_size;

    queue.submit([&](sycl::handler &h) {
        sycl::accessor akernel = sycl::accessor(kernel_buf, h, sycl::read_only);
        sycl::accessor acol = sycl::accessor(col_buf, h, sycl::read_only);
        sycl::accessor abias = sycl::accessor(bias_buf, h, sycl::read_only);
        sycl::accessor aoutput =
            sycl::accessor(output_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(
            sycl::nd_range(sycl::range(num_samples, output_channels,
                                       output_size_per_channel_total),
                           sycl::range(1, 1, work_group_size)),
            [=](sycl::nd_item<3> item) {
                int samp = item.get_global_id(0);
                int out_c = item.get_global_id(1);
                int out_id = item.get_global_id(2);
                if (out_id >= output_area)
                    return;
                dtype res = 0;
                for (int in_c = 0; in_c < input_channels; in_c++) {
                    for (int ker = 0; ker < kernel_area; ker++) {
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
