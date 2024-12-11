#pragma once

#include <ai3.hpp>
#include <CL/sycl.hpp>
#include <algos.hpp>
#include <optional>
/**
 * @DEFAULT_BOOL{Conv2D}
 */
const bool DEFAULT_CONV2D = false;

/**
 * @CUSTOM_OP{Conv2D,conv2d}
 */
template <typename dtype>
Tensor conv2d_custom(Tensor input, const Tensor &kernel,
                     const std::optional<const Tensor> &bias,
                     const uint padding_h, const uint padding_w,
                     const uint stride_h, const uint stride_w,
                     const uint dilation_h, const uint dilation_w,
                     const PaddingMode padding_mode, uint groups) {
    ensure_same_type(input, kernel, bias);
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();

    const uint kernel_height = kernel.height();
    const uint kernel_width = kernel.width();

    const uint output_channels = kernel.output_channels();

    const uint output_height = output_hw_for_2d<dtype>(
        input_height, kernel_height, padding_h, dilation_h, stride_h, false);
    const uint output_width = output_hw_for_2d<dtype>(
        input_width, kernel_width, padding_w, dilation_w, stride_w, false);

    uint num_samples;
    Tensor output;
    if (input.batched(sample_dims::CONV2D)) {
        num_samples = input.batch_size(sample_dims::CONV2D);
        output =
            Tensor({num_samples, output_channels, output_height, output_width},
                   input.scalar_type);
    } else {
        num_samples = 1;
        output = Tensor({output_channels, output_height, output_width},
                        input.scalar_type);
    }
    
    ////////////////////////////////////////////////////////////////////////

    uint col_height = input_channels * kernel_height * kernel_width;
    uint col_width = output_height * output_width;

    sycl::queue *queue_ptr = static_cast<sycl::queue *>(Context::sycl_queue());
    sycl::queue queue = *queue_ptr;

    const bool has_bias = bias.has_value();
    
    sycl::buffer<dtype> bias_buf =
        has_bias
            ? sycl::buffer<dtype>(data_as<dtype>(bias->data), bias->count())
            : sycl::buffer<dtype>(sycl::range<1>(0));

    sycl::buffer<dtype> buf_cols(num_samples * col_height * col_width);

    sycl::buffer<dtype> buf_input(data_as<dtype>(input.data), input.count());
    sycl::buffer<dtype> buf_kernel(data_as<dtype>(kernel.data), kernel.count());

    sycl::buffer<dtype> buf_output(data_as<dtype>(output.data), output.count());

    const uint max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    const uint kernel_area = kernel_height * kernel_width;

    uint each_kernel = std::min<uint>({max_work_group_size, 
            GROUP_SIZE_GUESS, 
            kernel_area});

    const uint total_kernel =
        ((kernel_area + each_kernel - 1) / each_kernel) * each_kernel;

    ////////////////////////////////////////////////////////////////////////
    
    // WORKS
    queue.submit([&](sycl::handler &h) {
        sycl::accessor acc_output =
            buf_output.template get_access<sycl::access::mode::write>(h);
        sycl::accessor acc_bias =
            bias_buf.template get_access<sycl::access::mode::read>(h);
        h.parallel_for(
            sycl::nd_range(
                sycl::range(num_samples, output_channels, output_height),
                sycl::range(1, 1, output_height)),
            [=](sycl::nd_item<3> item) {
                uint samp = item.get_global_id(0);
                uint o_channel = item.get_global_id(1);
                uint o_row = item.get_global_id(2);

                for (uint out_w = 0; out_w < output_width; out_w++)
                {
                    if (has_bias)
                    {
                        acc_output[to_linear(samp, 
                                o_channel, o_row, out_w, 
                                output_channels, output_height, output_width)] = acc_bias[o_channel];
                    }
                    else
                    {
                        acc_output[to_linear(samp, 
                                o_channel, o_row, out_w, 
                                output_channels, output_height, output_width)] = 0;
                    }
                }
                
            }
        );
    });
    queue.wait_and_throw();


    queue.submit([&](sycl::handler &h) {
        // Accessors: Input, kernel, output.
        sycl::accessor acc_output =
            buf_output.template get_access<sycl::access::mode::write>(h);
        sycl::accessor acc_input =
            buf_input.template get_access<sycl::access::mode::write>(h);
        sycl::accessor acc_kern =
            buf_kernel.template get_access<sycl::access::mode::write>(h);

        h.parallel_for(
            sycl::nd_range(
                sycl::range(num_samples, output_channels),
                sycl::range(1, 1)),
            [=](sycl::nd_item<2> item) {
                // Parallelized by (1) sample, (2) output channel, (3) ...
                uint samp = item.get_global_id(0);
                uint out_c = item.get_global_id(1);

                for (uint in_c = 0; in_c < input_channels; ++in_c) {
                    for (uint ker_w = 0; ker_w < kernel_width; ++ker_w) {
                        for (uint ker_h = 0; ker_h < kernel_height; ++ker_h) {
                            
                            // Weight.
                            const dtype w = acc_kern[to_linear(
                                            out_c, 
                                            in_c, 
                                            ker_h, ker_w,
                                            input_channels, 
                                            kernel_height, kernel_width)];

                            // Multiply & accumulate the submatrix with "w".
                            for (uint out_h = 0; out_h < output_height; out_h++) {
                                for (uint out_w = 0; out_w < output_width; out_w++) {

                                    uint h_offset = out_h * stride_h - padding_h +
                                                    ker_h * dilation_h;
                                    uint w_offset = out_w * stride_w - padding_w +
                                                    ker_w * dilation_w;

                                    if (!(h_offset >= 0 && h_offset < input_height &&
                                            w_offset >= 0 && w_offset < input_width))
                                        continue;

                                    acc_output[to_linear(samp, 
                                            out_c, out_h, out_w,
                                            output_channels, output_height,
                                            output_width)]
                                            += acc_input[to_linear(samp, 
                                                    in_c, h_offset, w_offset,
                                                    input_channels, input_height,
                                                    input_width)] * w;
                                }
                            }
                        }
                    }
                }
            }
        );
    });
    queue.wait_and_throw();

    return output;
}
