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

    const bool has_bias = bias.has_value();

    const dtype *in_data = data_as<dtype>(input.data);
    const dtype *kern_data = data_as<dtype>(kernel.data);
    dtype *bias_data = nullptr;
    if (has_bias) {
        bias_data = data_as<dtype>(bias->data);
    }

    dtype *out_data = data_as<dtype>(output.data);
    
    for (uint samp = 0; samp < num_samples; samp++) {
        for (size_t c = 0; c < input_channels; c++)
        {
            for (size_t k_col  = 0; k_col < kernel_width; k_col++)
            {
                // Sliced Mat ← T_j^c (slice j)
                for (k_row = 0; k_row < kernel_height; k_row++)
                {
                    // Shifted Mat ← Sliced Mat[k:h' +k,:] (slice j, shifted down k).
                    for (size_t m = 0; m < output_channels; m++)
                    {                        
                        // w ← K[c,j,k,m]
                        const float w = kern_data[to_linear(
                                m, c, 
                                k_row, k_col, 
                                output_channels, input_channels, 
                                kernel_height)];
                        
                        // Multiply w by submatrix of T_j^c, add it to output.
                        for (size_t o_row = 0; o_row < output_height; o_row++)
                        {
                            for (size_t o_col = 0; o_col < output_width; o_col++)
                            {
                                // Input row and column offsets.
                                const size_t i_row = o_row * stride_h + k_row * dilation_h - padding_h;
                                const size_t i_col = o_col * stride_w + k_col * dilation_w - padding_w;
                                
                                // Output flattened offset.
                                const size_t o_offset = to_linear(samp, m, 
                                        o_row, o_col, 
                                        output_channels, 
                                        output_height, 
                                        output_width);

                                // Input flattened offset.
                                const size_t i_offset = to_linear(samp, c, 
                                        i_row, i_col,
                                        input_channels, 
                                        input_height,
                                        input_width);

                                out_data[o_offset] = w * in_data[i_offset];
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (has_bias) {
        for (size_t o_channel = 0; o_channel < output_channels; o_channel++)
        {
            for (size_t o_row = 0; o_row < output_height; o_row++)
            {
                for (size_t o_col = 0; o_col < output_width; o_col++)
                {
                    const size_t o_offset = to_linear(o_channel, 
                            o_row, o_col, 
                            output_channels,
                            output_height);

                    out_data[o_offset] += bias;
                }
            }
        }
    }

    return output;
}
