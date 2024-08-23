// TODO check what is happening on large models
#import "ai3.hpp"
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <mutex>

const uint MPSTENSOR_RANK = 4;

inline uint computeMPSAlignOffset(const uint kernel, const uint padding,
                                  const uint dilation = 1) {
    return (((kernel - 1) * dilation + 1) / 2) - padding;
}

void *gen_mps_graph_device(void) {
    MPSGraphDevice *device =
        [MPSGraphDevice deviceWithMTLDevice:MTLCreateSystemDefaultDevice()];
    return device;
}

MPSShape *mps_shape(const std::vector<uint> &shape, const bool bias) {
    NSMutableArray *ret = [NSMutableArray arrayWithCapacity:MPSTENSOR_RANK];
    uint i = 0;
    uint shift = 0;
    if (bias) {
        [ret addObject:@1];
        [ret addObject:@(shape[0])];
        [ret addObject:@1];
        [ret addObject:@1];
    } else {
        if (shape.size() < MPSTENSOR_RANK) {
            ret[0] = @1;
            i = 1;
            shift = 1;
        }
        for (; i < shape.size() + shift; i++) {
            ret[i] = @(shape[i - shift]);
        }
    }

    return ret;
}

struct MPSTensor {
    MPSGraphTensor *placeholder;
    MPSGraphTensorData *data;
};

MPSGraphTensorData *output_tensor_data(MPSGraphDevice *device,
                                       MPSGraphTensor *placeholder,
                                       const Tensor<float> &tens) {
    id<MTLBuffer> output_buffer = [[device metalDevice]
        newBufferWithBytesNoCopy:tens.data
                          length:sizeof(float) * tens.count()
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    MPSGraphTensorData *output_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer
                                                shape:[placeholder shape]
                                             dataType:MPSDataTypeFloat32];
    return output_data;
}

MPSTensor feed_tensor(MPSGraph *graph, MPSGraphDevice *device,
                      const Tensor<float> &tens, const bool bias = false) {
    MPSGraphTensor *placeholder =
        [graph placeholderWithShape:mps_shape(tens.shape, bias)
                           dataType:MPSDataTypeFloat32
                               name:nil];
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
        initWithDevice:device
                  data:[NSData dataWithBytesNoCopy:tens.data
                                            length:sizeof(float) * tens.count()]

                 shape:[placeholder shape]
              dataType:MPSDataTypeFloat32];
    return MPSTensor{
        placeholder,
        data,
    };
}

Tensor<float> metal_conv2d(Tensor<float> input, const Tensor<float> &kernel,
                           const std::optional<const Tensor<float>> &bias,
                           const uint padding_h, const uint padding_w,
                           const uint stride_h, const uint stride_w,
                           const uint dilation_h, const uint dilation_w,
                           const PaddingMode padding_mode, uint groups) {
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

    const uint output_channels = kernel.output_channels();
    const uint output_h =
        output_hw_for_2d<float>(input.height(), kernel.height(), padding_h,
                                dilation_h, stride_h, false);
    const uint output_w = output_hw_for_2d<float>(
        input.width(), kernel.width(), padding_w, dilation_w, stride_w, false);

    uint num_samples;
    Tensor<float> output;
    if (input.batched(sample_dims::CONV2D)) {
        num_samples = input.batch_size(sample_dims::CONV2D);
        output =
            Tensor<float>({num_samples, output_channels, output_h, output_w});
    } else {
        num_samples = 1;
        output = Tensor<float>({output_channels, output_h, output_w});
    }

    MPSGraphDevice *device = (MPSGraphDevice *)Context::mps_graph_device();

    MPSGraph *graph = [MPSGraph new];
    MPSGraphConvolution2DOpDescriptor *conv_desc =
        [MPSGraphConvolution2DOpDescriptor new];
    conv_desc.strideInY = stride_h;
    conv_desc.strideInX = stride_w;
    conv_desc.dilationRateInY = dilation_h;
    conv_desc.dilationRateInX = dilation_w;
    conv_desc.paddingTop = padding_h;
    conv_desc.paddingBottom = padding_h;
    conv_desc.paddingLeft = padding_w;
    conv_desc.paddingRight = padding_w;
    conv_desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    conv_desc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    conv_desc.groups = 1;

    MPSTensor in_tens = feed_tensor(graph, device, input);

    MPSTensor kern_tens = feed_tensor(graph, device, kernel);

    MPSGraphTensor *output_tensor =
        [graph convolution2DWithSourceTensor:in_tens.placeholder
                               weightsTensor:kern_tens.placeholder
                                  descriptor:conv_desc
                                        name:nil];

    std::optional<MPSTensor> bias_tens = std::nullopt;
    const bool has_bias = bias.has_value();
    if (has_bias) {
        bias_tens = feed_tensor(graph, device, *bias, true);
        output_tensor = [graph additionWithPrimaryTensor:output_tensor
                                         secondaryTensor:bias_tens->placeholder
                                                    name:nil];
    }
    MPSGraphTensorData *output_data =
        output_tensor_data(device, output_tensor, output);

    NSMutableDictionary *feeds =
        [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
    feeds[in_tens.placeholder] = in_tens.data;
    feeds[kern_tens.placeholder] = kern_tens.data;

    if (has_bias) {
        feeds[bias_tens->placeholder] = bias_tens->data;
    }

    id<MTLCommandQueue> command_queue = [[device metalDevice] newCommandQueue];
    MPSCommandBuffer *command_buffer =
        [MPSCommandBuffer commandBufferFromCommandQueue:command_queue];

    [graph encodeToCommandBuffer:command_buffer
                           feeds:feeds
                targetOperations:nil
               resultsDictionary:@{output_tensor : output_data}
             executionDescriptor:[MPSGraphExecutionDescriptor new]];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    return output;
}

Tensor<double> metal_conv2d(Tensor<double> input, const Tensor<double> &kernel,
                            const std::optional<const Tensor<double>> &bias,
                            const uint padding_h, const uint padding_w,
                            const uint stride_h, const uint stride_w,
                            const uint dilation_h, const uint dilation_w,
                            const PaddingMode padding_mode, uint groups) {
    Tensor<float> input_float = input.template to_type<float>();
    Tensor<float> kernel_float = kernel.template to_type<float>();
    std::optional<const Tensor<float>> bias_float =
        bias.has_value()
            ? std::optional<Tensor<float>>(bias->template to_type<float>())
            : std::nullopt;
    errs::warning(
        "MPS does not support double precision, transforming tensors to float "
        "precision and back see: "
        "https://developer.apple.com/documentation/metalperformanceshaders/"
        "mpsdatatype");
    return metal_conv2d(std::move(input_float), kernel_float, bias_float,
                        padding_h, padding_w, stride_h, stride_w, dilation_h,
                        dilation_w, padding_mode, groups)
        .template to_type<double>();
}
