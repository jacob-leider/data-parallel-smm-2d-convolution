from collections import defaultdict
import time
import torch
from bench import USE_TORCH_COMPILE, warm_up
from torch import nn
from test import compare_tensors
import ai3
try:
    import intel_extension_for_pytorch as ipex
    ipex_found = True
except ModuleNotFoundError:
    ipex_found = False

USE_TORCH_COMPILE = False

def time_forward(runner, data):
    warm_up(runner, data)
    start = time.time()
    out = runner(data)
    end= time.time()
    return out, end - start

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


def gather_conv2d_times(input):
    times_for_layer = defaultdict(float)
    orig = Conv2D(input.shape[1], input.shape[1], 3)
    orig.eval()
    torch_out, times_for_layer["torch conv2d"] = time_forward(orig, input)

    if USE_TORCH_COMPILE:
        torch_comp = torch.compile(orig)
        torch_comp_out, times_for_layer["torch comp conv2d"] = time_forward(torch_comp, input)
        compare_tensors(torch_comp_out, torch_out)

    swap_direct = ai3.swap_backend(orig, {"conv2d": "direct"})
    direct_out, times_for_layer[ "ai3 direct conv2d"]= time_forward(swap_direct, input)

    swap_smm = ai3.swap_backend(orig, {"conv2d": "smm"})
    smm_out, times_for_layer[ "ai3 smm conv2d"]= time_forward(swap_smm, input)

    if ipex_found:
        ipex_model = ipex.optimize(orig, dtype=torch.float32)
        ipex_out,  times_for_layer["ipex conv2d"] = time_forward(ipex_model, input)
        compare_tensors(ipex_out, torch_out)
    compare_tensors(smm_out, torch_out)
    compare_tensors(direct_out, torch_out)
    return times_for_layer

with torch.inference_mode():
    input = torch.randn(100, 3, 224, 224)
    conv2d_times_area = gather_conv2d_times(input)
    input = torch.randn(100, 512, 14, 14)
    conv2d_times_channels = gather_conv2d_times(input)
    print(conv2d_times_area)
    print(conv2d_times_channels)


# using Conv2d in existing models
# orig_models = {"AlexNet" :tvm.alexnet(),
#                "DenseNet" : tvm.DenseNet(),
#                "GoogleNet" : tvm.googlenet(),
#                "Incetion V3" : tvm.inception_v3(),
#                "ResNet152" : tvm.resnet152(),
#                "Squeezenet 1.1": tvm.squeezenet1_1(),
#                "Swin Transformer Base": tvm.swin_b(),
#                "VGG16" : tvm.vgg16(),
#                "VIT Base 16" : tvm.vit_b_16()}
