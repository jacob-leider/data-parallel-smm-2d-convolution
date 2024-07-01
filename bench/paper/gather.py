from collections import defaultdict
import torch
from bench import predict_show_time
from torch import nn
from test import compare_tensors
import torchvision.models as tvm
import ai3
try:
    import intel_extension_for_pytorch as ipex
    ipex_found = True
except ModuleNotFoundError:
    ipex_found = False

# just Conv2d
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


times_for_layer = defaultdict(float)
input = torch.randn(100, 3, 224, 224)
orig = Conv2D(3, 3, 3)
orig.eval()
torch_out = predict_show_time(orig, input, "torch conv2d", times_for_layer)
assert(isinstance(torch_out, torch.Tensor))
swap_direct = ai3.swap_backend(orig, {"conv2d": "direct"})
swap_direct_out = predict_show_time(swap_direct, input, "ai3 direct conv2d", times_for_layer)
swap_smm = ai3.swap_backend(orig, {"conv2d": "smm"})
swap_smm_out = predict_show_time(swap_smm, input, "ai3 smm conv2d", times_for_layer)
if ipex_found:
     ipex_model = ipex.optimize(orig, dtype=torch.float32)
     ipex_out = predict_show_time(ipex_model, input, "ipex conv2d", times_for_layer)
     compare_tensors(ipex_out, torch_out)
compare_tensors(swap_smm_out, torch_out)
compare_tensors(swap_direct_out, torch_out)

print(times_for_layer)

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
