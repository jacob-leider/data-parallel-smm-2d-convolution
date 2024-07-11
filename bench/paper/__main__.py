from collections import defaultdict
import time
import torch
from bench import warm_up
from torch import nn
import pandas as pd
from test import compare_tensors
import matplotlib.pyplot as plt
import ai3
import os
import torchvision.models as tvm
try:
    import intel_extension_for_pytorch as ipex
    ipex_found = True
except ModuleNotFoundError:
    ipex_found = False

CUDA_AVAILABLE = torch.cuda.is_available()
result_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "results")
if CUDA_AVAILABLE:
    SAVE_TO_DIR = os.path.join(result_dir, "gpu")
else:
    SAVE_TO_DIR = os.path.join(result_dir, "cpu")
os.makedirs(SAVE_TO_DIR, exist_ok=True)
plt.rcParams['savefig.dpi'] = 500


def time_forward(runner, data):
    warm_up(runner, data)
    start = time.time()
    out = runner(data)
    end = time.time()
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
    if CUDA_AVAILABLE:
        torch_name = "torch"
    else:
        torch_name = "torch eager mode"
    torch_out, times_for_layer[torch_name] = time_forward(orig, input)

    if CUDA_AVAILABLE:
        torch_comp = torch.compile(orig)
        torch_comp_out, times_for_layer["torch graph mode"] = time_forward(
            torch_comp, input)
        compare_tensors(torch_comp_out, torch_out)

    swap_direct = ai3.swap_backend(orig, {"conv2d": "direct"})
    direct_out, times_for_layer["ai3 direct"] = time_forward(
        swap_direct, input)

    swap_smm = ai3.swap_backend(orig, {"conv2d": "smm"})
    smm_out, times_for_layer["ai3 SMM"] = time_forward(swap_smm, input)

    if ipex_found and not CUDA_AVAILABLE:
        ipex_model = ipex.optimize(orig, dtype=torch.float32)
        ipex_out,  times_for_layer["ipex"] = time_forward(
            ipex_model, input)
        compare_tensors(ipex_out, torch_out)
    compare_tensors(smm_out, torch_out)
    compare_tensors(direct_out, torch_out)
    return times_for_layer


def save_plot(data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(data.keys(), data.values())
    plt.xlabel('Backends')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.savefig(os.path.join(SAVE_TO_DIR, filename))
    plt.close()


N = 100
BENCH_AREA_SHAPE = (N, 3, 224, 224)
BENCH_CHANNEL_SHAPE = (N, 512, 14, 14)


def save_combined_plot(data_area, data_channel):
    plt.figure(figsize=(10, 5))
    keys = list(data_area.keys())
    x = range(len(keys))

    bar_width = 0.3
    plt.bar(x, data_area.values(), width=bar_width, align='center',
            label=f'Input Shape {BENCH_AREA_SHAPE}', color='blue')
    plt.bar([i + bar_width for i in x], data_channel.values(), width=bar_width,
            align='center', label=f'Input Shape {BENCH_CHANNEL_SHAPE}', color='orange')

    plt.xlabel('Backends', fontsize=14)
    plt.ylabel('Time (s)', fontsize=14)
    plt.title('Latency of Conv2D Operation', fontsize=16)
    plt.xticks([i + bar_width / 2 for i in x], keys)
    plt.legend()
    plt.savefig(os.path.join(SAVE_TO_DIR, "combined_conv2d_times.png"))
    plt.close()

def gather_model_times(model, input):
    times_for_model = defaultdict(float)
    model.eval()
    if CUDA_AVAILABLE:
        torch_name = "torch"
    else:
        torch_name = "torch eager mode"
    torch_out, times_for_model[torch_name] = time_forward(model, input)

    if CUDA_AVAILABLE:
        torch_comp = torch.compile(model)
        torch_comp_out, times_for_model["torch graph mode"] = time_forward(
            torch_comp, input)
        compare_tensors(torch_comp_out, torch_out)

    if ipex_found and not CUDA_AVAILABLE:
        ipex_model = ipex.optimize(model, dtype=torch.float32)
        ipex_out,  times_for_model["ipex"] = time_forward(
            ipex_model, input)
        compare_tensors(ipex_out, torch_out)

    ai3.swap_conv2d(model, "direct")
    direct_out, times_for_model["ai3 direct"] = time_forward(
        model, input)

    ai3.swap_conv2d(model, "smm")
    smm_out, times_for_model["ai3 SMM"] = time_forward(model, input)

    compare_tensors(smm_out, torch_out)
    compare_tensors(direct_out, torch_out)
    return times_for_model

def save_model_data_table(models_data):
    df = pd.DataFrame(models_data).transpose()
    df = df.round(4)

    if 'torch' in df.columns:
        norm_column = 'torch'
    else:
        norm_column = 'torch eager mode'
    df = df.div(df[norm_column], axis=0)

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                     cellLoc='center', loc='center')

    plt.savefig(os.path.join(SAVE_TO_DIR, 'model_times_table.png'), bbox_inches='tight')
    plt.close()


# ['SIMPLE_CREATED', 'VGG16', 'ALEXNET', 'DENSENET', 'GOOGLENET', 'INCEPTION', 'SQUEEZENET', 'VISION_TRANSFORMER', 'SWIN_TRANSFORMER', 'RESNET']
with torch.inference_mode():
    input = torch.randn(BENCH_AREA_SHAPE)
    print('conv2d area')
    conv2d_times_area = gather_conv2d_times(input)

    input = torch.randn(BENCH_CHANNEL_SHAPE)
    print('conv2d channels')
    conv2d_times_channels = gather_conv2d_times(input)
    save_combined_plot(conv2d_times_area, conv2d_times_channels)
    print(conv2d_times_area)
    print(conv2d_times_channels)

    input = torch.randn(BENCH_AREA_SHAPE)
    orig_models = {"AlexNet" :tvm.alexnet(),
                   "DenseNet" : tvm.DenseNet(),
                   "GoogleNet" : tvm.googlenet(),
                   "Incetion V3" : tvm.inception_v3(),
                   "ResNet152" : tvm.resnet152(),
                   "Squeezenet 1.1": tvm.squeezenet1_1(),
                   "Swin Transformer Base": tvm.swin_b(),
                   "VGG16" : tvm.vgg16(),
                   "Visinon Transformer Base 16" : tvm.vit_b_16()}
    models_data = {}

    for model_name, model in orig_models.items():
        print('model name')
        models_data[model_name] = gather_model_times(model, input)
    save_model_data_table(models_data)
