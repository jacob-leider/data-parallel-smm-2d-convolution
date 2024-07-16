import torchvision.models as tvm
from .helper import wrapped_run
from example.simple_conv2d import ConvNet

def alexnet(runner):
    wrapped_run(tvm.alexnet(),
                (3, 224, 224), runner)

def convnext(runner):
    wrapped_run(tvm.convnext_base(weights='DEFAULT'),
                (3, 224, 224), runner)

def densenet(runner):
    wrapped_run(tvm.DenseNet(), (3, 224, 224), runner)

def efficientnet(runner):
    wrapped_run(tvm.efficientnet_v2_s(
        weights=tvm.EfficientNet_V2_S_Weights.DEFAULT), (3, 224, 224), runner)

def googlenet(runner):
    wrapped_run(tvm.googlenet(
        weights=tvm.GoogLeNet_Weights.DEFAULT),
        (3, 224, 224), runner)

def inception(runner):
    wrapped_run(tvm.inception_v3(
        weights=tvm.Inception_V3_Weights.DEFAULT),
        (3, 224, 224), runner)

def maxvit(runner):
    wrapped_run(tvm.maxvit_t(),
                (3, 224, 224), runner)

def mnasnet(runner):
    wrapped_run(tvm.mnasnet1_0(
        weights=tvm.MNASNet1_0_Weights.DEFAULT),
        (3, 224, 224),
        runner)

def mobilenet(runner):
    wrapped_run(tvm.mobilenet_v3_large(),
                (3, 224, 224), runner)

def regnet(runner):
    wrapped_run(tvm.regnet_y_16gf(),
                (3, 224, 224), runner)

def resnet(runner):
    wrapped_run(tvm.resnet152(),
                (3, 224, 224), runner)

def shufflenet(runner):
    wrapped_run(tvm.shufflenet_v2_x2_0(),
                (3, 224, 224), runner)

def simple_created(runner):
    wrapped_run(ConvNet(), (3, 224, 224), runner)

def squeezenet(runner):
    wrapped_run(tvm.squeezenet1_1(),
                (3, 224, 224), runner)

def swintransformer(runner):
    wrapped_run(tvm.swin_b(),
                (3, 224, 224), runner)

def vgg16(runner):
    wrapped_run(tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT),
                (3, 224, 224), runner)

def visiontransformer(runner):
    wrapped_run(tvm.vit_b_16(),
                (3, 224, 224), runner)
