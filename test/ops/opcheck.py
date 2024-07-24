import torch

PASS_MES = 'Passed opcheck for '


def convolution_samples():
    input_size = (1, 3, 224, 224)
    kernel_size = (3, 3, 5, 5)
    return [(
        torch.randn(input_size, requires_grad=grad),
        torch.randn(kernel_size, requires_grad=grad),
        torch.randn(kernel_size[0], requires_grad=grad),
        1, 1, 1, 1, 1, 1, 0, 1, 'default') for grad in [False, True]]


def conv2d():
    for samp in convolution_samples():
        torch.library.opcheck(torch.ops.ai3.conv2d, samp)  # type: ignore
    print(PASS_MES + 'conv2d')
