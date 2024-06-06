import torch
import ai3
import torchvision
from bench import predict_show_time

def run():
    print('VGG')
    input_data = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        pytorch_vgg16 = torchvision.models.vgg16()
        pytorch_vgg16.eval()
        predict_show_time(pytorch_vgg16, input_data, "pytorch")

        # opt_pytorch_vgg16 = torch.compile(pytorch_vgg16)
        # show_time(opt_pytorch_vgg16, input_data, "pytorch compiled")

        ai3_model = ai3.optimize(pytorch_vgg16)
        predict_show_time(ai3_model, input_data, "ai3")


if __name__ == "__main__":
    run()
