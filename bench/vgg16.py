import torch
import torchvision
from torchvision.models.vgg import VGG16_Weights
import ai3
import time


def run():
    input_data = torch.randn(1, 3, 224, 224)

    with torch.inference_mode():
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.eval()

        start_time = time.time()
        _ = model(input_data)
        end_time = time.time()

        inference_time = end_time - start_time
        print("Inference time original:", inference_time, "seconds")

        ai3.optimize(model, replace=True)
        start_time = time.time()
        _ = model(input_data)
        end_time = time.time()
        inference_time = end_time - start_time
        print("Inference time ai3:", inference_time, "seconds")


if __name__ == "__main__":
    run()
