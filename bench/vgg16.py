import torch
import ai3
import time
from tests.integration.vgg16 import VGG16


def run():
    input_data = torch.randn(3, 224, 224)
    with torch.inference_mode():
        pytorch_vgg16 = VGG16()
        pytorch_vgg16.eval()
        start_time = time.time()
        _ = pytorch_vgg16(input_data)
        end_time = time.time()
        inference_time = end_time - start_time
        print("Inference time pytorch:", inference_time, "seconds")
        ai3_model = ai3.optimize(pytorch_vgg16)
        start_time = time.time()
        _ = ai3_model.predict(input_data)
        end_time = time.time()
        inference_time = end_time - start_time
        print("Inference time ai3:", inference_time, "seconds")


if __name__ == "__main__":
    run()
