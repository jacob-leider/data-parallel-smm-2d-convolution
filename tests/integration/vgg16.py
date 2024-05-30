import torch
import torchvision
from torchvision.models.vgg import VGG16_Weights
import ai3
from tests import compare_tensors

# TODO we are missing flatten for vgg
# flatten is called in the forward but is not a class attribute
def run():
    # input_data = torch.randn(1, 3, 224, 224)
    # with torch.inference_mode():
    #     model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
    #     model.eval()
    #     original = model(input_data)
    #     ai3_model = ai3.optimize(model)
    #     output = ai3_model.predict(input_data)
    # compare_tensors(output, original, "vgg16")
    pass


if __name__ == "__main__":
    run()
