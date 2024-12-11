import torch
import numpy as np
import ai3
import time
import torchvision.models as models



def time_algorithm_for_resnet(algo=None, num_samples=100):
    """
    Tracks the inference time of a pretained ResNet model using a user-selected
    convolution algorithm.

    Args:
        algo (str): Name of the algorithm to use for conv2d. If `None` this uses
        pytorch's default conv2d implementation.
        num_samples (int): Number of samples to train on.
    """
    net = models.resnet50(pretrained=True)
    if algo != None:
      ai3.swap_conv2d(net, algo)
    
    # ResNet expects images of dimensions 3 x 224 x 224
    dims = (num_samples, 3, 224, 224)
    np.random.seed(0) # For reproducability
    data = np.random.random(dims)
    data = np.array(data, dtype=np.float32)
    data = torch.from_numpy(data)

    t1 = time.time()
    y = net(data)
    t2 = time.time()
    dt = t2 - t1
    return dt


if __name__ == '__main__':
  # Check for SYCL.
  if ai3.using_sycl():
      print("Using SYCL.")
  else:
      print("NOT using SYCL.")

  torch_time = time_algorithm_for_resnet(num_samples=100)
  smm_time = time_algorithm_for_resnet(algo='custom', num_samples=100)

  print(f"Inference time by torch default: {torch_time}s")
  print(f"Inference time by custom SMM:    {smm_time}s")
