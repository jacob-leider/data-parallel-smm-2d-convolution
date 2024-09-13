.. _repo: https://github.com/KLab-ai3/ai3
.. |repo| replace:: **Source Code**
.. _custom: https://github.com/KLab-ai3/ai3/tree/main/src/ai3/custom
.. |custom| replace:: custom
.. _custom_cmake: https://github.com/KLab-ai3/ai3/tree/main/src/ai3/cmake/custom.cmake
.. |custom_cmake| replace:: *custom.cmake*
.. _doc: https://klab-ai3.github.io/ai3
.. |doc| replace:: **Documentation**
.. _model_zoo: https://github.com/KLab-ai3/ai3/tree/main/model_zoo/models.py
.. |model_zoo| replace:: *model_zoo*
.. |name| replace:: *ai3*
.. |pkg_name| replace:: *aithree*

|name|
======

The |name| (Algorithmic Innovations for Accelerated Implementations of
Artificial Intelligence) framework provides easy-to-use fine-grain algorithmic
control over an existing *DNN*. |name| contains built-in high performance
implementations of common deep learning operations and methods by which users
can implement their own algorithms in *C++*. |name| incurs no additional
performance overhead, meaning that performance depends solely on the algorithms
chosen by the user.

|doc|_ |repo|_

Installation
------------

**Default Implementations:** *pip install* |pkg_name|

**Custom Implementations:**
   1. Download the source code
   2. Create an implementation with the operations defined in |custom|_
   3. If needed, configure the build process with |custom_cmake|_
   4. ``pip install <path to source code>``

The framework currently features two methods for algorithmic swapping. `swap_backend`
which swaps every module type of a *DNN* returning an object completely managed
by |name| and `swap_conv2d` which swaps convolution operations out of the
existing *DNN*.

*swap_conv2d*
~~~~~~~~~~~~~
Swaps, in-place, *conv2d* operations out of the existing *DNN* for an implementation of
the user specified algorithm. After swapping, the same *DNN* can still be trained
and compiled. If no `AlgorithmicSelector` is given then the default
algorithm decided by the framework are used.

Example:
    Swaps the first *conv2d* operation for an implementation of direct convolution
    and the second *conv2d* operation for an implementation of *SMM* convolution

    >>> input_data = torch.randn(10, 3, 224, 224)
    >>> orig = ConvNet()
    >>> orig_out = orig(input_data)
    >>> ai3.swap_conv2d(orig, ['direct', 'smm'])
    >>> sc_out = orig(input_data)
    >>> torch.allclose(orig_out, sc_out, atol=1e-6)
    True

*swap_backend*
~~~~~~~~~~~~~~
Swaps every module in an exsiting *DNN* for an implementation
of the user specified algorithm returning
a `Model` completly managed by the framework.

Algorithmic selection is performed by passing a mapping from strings
containing names of the operations to swap to a `AlgorithmicSelector`.
If no `AlgorithmicSelector` is passed for a given operation then the default
algorithm decided by the framework are used.

Example:
Swaps the first *conv2d* operation for an implementation of direct convolution
and the second *conv2d* operation for an implementation of *SMM* convolution

    >>> def auto_selector(orig: torch.nn.Conv2d, input_shape) -> str:
    ...     out_channels = orig.weight.shape[0]
    ...     if (out_channels < 50 and
    ...         input_shape[1] < 50 and
    ...         input_shape[2] > 150 and
    ...         input_shape[3] > 150):
    ...         return 'direct'
    ...     return 'smm'
    ...
    >>> input_data = torch.randn(1, 3, 224, 224)
    >>> vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    >>> vgg16 = vgg16.eval()
    >>> with torch.inference_mode():
    ...     torch_out = vgg16(input_data)
    ...     model: ai3.Model = ai3.swap_backend(vgg16, {"conv2d": auto_selector,
    ...                                                 "maxpool2d": "default"},
    ...                                         sample_input_shape=(1, 3, 224, 224))
    ...     sb_out = model(input_data)
    ...     torch.allclose(torch_out, sb_out, atol=1e-4)
    True

Supported Operations, their Algorithms, and Acceleration Platform Compatibility
-------------------------------------------------------------------------------

.. |y| unicode:: U+2713
.. |n| unicode:: U+2717

*2D* Convolution
~~~~~~~~~~~~~~~~

The *guess* algorithm uses the algorithm returned by `cudnnGetConvolutionForwardAlgorithm_v7`.

.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
     - *smm*
     - *gemm*
     - *implicit precomp gemm*
     - *implicit gemm*
     - *winograd*
     - *guess*
     - some
   * - *none*
     - |y|
     - |y|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|
   * - *sycl*
     - |y|
     - |y|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|
   * - *cudnn*
     - |n|
     - |n|
     - |y|
     - |y|
     - |y|
     - |y|
     - |y|
     - |y|
   * - *cublas*
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
   * - *mps*
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|
   * - *metal*
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
     - |y|

Linear
~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - *gemm*
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |y|
   * - *mps*
     - |n|
   * - *metal*
     - |n|


*2D* MaxPool
~~~~~~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|

*2D* AvgPool
~~~~~~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|

*2D* AdaptiveAvgPool
~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|

*ReLU*
~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|


Flatten
~~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - direct
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |n|
   * - *mps*
     - |n|
   * - *metal*
     - |n|
