*ai3*
=====

The *ai3* (Algorithmic Innovations for Accelerated Implementations of
Artificial Intelligence) framework provides easy-to-use fine-grain algorithmic
control over an existing *DNN*. The framework contains built-in high performance
implementations of common deep learning operations and methods by which users
can implement their own algorithms in *C++*. The framework incurs no additional
performance overhead, meaning that performance depends solely on the algorithms
chosen by the user.

.. TODO fill this out once published

.. _doc: http://www.example.com
.. |doc| replace:: **Documentation**
.. _ins_cus: http://www.example.com
.. |ins_cus| replace:: **Installation for Customization**

|doc|_

**Installation:** ``pip install aix3``

|ins_cus|_

The framework currently features two methods for algorithmic swapping. *swap_backend*
which swaps every module type of a *DNN* returning an object completely managed
by the framework and *swap_conv2d* which swaps convolution operations out of the
existing *DNN*.

*swap_conv2d*
~~~~~~~~~~~~~
Swaps, in-place, conv2d operations out of the existing *DNN* for an implementation of
the user specified algorithm. If no *AlgorithmicSelector* is given then the default
algorithm decided by the framework are used.

Example:
    Swaps the first *conv2d* operation for an implementation of direct convolution
    and the second *conv2d* operation for an implementation of *SMM* convolution

        >>> def auto_selector(orig: torch.nn.Conv2d, input_shape: Sequence[int]) -> str:
        ...     out_channels = orig.weight.shape[0]
        ...     if (out_channels < 50 and
        ...         input_shape[0] < 50 and
        ...         input_shape[1] > 150 and
        ...         input_shape[2] > 150):
        ...         return 'direct'
        ...     return 'smm'
        ...
        >>> input_data = torch.randn(10, 3, 224, 224)
        >>> orig = ConvNet()
        >>> orig_out = orig(input_data)
        >>> ai3.swap_conv2d(orig, ['direct', 'smm'])
        >>> sc_out = orig(input_data)
        >>> torch.allclose(torch_out, sc_out, atol=1e-6)
        True

*swap_backend*
~~~~~~~~~~~~~
Swaps, in-place, conv2d operations out of the existing *DNN* for an implementation of
the user specified algorithm. If no *AlgorithmicSelector* is given then the default
algorithm decided by the framework are used.

Example:
    Swaps the first *conv2d* operation for an implementation of direct convolution
    and the second *conv2d* operation for an implementation of *SMM* convolution

        >>> def auto_selector(orig: torch.nn.Conv2d, input_shape: Sequence[int]) -> str:
        ...     out_channels = orig.weight.shape[0]
        ...     if (out_channels < 50 and
        ...         input_shape[0] < 50 and
        ...         input_shape[1] > 150 and
        ...         input_shape[2] > 150):
        ...         return 'direct'
        ...     return 'smm'
        ...
        >>> input_data = torch.randn(10, 3, 224, 224)
        >>> orig = ConvNet()
        >>> orig_out = orig(input_data)
        >>> ai3.swap_conv2d(orig, ['direct', 'smm'])
        >>> sc_out = orig(input_data)
        >>> torch.allclose(torch_out, sc_out, atol=1e-6)
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
   * - *none*
     - |y|
     - |y|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
   * - *sycl*
     - |y|
     - |y|
     - |n|
     - |n|
     - |n|
     - |n|
     - |n|
   * - *cudnn*
     - |n|
     - |n|
     - |y|
     - |y|
     - |y|
     - |y|
     - |y|
   * - *cublas*
     - |n|
     - |n|
     - |y|
     - |y|
     - |y|
     - |y|
     - |y|

Linear
~~~~~~
.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1
   :align: left

   * - Algorithm
     - gemm
   * - *none*
     - |y|
   * - *sycl*
     - |n|
   * - *cudnn*
     - |n|
   * - *cublas*
     - |y|


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
