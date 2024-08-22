#pragma once

#include <iostream>
#include <memory>

#if defined USE_CUDA_TOOLS
#include "cuda_utils.hpp"
#include <cudnn.h>
#endif

#if defined USE_SYCL
#include <CL/sycl.hpp>
using namespace cl;
#endif

struct Context {
    Context() {
#if defined USE_CUDA_TOOLS
        CUDNN_CHECK(cudnnCreate(&cudnn_handle));
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
#endif

#if defined USE_SYCL
        sycl_queue = sycl::queue(sycl::default_selector_v);
#endif
    }

#if defined USE_CUDA_TOOLS
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
#endif
#if defined USE_SYCL
    sycl::queue sycl_queue;
#endif

    ~Context() {
#if defined USE_CUDA_TOOLS
        CUDNN_CHECK(cudnnDestroy(cudnn_handle));
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
#endif
    }
};
