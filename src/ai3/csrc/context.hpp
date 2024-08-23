#pragma once

#include "utils.hpp"

#if defined USE_CUDA_TOOLS
#include "cuda_utils.hpp"
#endif

#if defined USE_SYCL
#include <CL/sycl.hpp>
using namespace cl;
#endif

#if defined USE_MPS
void *gen_mps_graph_device(void);
#endif

/**
 * @brief Initializes and caches data useful in acceleration platforms
 */
class Context {
  public:
    /**
     * @brief Returns the cached `MPSGraphDevice`, initializing it if
     * necessary
     */
#if defined USE_MPS
    inline static void *mps_graph_device() {
        if (mps_device_init) {
            return mps_g_device;
        }
        mps_g_device = gen_mps_graph_device();
        mps_device_init = true;
        return mps_g_device;
    }
#else
    inline static void *mps_graph_device() {
        errs::bail("trying to get mps graph device when mps not supported");
    }
#endif

    /**
     * @brief Returns the cached `cudnnHandle_t`, initializing it if
     * necessary
     */
#if defined USE_CUDA_TOOLS
    inline static void *cudnn_handle_t() {
        if (cudnn_init) {
            return cudnn_handle;
        }
        CUDNN_CHECK(cudnnCreate(&cudnn_handle));
        cudnn_init = true;
        return cudnn_handle;
    }
#else
    inline static void *cudnn_handle_t() {
        errs::invalid_context_access("cudnn handle", "cudnn");
    }
#endif

    /**
     * @brief Returns the cached `cublasHandle_t`, initializing it if
     * necessary
     */
#if defined USE_CUDA_TOOLS
    inline static void *cublas_handle_t() {
        if (cublas_init) {
            return cublas_handle;
        }
        CUBLAS_CHECK(cublasCreate(&cublas_handle))
        cublas_init = true;
        return cublas_handle;
    }
#else
    inline static void *cublas_handle_t() {
        errs::invalid_context_access("cublas handle", "cublas");
    }
#endif

    /**
     * @brief Returns the cached `sycl::queue`, initializing it if
     * necessary
     */
#if defined USE_SYCL
    inline static void *sycl_queue() {
        if (sycl_init) {
            return &sycl_q;
        }
        sycl_q = sycl::queue(sycl::default_selector_v);
        sycl_init = true;
        return &sycl_q;
    }
#else
    inline static void *sycl_queue() {
        errs::invalid_context_access("sycl queue", "sycl");
    }
#endif

    ~Context() {
#if defined USE_CUDA_TOOLS
        if (cudnn_init) {
            CUDNN_CHECK(cudnnDestroy(cudnn_handle));
        }
        if (cublas_init) {
            CUBLAS_CHECK(cublasDestroy(cublas_handle));
        }
#endif
    }

  private:
#if defined USE_MPS
    inline static void *mps_g_device = nullptr;
    inline static bool mps_device_init = false;
#endif

#if defined USE_CUDA_TOOLS
    inline static cudnnHandle_t cudnn_handle;
    inline static bool cudnn_init = false;
    inline static cublasHandle_t cublas_handle;
    inline static bool cublas_init = false;
#endif

#if defined USE_SYCL
    inline static sycl::queue sycl_q;
    inline static bool sycl_init = false;
#endif
};
