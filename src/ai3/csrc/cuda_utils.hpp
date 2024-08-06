#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

template <typename T> cudnnDataType_t cudnn_data_type();

template <> cudnnDataType_t cudnn_data_type<float>() {
    return CUDNN_DATA_FLOAT;
}

template <> cudnnDataType_t cudnn_data_type<double>() {
    return CUDNN_DATA_DOUBLE;
}

template <typename T> cudaDataType cublas_data_type();

template <> cudaDataType cublas_data_type<float>() { return CUDA_R_32F; }

template <> cudaDataType cublas_data_type<double>() { return CUDA_R_64F; }

#define CUDA_CHECK(status)                                                     \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(status)              \
                  << std::endl;                                                \
        exit(1);                                                               \
    }

#define CUBLAS_CHECK(status)                                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS error: " << status << std::endl;                  \
        exit(1);                                                               \
    }

#define CUDNN_CHECK(status)                                                    \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status)            \
                  << std::endl;                                                \
        exit(1);                                                               \
    }
