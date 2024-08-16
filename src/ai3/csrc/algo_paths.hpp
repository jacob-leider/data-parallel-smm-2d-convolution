#pragma once

#define STRINGIFY(x) #x
// clang-format off
#define CUST(file) STRINGIFY(../custom/file.hpp)
#define CONV2D(file) STRINGIFY(conv2d/file.hpp)
#define MAXPOOL2D(file) STRINGIFY(maxpool2d/file.hpp)
#define LINEAR(file) STRINGIFY(linear/file.hpp)
#define RELU(file) STRINGIFY(relu/file.hpp)
#define AVGPOOL2D(file) STRINGIFY(avgpool2d/file.hpp)
#define ADAPTIVEAVGPOOL2D(file) STRINGIFY(adaptiveavgpool2d/file.hpp)
#define FLATTEN(file) STRINGIFY(flatten/file.hpp)
// clang-format on

#include CUST(conv2d)
#ifdef USE_CUDA_TOOLS
#include CONV2D(gemm_cudnn)
#include CONV2D(guess_cudnn)
#include CONV2D(implicit_gemm_cudnn)
#include CONV2D(implicit_precomp_gemm_cudnn)
#include CONV2D(winograd_cudnn)
#else
#include CONV2D(gemm_plain)
#include CONV2D(guess_plain)
#include CONV2D(implicit_gemm_plain)
#include CONV2D(implicit_precomp_gemm_plain)
#include CONV2D(winograd_plain)
#endif
#ifdef USE_SYCL
#include CONV2D(direct_sycl)
#include CONV2D(smm_sycl)
#else
#include CONV2D(direct_plain)
#include CONV2D(smm_plain)
#endif

#include CUST(linear)
#ifdef USE_CUDA_TOOLS
#include LINEAR(cublas)
#endif
#include LINEAR(plain)

#include CUST(maxpool2d)
#include MAXPOOL2D(plain)

#include CUST(avgpool2d)
#include AVGPOOL2D(plain)

#include CUST(adaptiveavgpool2d)
#include ADAPTIVEAVGPOOL2D(plain)

#include CUST(relu)
#include RELU(plain)

#include CUST(flatten)
#include FLATTEN(plain)
