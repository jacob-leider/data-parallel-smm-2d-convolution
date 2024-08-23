#pragma once

#define STRINGIFY(x) #x
// clang-format off
#define CUSTOM(file) STRINGIFY(../custom/file.hpp)
#define CONV2D(file) STRINGIFY(conv2d/file.hpp)
#define OBJC_CONV2D(file) STRINGIFY(conv2d/file.h)
#define MAXPOOL2D(file) STRINGIFY(maxpool2d/file.hpp)
#define LINEAR(file) STRINGIFY(linear/file.hpp)
#define RELU(file) STRINGIFY(relu/file.hpp)
#define AVGPOOL2D(file) STRINGIFY(avgpool2d/file.hpp)
#define ADAPTIVEAVGPOOL2D(file) STRINGIFY(adaptiveavgpool2d/file.hpp)
#define FLATTEN(file) STRINGIFY(flatten/file.hpp)
// clang-format on

#include CUSTOM(conv2d)
#if defined USE_CUDA_TOOLS
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
#if defined USE_SYCL
#include CONV2D(direct_sycl)
#include CONV2D(smm_sycl)
#else
#include CONV2D(direct_plain)
#include CONV2D(smm_plain)
#endif
#if defined USE_MPS
#include OBJC_CONV2D(mps)
#else
#include CONV2D(mps)
#endif

#include CUSTOM(linear)
#if defined USE_CUDA_TOOLS
#include LINEAR(cublas)
#else
#include LINEAR(plain)
#endif

#include CUSTOM(maxpool2d)
#include MAXPOOL2D(plain)

#include CUSTOM(avgpool2d)
#include AVGPOOL2D(plain)

#include CUSTOM(adaptiveavgpool2d)
#include ADAPTIVEAVGPOOL2D(plain)

#include CUSTOM(relu)
#include RELU(plain)

#include CUSTOM(flatten)
#include FLATTEN(plain)

#if defined USE_CUDA_TOOLS
const bool USING_CUDA_TOOLS = true;
#else
const bool USING_CUDA_TOOLS = false;
#endif
#if defined USE_MPS
const bool USING_MPS = true;
#else
const bool USING_MPS = false;
#endif
#if defined USE_SYCL
const bool USING_SYCL = true;
#else
const bool USING_SYCL = false;
#endif

#undef STRINGIFY
#undef CUSTOM
#undef CONV2D
#undef MAXPOOL2D
#undef LINEAR
#undef RELU
#undef AVGPOOL2D
#undef ADAPTIVEAVGPOOL2D
#undef FLATTEN
