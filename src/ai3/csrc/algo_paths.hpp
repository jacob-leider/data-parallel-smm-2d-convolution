#pragma once

#define STRINGIFY(x) #x
// clang-format off
#define CONV2D(file) STRINGIFY(conv2d/file.hpp)
#define MAXPOOL2D(file) STRINGIFY(maxpool2d/file.hpp)
#define LINEAR(file) STRINGIFY(linear/file.hpp)
#define RELU(file) STRINGIFY(relu/file.hpp)
#define AVGPOOL2D(file) STRINGIFY(avgpool2d/file.hpp)
#define ADAPTIVEAVGPOOL2D(file) STRINGIFY(adaptiveavgpool2d/file.hpp)
#define FLATTEN(file) STRINGIFY(flatten/file.hpp)
// clang-format on

#include "../custom/conv2d.hpp"
#if USE_CUDA_TOOLS
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
#if defined(USE_SYCL)
#include CONV2D(direct_sycl)
#include CONV2D(smm_sycl)
#else
#include CONV2D(direct_plain)
#include CONV2D(smm_plain)
#endif

#include "../custom/linear.hpp"
#if USE_CUDA_TOOLS
#include LINEAR(cublas)
#endif
#include LINEAR(plain)

#include "../custom/maxpool2d.hpp"
#include MAXPOOL2D(plain)

#include "../custom/avgpool2d.hpp"
#include AVGPOOL2D(plain)

#include "../custom/adaptiveavgpool2d.hpp"
#include ADAPTIVEAVGPOOL2D(plain)

#include "../custom/relu.hpp"
#include RELU(plain)

#include "../custom/flatten.hpp"
#include FLATTEN(plain)
