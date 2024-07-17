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
#if USE_CUDNN
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
#if defined(USE_SYCL) && defined(CONV2D_SYCL)
#include CONV2D(direct_sycl)
#include CONV2D(smm_sycl)
#else
#include CONV2D(direct_plain)
#include CONV2D(smm_plain)
#endif

#include "../custom/linear.hpp"
#if defined(USE_SYCL) && defined(LINEAR_SYCL)
#include LINEAR(sycl)
#else
#include LINEAR(plain)
#endif

#include "../custom/maxpool2d.hpp"
#if defined(USE_SYCL) && defined(MAXPOOL2D_SYCL)
#include MAXPOOL2D(sycl)
#else
#include MAXPOOL2D(plain)
#endif

#include "../custom/avgpool2d.hpp"
#if defined(USE_SYCL) && defined(AVGPOOL2D_SYCL)
#include AVGPOOL2D(sycl)
#else
#include AVGPOOL2D(plain)
#endif

#include "../custom/adaptiveavgpool2d.hpp"
#if defined(USE_SYCL) && defined(ADAPTIVEAVGPOOL2D_SYCL)
#include ADAPTIVEAVGPOOL2D(sycl)
#else
#include ADAPTIVEAVGPOOL2D(plain)
#endif

#include "../custom/relu.hpp"
#if defined(USE_SYCL) && defined(RELU_SYCL)
#include RELU(sycl)
#else
#include RELU(plain)
#endif

#include "../custom/flatten.hpp"
#if defined(USE_SYCL) && defined(FLATTEN_SYCL)
#include FLATTEN(sycl)
#else
#include FLATTEN(plain)
#endif
