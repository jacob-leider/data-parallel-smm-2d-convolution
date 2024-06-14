#pragma once

#ifdef CONV2D_USER
#define CONV2D_PATH CONV2D_USER
#elif defined(USE_SYCL) && defined(CONV2D_SYCL)
#define CONV2D_PATH "conv2d_sycl.hpp"
#else
#define CONV2D_PATH "conv2d_plain.hpp"
#endif

#ifdef LINEAR_USER
#define CONV2D_PATH LINEAR_USER
#elif defined(USE_SYCL) && defined(LINEAR_SYCL)
#define LINEAR_PATH "linear_sycl.hpp"
#else
#define LINEAR_PATH "linear_plain.hpp"
#endif

#ifdef MAXPOOL2D_USER
#define CONV2D_PATH MAXPOOL2D_USER
#elif defined(USE_SYCL) && defined(MAXPOOL2D_SYCL)
#define MAXPOOL2D_PATH "maxpool2d_sycl.hpp"
#else
#define MAXPOOL2D_PATH "maxpool2d_plain.hpp"
#endif

#ifdef AVGPOOL2D_USER
#define CONV2D_PATH AVGPOOL2D_USER
#elif defined(USE_SYCL) && defined(AVGPOOL2D_SYCL)
#define AVGPOOL2D_PATH "avgpool2d_sycl.hpp"
#else
#define AVGPOOL2D_PATH "avgpool2d_plain.hpp"
#endif

#ifdef ADAPTIVEAVGPOOL2D_USER
#define CONV2D_PATH ADAPTIVEAVGPOOL2D_USER
#elif defined(USE_SYCL) && defined(ADAPTIVEAVGPOOL2D_SYCL)
#define ADAPTIVEAVGPOOL2D_PATH "adaptiveavgpool2d_sycl.hpp"
#else
#define ADAPTIVEAVGPOOL2D_PATH "adaptiveavgpool2d_plain.hpp"
#endif

#ifdef RELU_USER
#define CONV2D_PATH RELU_USER
#elif defined(USE_SYCL) && defined(RELU_SYCL)
#define RELU_PATH "relu_sycl.hpp"
#else
#define RELU_PATH "relu_plain.hpp"
#endif

#ifdef FLATTEN_USER
#define CONV2D_PATH FLATTEN_USER
#elif defined(USE_SYCL) && defined(FLATTEN_SYCL)
#define FLATTEN_PATH "flatten_sycl.hpp"
#else
#define FLATTEN_PATH "flatten_plain.hpp"
#endif

#include "tensor.hpp"
#include "utils.hpp"
