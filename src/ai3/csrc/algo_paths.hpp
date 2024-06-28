#pragma once

#include "../custom/conv2d.hpp"
#if defined(USE_SYCL) && defined(CONV2D_SYCL)
#include "direct_conv2d_sycl.hpp"
#include "smm_conv2d_sycl.hpp"
#else
#include "direct_conv2d_plain.hpp"
#include "smm_conv2d_plain.hpp"
#endif

#include "../custom/linear.hpp"
#if defined(USE_SYCL) && defined(LINEAR_SYCL)
#include "linear_sycl.hpp"
#else
#include "linear_plain.hpp"
#endif

#include "../custom/maxpool2d.hpp"
#if defined(USE_SYCL) && defined(MAXPOOL2D_SYCL)
#include "maxpool2d_sycl.hpp"
#else
#include "maxpool2d_plain.hpp"
#endif

#include "../custom/avgpool2d.hpp"
#if defined(USE_SYCL) && defined(AVGPOOL2D_SYCL)
#include "avgpool2d_sycl.hpp"
#else
#include "avgpool2d_plain.hpp"
#endif

#include "../custom/adaptiveavgpool2d.hpp"
#if defined(USE_SYCL) && defined(ADAPTIVEAVGPOOL2D_SYCL)
#include "adaptiveavgpool2d_sycl.hpp"
#else
#include "adaptiveavgpool2d_plain.hpp"
#endif

#include "../custom/relu.hpp"
#if defined(USE_SYCL) && defined(RELU_SYCL)
#include "relu_sycl.hpp"
#else
#include "relu_plain.hpp"
#endif

#include "../custom/flatten.hpp"
#if defined(USE_SYCL) && defined(FLATTEN_SYCL)
#include "flatten_sycl.hpp"
#else
#include "flatten_plain.hpp"
#endif
