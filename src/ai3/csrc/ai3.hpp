/**
 * @mainpage
 * @section intro_sec Introduction
 *
 * The *ai3* *C++* library for creating custom
 * implementations of common deep learning operations.
 * The library provides a simple `Tensor` class
 * and function declarations to be completed by the user.
 *
 * @section install_sec Installation For Customization
 *
 * -# Clone the source repository
 *
 * -# Complete an implemenation of the operation desired. Function
 * declarations are in the `custom` directory
 *
 * -# Use the *bool* associated with each function to control whether
 *  the custom implementation is used by default.
 *
 * -# Install the package
 * @code
 * $ pip install <path to local clone of this repository>
 * @endcode
 *
 *
 * @defgroup custom
 *
 * Operations available for customization create implementations in the
 * <a href="@SRC_REPO/custom">custom</a> directory.
 */

#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "utils.hpp"
