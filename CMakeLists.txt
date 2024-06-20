cmake_minimum_required(VERSION 3.15...3.27)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(core MODULE src/ai3/csrc/ai3.cpp WITH_SOABI)
target_link_libraries(core PRIVATE pybind11::headers)

find_package(SYCL)
if(SYCL_FOUND)
    message(STATUS "Using SYCL")
    set(CMAKE_CXX_COMPILER "${SYCL_COMPILER}")
    include_directories(${SYCL_INCLUDE_DIR})
    include_directories(${SYCL_SYCL_INCLUDE_DIR})
    link_directories(${SYCL_LIBRARY_DIR})
    separate_arguments(SYCL_CFLAGS)
    separate_arguments(SYCL_LFLAGS)
    target_compile_options(core PRIVATE "${SYCL_CFLAGS}")
    target_link_options(core PRIVATE "${SYCL_LFLAGS}")
    add_definitions(-DUSE_SYCL)
else()
    find_program(ACPP acpp)
    if(ACPP)
        message(STATUS "Using acpp")
        set(CMAKE_CXX_COMPILER ${ACPP})
        message(STATUS ${CMAKE_CXX_COMPILER})
        add_definitions(-DUSE_SYCL)
    endif()
endif()

install(TARGETS core DESTINATION ${SKBUILD_PROJECT_NAME})

