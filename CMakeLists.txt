cmake_minimum_required(VERSION 3.15...3.27)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

set(CSRC_DIR src/ai3/csrc)
set(CSRC_FILES ${CSRC_DIR}/ai3.cpp)

set(USE_MPS NO)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_library(metal NAMES Metal)
    find_library(mps NAMES MetalPerformanceShaders)
    find_library(foundation NAMES Foundation)
    find_library(coreml NAMES CoreML)
    if(metal AND mps AND foundation AND coreml)
        list(APPEND CSRC_FILES ${CSRC_DIR}/conv2d/mps.mm)
        set(USE_MPS YES)
    endif()
endif()

pybind11_add_module(core MODULE ${CSRC_FILES})
set_property(TARGET core PROPERTY CXX_STANDARD 17)
set_property(TARGET core PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)
target_compile_options(core PRIVATE -Wfatal-errors)
target_link_libraries(core PRIVATE pybind11::headers)
target_include_directories(core PRIVATE ${CSRC_DIR})

if(USE_MPS)
    message(STATUS "Using Metal, Foundation, MetalPerformanceShaders, CoreML")
    target_link_libraries(core PRIVATE ${metal} ${mps} ${foundation} ${coreml})
    target_compile_definitions(core PRIVATE USE_MPS)
endif()


find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "Using CUDAToolkit")
    set_target_properties(core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED TRUE
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
    )

    target_link_libraries(core PRIVATE CUDA::cudart CUDA::cublas cudnn)
    target_compile_definitions(core PRIVATE USE_CUDA_TOOLS)
endif()



find_package(SYCL)
if(SYCL_FOUND)
    message(STATUS "Using SYCL")
    set(CMAKE_CXX_COMPILER "${SYCL_COMPILER}")

    target_include_directories(core PRIVATE ${SYCL_INCLUDE_DIR})
    target_include_directories(core PRIVATE ${SYCL_SYCL_INCLUDE_DIR})
    target_link_directories(core PRIVATE ${SYCL_LIBRARY_DIR})

    separate_arguments(SYCL_CFLAGS)
    target_compile_options(core PRIVATE "${SYCL_CFLAGS}")

    separate_arguments(SYCL_LFLAGS)
    target_link_options(core PRIVATE "${SYCL_LFLAGS}")

    target_compile_definitions(core PRIVATE USE_SYCL)
elseif(FIND_ACPP)
    find_program(ACPP acpp)
    if(ACPP)
        message(STATUS "Using acpp")
        set(CMAKE_CXX_COMPILER ${ACPP})
        target_compile_definitions(core PRIVATE USE_SYCL)
    endif()
endif()

install(TARGETS core DESTINATION ${SKBUILD_PROJECT_NAME})
