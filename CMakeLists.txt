cmake_minimum_required(VERSION 3.15...3.27)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(LIBRARIES)
set(TARGET_PROPERTIES)
set(COMPILE_OPTIONS)
list(APPEND COMPILE_OPTIONS -Wfatal-errors)
set(LINK_OPTIONS)
set(INCLUDE_DIRECTORIES)
set(LINK_DIRECTORIES)

find_package(pybind11 CONFIG REQUIRED)
list(APPEND LIBRARIES pybind11::headers)

set(CSRC_DIR src/ai3/csrc)
list(APPEND INCLUDE_DIRECTORIES ${CSRC_DIR})

set(CSRC_FILES
    ${CSRC_DIR}/ai3.cpp
)

set(USE_MPS NO)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_library(metal NAMES Metal)
    find_library(mps NAMES MetalPerformanceShaders)
    find_library(foundation NAMES Foundation)
    find_library(coreml NAMES CoreML)
    if(metal AND mps AND foundation AND coreml)
        message(STATUS "Using Metal, Foundation, MetalPerformanceShaders, CoreML")
        list(APPEND CSRC_FILES ${CSRC_DIR}/conv2d/mps.mm
            ${CSRC_DIR}/conv2d/metal.mm)
        list(APPEND LIBRARIES ${metal} ${mps} ${foundation} ${coreml})
        set(USE_MPS YES)
    endif()
endif()

find_package(CUDAToolkit)
set(USE_CUDA_TOOLS NO)
if(CUDAToolkit_FOUND)
    message(STATUS "Using CUDAToolkit")
    set(USE_CUDA_TOOLS YES)
    list(APPEND TARGET_PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED TRUE
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
    )

    list(APPEND LIBRARIES CUDA::cudart CUDA::cublas cudnn)
endif()


find_package(SYCL)
set(USE_SYCL no)
if(SYCL_FOUND)
    message(STATUS "Using SYCL")
    set(USE_SYCL YES)
    set(CMAKE_CXX_COMPILER "${SYCL_COMPILER}")

    list(APPEND INCLUDE_DIRECTORIES ${SYCL_INCLUDE_DIR} ${SYCL_SYCL_INCLUDE_DIR})
    list(APPEND LINK_DIRECTORIES ${SYCL_LIBRARY_DIR})

    separate_arguments(SYCL_CFLAGS)
    list(APPEND COMPILE_OPTIONS ${SYCL_CFLAGS})

    separate_arguments(SYCL_LFLAGS)
    list(APPEND LINK_OPTIONS ${SYCL_LFLAGS})
endif()

pybind11_add_module(_core MODULE ${CSRC_FILES})
set_target_properties(_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    ${TARGET_PROPERTIES}
)
target_compile_options(_core PRIVATE ${COMPILE_OPTIONS})
target_link_options(_core PRIVATE ${LINK_OPTIONS})
target_link_libraries(_core PRIVATE ${LIBRARIES})
target_include_directories(_core PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_core PRIVATE ${LINK_DIRECTORIES})

if(USE_MPS)
    target_compile_definitions(_core PRIVATE USE_MPS)
endif()

if(USE_CUDA_TOOLS)
    target_compile_definitions(_core PRIVATE USE_CUDA_TOOLS)
endif()

if(USE_SYCL)
    target_compile_definitions(_core PRIVATE USE_SYCL)
endif()

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
