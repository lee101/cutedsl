cmake_minimum_required(VERSION 3.20)
project(cutezimage_cuda LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# CMake alternative to setup_cuda.py – useful when embedding in a larger build.
# Produces a shared library loadable by torch.ops.load_library().
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Multi-arch fat binary for A40/H100/RTX5090
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120")

# Find Python and PyTorch
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Gather sources
set(SOURCES
    cute_kernels_ext.cpp
    cute_rms_norm.cu
    cute_silu_gate.cu
    cute_qk_norm.cu
)

add_library(cutezimage_C SHARED ${SOURCES})

target_compile_options(cutezimage_C PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --maxrregcount=64
        --ptxas-options=-v
    >
)

target_link_libraries(cutezimage_C PRIVATE
    ${TORCH_LIBRARIES}
    cublas
)

set_target_properties(cutezimage_C PROPERTIES
    OUTPUT_NAME "_C"
    PREFIX ""
    SUFFIX ".so"
)

# Install next to the cutezimage Python package
install(TARGETS cutezimage_C
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}"
)
