#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

void launch_silu_gate(
    const void* x1, const void* x3, void* out,
    int64_t total_elements,
    at::ScalarType dtype,
    cudaStream_t stream);
