#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

void launch_rms_norm(
    const void* x, const void* w, void* y,
    int64_t num_rows, int64_t N,
    float eps, bool has_weight,
    at::ScalarType dtype,
    cudaStream_t stream);
