#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

void launch_fused_qk_norm(
    const void* q, const void* k,
    const void* qw, const void* kw,
    void* q_out, void* k_out,
    int64_t B, int64_t S, int64_t H, int64_t D,
    float eps,
    at::ScalarType dtype,
    cudaStream_t stream);
