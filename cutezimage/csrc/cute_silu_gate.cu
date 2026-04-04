/**
 * cute_silu_gate.cu
 * Fused SiLU-gated multiply: out = silu(x1) * x3
 *
 * Uses vectorized 128-bit loads/stores (bfloat162 / half2 / float4).
 * Avoids any intermediate buffer beyond the output tensor.
 */

#include "cute_silu_gate.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.f + expf(-x));
}

// ---------------------------------------------------------------------------
// float32 – vectorized with float4
// ---------------------------------------------------------------------------

__global__ void silu_gate_f32_kernel(
        const float* __restrict__ x1,
        const float* __restrict__ x3,
        float*       __restrict__ out,
        int64_t N)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    const int64_t N4     = N >> 2;

    const float4* x1v = reinterpret_cast<const float4*>(x1);
    const float4* x3v = reinterpret_cast<const float4*>(x3);
    float4*       ov  = reinterpret_cast<float4*>(out);

    for (int64_t i = tid; i < N4; i += stride) {
        float4 a = x1v[i], b = x3v[i];
        ov[i] = {silu_f(a.x)*b.x, silu_f(a.y)*b.y,
                 silu_f(a.z)*b.z, silu_f(a.w)*b.w};
    }
    // scalar tail
    for (int64_t i = (N4 << 2) + tid; i < N; i += stride)
        out[i] = silu_f(x1[i]) * x3[i];
}

// ---------------------------------------------------------------------------
// BF16 – vectorized with bfloat162 (2 elements)
// ---------------------------------------------------------------------------

__global__ void silu_gate_bf16_kernel(
        const __nv_bfloat16* __restrict__ x1,
        const __nv_bfloat16* __restrict__ x3,
        __nv_bfloat16*       __restrict__ out,
        int64_t N)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    const int64_t N2     = N >> 1;

    const __nv_bfloat162* x1v = reinterpret_cast<const __nv_bfloat162*>(x1);
    const __nv_bfloat162* x3v = reinterpret_cast<const __nv_bfloat162*>(x3);
    __nv_bfloat162*       ov  = reinterpret_cast<__nv_bfloat162*>(out);

    for (int64_t i = tid; i < N2; i += stride) {
        float2 a = __bfloat1622float2(x1v[i]);
        float2 b = __bfloat1622float2(x3v[i]);
        ov[i] = __float22bfloat162_rn({silu_f(a.x)*b.x, silu_f(a.y)*b.y});
    }
    if ((N & 1) && tid == 0)
        out[N-1] = __float2bfloat16(silu_f(__bfloat162float(x1[N-1]))
                                    * __bfloat162float(x3[N-1]));
}

// ---------------------------------------------------------------------------
// FP16 – same structure as BF16
// ---------------------------------------------------------------------------

__global__ void silu_gate_f16_kernel(
        const __half* __restrict__ x1,
        const __half* __restrict__ x3,
        __half*       __restrict__ out,
        int64_t N)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    const int64_t N2     = N >> 1;

    const half2* x1v = reinterpret_cast<const half2*>(x1);
    const half2* x3v = reinterpret_cast<const half2*>(x3);
    half2*       ov  = reinterpret_cast<half2*>(out);

    for (int64_t i = tid; i < N2; i += stride) {
        float2 a = __half22float2(x1v[i]);
        float2 b = __half22float2(x3v[i]);
        ov[i] = __float22half2_rn({silu_f(a.x)*b.x, silu_f(a.y)*b.y});
    }
    if ((N & 1) && tid == 0)
        out[N-1] = __float2half(silu_f(__half2float(x1[N-1]))
                                * __half2float(x3[N-1]));
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void launch_silu_gate(
        const void* x1, const void* x3, void* out,
        int64_t total_elements,
        at::ScalarType dtype,
        cudaStream_t stream)
{
    const int threads = 256;
    // Cap grid at 65535 blocks; each block handles multiple tiles via stride loop
    const int64_t max_blocks = 65535;
    int64_t blocks;

    switch (dtype) {
        case at::ScalarType::Float:
            blocks = std::min(max_blocks, (total_elements / 4 + threads - 1) / threads + 1);
            silu_gate_f32_kernel<<<(int)blocks, threads, 0, stream>>>(
                (const float*)x1, (const float*)x3, (float*)out, total_elements);
            break;
        case at::ScalarType::BFloat16:
            blocks = std::min(max_blocks, (total_elements / 2 + threads - 1) / threads + 1);
            silu_gate_bf16_kernel<<<(int)blocks, threads, 0, stream>>>(
                (const __nv_bfloat16*)x1, (const __nv_bfloat16*)x3,
                (__nv_bfloat16*)out, total_elements);
            break;
        case at::ScalarType::Half:
            blocks = std::min(max_blocks, (total_elements / 2 + threads - 1) / threads + 1);
            silu_gate_f16_kernel<<<(int)blocks, threads, 0, stream>>>(
                (const __half*)x1, (const __half*)x3, (__half*)out, total_elements);
            break;
        default:
            TORCH_CHECK(false, "fused_silu_gate_cuda: unsupported dtype");
    }
}
