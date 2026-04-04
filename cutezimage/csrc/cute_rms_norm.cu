/**
 * cute_rms_norm.cu
 * Vectorized RMS LayerNorm for BF16/FP16/FP32.
 *
 * Design goals:
 *   - One CUDA block per row; blockDim.x chosen by caller for occupancy.
 *   - Warp-level reductions via __shfl_down_sync (zero shared-mem bank conflicts).
 *   - Vectorized 128-bit loads using float4 / __nv_bfloat162 / half2 reinterpret.
 *   - Compiled for sm_86 (A40), sm_90 (H100), sm_100 (H200), sm_120 (RTX5090).
 */

#include "cute_rms_norm.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Warp / block reduction helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[32];  // one slot per warp (max 32 warps / block)
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    const int n_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < n_warps) ? smem[lane] : 0.f;
    v = warp_reduce_sum(v);  // only first warp participates
    return v;
}

// ---------------------------------------------------------------------------
// Generic scalar kernel  (float32)
// ---------------------------------------------------------------------------

__global__ void rms_norm_f32_kernel(
        const float* __restrict__ x,
        const float* __restrict__ w,   // may be nullptr
        float*       __restrict__ y,
        int64_t N, float eps, bool has_weight)
{
    const int64_t row    = blockIdx.x;
    const int64_t stride = blockDim.x;
    const float*  xrow   = x + row * N;
    float*        yrow   = y + row * N;

    // Vectorized load: process 4 floats at a time when aligned
    float sq_sum = 0.f;
    int64_t i = threadIdx.x;
    for (; i + 3 < N; i += stride * 4) {
        float4 v = *reinterpret_cast<const float4*>(xrow + i);
        sq_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (; i < N; i += stride) {
        float v = xrow[i];
        sq_sum += v * v;
    }
    sq_sum = block_reduce_sum(sq_sum);
    const float rrms = rsqrtf(sq_sum / (float)N + eps);

    // Apply
    i = threadIdx.x;
    if (has_weight) {
        for (; i < N; i += stride)
            yrow[i] = xrow[i] * rrms * w[i];
    } else {
        for (; i < N; i += stride)
            yrow[i] = xrow[i] * rrms;
    }
}

// ---------------------------------------------------------------------------
// BF16 kernel: vectorized via __nv_bfloat162 (2 elements per instruction)
// ---------------------------------------------------------------------------

__global__ void rms_norm_bf16_kernel(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ w,
        __nv_bfloat16*       __restrict__ y,
        int64_t N, float eps, bool has_weight)
{
    const int64_t row    = blockIdx.x;
    const int64_t stride = blockDim.x;
    const __nv_bfloat16* xrow = x + row * N;
    __nv_bfloat16*        yrow = y + row * N;

    float sq_sum = 0.f;

    // Process pairs via bfloat162
    const int64_t N2 = N >> 1;
    const __nv_bfloat162* xrow2 = reinterpret_cast<const __nv_bfloat162*>(xrow);
    for (int64_t i = threadIdx.x; i < N2; i += stride) {
        float2 v = __bfloat1622float2(xrow2[i]);
        sq_sum += v.x*v.x + v.y*v.y;
    }
    // Tail element
    if ((N & 1) && threadIdx.x == 0) {
        float v = __bfloat162float(xrow[N - 1]);
        sq_sum += v * v;
    }
    sq_sum = block_reduce_sum(sq_sum);
    const float rrms = rsqrtf(sq_sum / (float)N + eps);

    // Apply
    __nv_bfloat162* yrow2 = reinterpret_cast<__nv_bfloat162*>(yrow);
    if (has_weight) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
        for (int64_t i = threadIdx.x; i < N2; i += stride) {
            float2 xv = __bfloat1622float2(xrow2[i]);
            float2 wv = __bfloat1622float2(w2[i]);
            yrow2[i] = __float22bfloat162_rn({xv.x * rrms * wv.x, xv.y * rrms * wv.y});
        }
        if ((N & 1) && threadIdx.x == 0)
            yrow[N-1] = __float2bfloat16(__bfloat162float(xrow[N-1]) * rrms
                                         * __bfloat162float(w[N-1]));
    } else {
        for (int64_t i = threadIdx.x; i < N2; i += stride) {
            float2 xv = __bfloat1622float2(xrow2[i]);
            yrow2[i] = __float22bfloat162_rn({xv.x * rrms, xv.y * rrms});
        }
        if ((N & 1) && threadIdx.x == 0)
            yrow[N-1] = __float2bfloat16(__bfloat162float(xrow[N-1]) * rrms);
    }
}

// ---------------------------------------------------------------------------
// FP16 kernel (same structure as BF16)
// ---------------------------------------------------------------------------

__global__ void rms_norm_f16_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ w,
        __half*       __restrict__ y,
        int64_t N, float eps, bool has_weight)
{
    const int64_t row    = blockIdx.x;
    const int64_t stride = blockDim.x;
    const __half* xrow = x + row * N;
    __half*       yrow = y + row * N;

    float sq_sum = 0.f;
    const int64_t N2 = N >> 1;
    const half2*  xrow2 = reinterpret_cast<const half2*>(xrow);
    for (int64_t i = threadIdx.x; i < N2; i += stride) {
        float2 v = __half22float2(xrow2[i]);
        sq_sum += v.x*v.x + v.y*v.y;
    }
    if ((N & 1) && threadIdx.x == 0) {
        float v = __half2float(xrow[N-1]);
        sq_sum += v * v;
    }
    sq_sum = block_reduce_sum(sq_sum);
    const float rrms = rsqrtf(sq_sum / (float)N + eps);

    half2* yrow2 = reinterpret_cast<half2*>(yrow);
    if (has_weight) {
        const half2* w2 = reinterpret_cast<const half2*>(w);
        for (int64_t i = threadIdx.x; i < N2; i += stride) {
            float2 xv = __half22float2(xrow2[i]);
            float2 wv = __half22float2(w2[i]);
            yrow2[i] = __float22half2_rn({xv.x * rrms * wv.x, xv.y * rrms * wv.y});
        }
        if ((N & 1) && threadIdx.x == 0)
            yrow[N-1] = __float2half(__half2float(xrow[N-1]) * rrms * __half2float(w[N-1]));
    } else {
        for (int64_t i = threadIdx.x; i < N2; i += stride) {
            float2 xv = __half22float2(xrow2[i]);
            yrow2[i] = __float22half2_rn({xv.x * rrms, xv.y * rrms});
        }
        if ((N & 1) && threadIdx.x == 0)
            yrow[N-1] = __float2half(__half2float(xrow[N-1]) * rrms);
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

void launch_rms_norm(
        const void* x, const void* w, void* y,
        int64_t num_rows, int64_t N,
        float eps, bool has_weight,
        at::ScalarType dtype,
        cudaStream_t stream)
{
    // Block size: 256 threads (8 warps) gives good occupancy across all arches.
    // For very large N (>16K) we bump to 512.
    const int block = (N > 16384) ? 512 : 256;

    switch (dtype) {
        case at::ScalarType::Float:
            rms_norm_f32_kernel<<<(int)num_rows, block, 0, stream>>>(
                (const float*)x, (const float*)w, (float*)y, N, eps, has_weight);
            break;
        case at::ScalarType::BFloat16:
            rms_norm_bf16_kernel<<<(int)num_rows, block, 0, stream>>>(
                (const __nv_bfloat16*)x, (const __nv_bfloat16*)w,
                (__nv_bfloat16*)y, N, eps, has_weight);
            break;
        case at::ScalarType::Half:
            rms_norm_f16_kernel<<<(int)num_rows, block, 0, stream>>>(
                (const __half*)x, (const __half*)w, (__half*)y, N, eps, has_weight);
            break;
        default:
            TORCH_CHECK(false, "rms_norm_cuda: unsupported dtype");
    }
}
