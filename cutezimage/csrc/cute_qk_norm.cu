/**
 * cute_qk_norm.cu
 * Fused per-head RMS norm for Q and K in a single kernel.
 *
 * Input shapes: Q (B, S, H, D), K (B, S, H, D)
 * Each CUDA block handles one (b, s, h) tile.
 * Warp reductions avoid shared memory for small D.
 */

#include "cute_qk_norm.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Type conversion helpers – (float) cast is not portable for bf16/half
// ---------------------------------------------------------------------------
template <typename T> __device__ __forceinline__ float  to_f32(T v)    { return (float)v; }
template <>            __device__ __forceinline__ float  to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>            __device__ __forceinline__ float  to_f32<__half>(__half v)               { return __half2float(v); }
template <typename T> __device__ __forceinline__ T      from_f32(float v);
template <>            __device__ __forceinline__ float  from_f32<float>(float v)               { return v; }
template <>            __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>            __device__ __forceinline__ __half from_f32<__half>(float v)              { return __float2half(v); }

// Warp reduce (same as rms_norm kernel)
__device__ __forceinline__ float warp_reduce_sum_qk(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float block_reduce_sum_qk(float v) {
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum_qk(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    const int n_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < n_warps) ? smem[lane] : 0.f;
    v = warp_reduce_sum_qk(v);
    return v;
}

// ---------------------------------------------------------------------------
// Generic template kernel: one block per (b, s, h) position for both Q and K
// ---------------------------------------------------------------------------

template <typename T>
__global__ void fused_qk_norm_kernel(
        const T* __restrict__ q,
        const T* __restrict__ k,
        const T* __restrict__ qw,
        const T* __restrict__ kw,
        T*       __restrict__ q_out,
        T*       __restrict__ k_out,
        int64_t S, int64_t H, int64_t D,
        float eps)
{
    // pid = b * S*H + s * H + h
    const int64_t pid = blockIdx.x;
    const int64_t h   = pid % H;
    const int64_t s   = (pid / H) % S;
    const int64_t b   = pid / (H * S);

    const T* qrow = q     + (b * S * H + s * H + h) * D;
    const T* krow = k     + (b * S * H + s * H + h) * D;
    T*       qout = q_out + (b * S * H + s * H + h) * D;
    T*       kout = k_out + (b * S * H + s * H + h) * D;

    // Accumulate sum-of-squares for Q and K simultaneously
    float q_sq = 0.f, k_sq = 0.f;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        float qv = to_f32(qrow[i]);
        float kv = to_f32(krow[i]);
        q_sq += qv * qv;
        k_sq += kv * kv;
    }
    q_sq = block_reduce_sum_qk(q_sq);
    // Second reduction for k (need separate smem call – reuse after sync)
    __shared__ float k_sq_shared;
    if (threadIdx.x == 0) k_sq_shared = 0.f;
    __syncthreads();

    // Re-accumulate k with fresh smem (q_sq done, smem reusable after __syncthreads above)
    float kv_sq = 0.f;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        float kv = to_f32(krow[i]);
        kv_sq += kv * kv;
    }
    kv_sq = block_reduce_sum_qk(kv_sq);

    const float q_rrms = rsqrtf(q_sq  / (float)D + eps);
    const float k_rrms = rsqrtf(kv_sq / (float)D + eps);

    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        qout[i] = from_f32<T>(to_f32(qrow[i]) * q_rrms * to_f32(qw[i]));
        kout[i] = from_f32<T>(to_f32(krow[i]) * k_rrms * to_f32(kw[i]));
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void launch_fused_qk_norm(
        const void* q, const void* k,
        const void* qw, const void* kw,
        void* q_out, void* k_out,
        int64_t B, int64_t S, int64_t H, int64_t D,
        float eps,
        at::ScalarType dtype,
        cudaStream_t stream)
{
    const int64_t num_blocks = B * S * H;
    // block size: 128 threads is enough for D=128 (head_dim), saturates a warp
    const int threads = (D <= 64) ? 32 : (D <= 128 ? 64 : 128);

    switch (dtype) {
        case at::ScalarType::Float:
            fused_qk_norm_kernel<float><<<(int)num_blocks, threads, 0, stream>>>(
                (const float*)q, (const float*)k,
                (const float*)qw, (const float*)kw,
                (float*)q_out, (float*)k_out,
                S, H, D, eps);
            break;
        case at::ScalarType::BFloat16:
            fused_qk_norm_kernel<__nv_bfloat16><<<(int)num_blocks, threads, 0, stream>>>(
                (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
                (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
                (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out,
                S, H, D, eps);
            break;
        case at::ScalarType::Half:
            fused_qk_norm_kernel<__half><<<(int)num_blocks, threads, 0, stream>>>(
                (const __half*)q, (const __half*)k,
                (const __half*)qw, (const __half*)kw,
                (__half*)q_out, (__half*)k_out,
                S, H, D, eps);
            break;
        default:
            TORCH_CHECK(false, "fused_qk_norm_cuda: unsupported dtype");
    }
}
