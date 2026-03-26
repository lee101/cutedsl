#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "qk_scores.cuh"

namespace {

template <typename scalar_t>
__device__ inline float to_float(scalar_t value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float to_float<c10::Half>(c10::Half value) {
    return __half2float(value);
}

template <>
__device__ inline float to_float<c10::BFloat16>(c10::BFloat16 value) {
    return static_cast<float>(value);
}

__device__ inline int unpack_code(const uint8_t* packed, int coord, int bits) {
    const int bit_pos = coord * bits;
    const int byte_idx = bit_pos >> 3;
    const int bit_offset = bit_pos & 7;
    const int mask = (1 << bits) - 1;

    int value = static_cast<int>(packed[byte_idx]) >> bit_offset;
    const int spill = bit_offset + bits - 8;
    if (spill > 0) {
        value |= static_cast<int>(packed[byte_idx + 1]) << (8 - bit_offset);
    }
    return value & mask;
}

template <typename scalar_t>
__global__ void qk_scores_mse_kernel(
    const scalar_t* __restrict__ rotated_query,
    const uint8_t* __restrict__ packed_indices,
    const scalar_t* __restrict__ norms,
    const float* __restrict__ codebook,
    float* __restrict__ out,
    int q_rows,
    int k_rows,
    int packed_len,
    int dim,
    int bits)
{
    const int q_idx = blockIdx.y;
    const int k_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (q_idx >= q_rows || k_idx >= k_rows) {
        return;
    }

    const scalar_t* q_ptr = rotated_query + static_cast<int64_t>(q_idx) * dim;
    const uint8_t* packed_ptr = packed_indices + static_cast<int64_t>(k_idx) * packed_len;

    float acc = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        const int code_idx = unpack_code(packed_ptr, d, bits);
        acc += to_float(q_ptr[d]) * codebook[code_idx];
    }

    __shared__ float partials[256];
    partials[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[static_cast<int64_t>(q_idx) * k_rows + k_idx] = partials[0] * to_float(norms[k_idx]);
    }
}

}  // namespace

torch::Tensor qk_scores_mse_cuda(
    torch::Tensor rotated_query,
    torch::Tensor packed_indices,
    torch::Tensor norms,
    torch::Tensor codebook,
    int64_t dim,
    int64_t bits)
{
    TORCH_CHECK(rotated_query.is_cuda(), "rotated_query must be CUDA");
    TORCH_CHECK(packed_indices.is_cuda(), "packed_indices must be CUDA");
    TORCH_CHECK(norms.is_cuda(), "norms must be CUDA");
    TORCH_CHECK(codebook.is_cuda(), "codebook must be CUDA");
    TORCH_CHECK(rotated_query.is_contiguous(), "rotated_query must be contiguous");
    TORCH_CHECK(packed_indices.is_contiguous(), "packed_indices must be contiguous");
    TORCH_CHECK(norms.is_contiguous(), "norms must be contiguous");
    TORCH_CHECK(codebook.is_contiguous(), "codebook must be contiguous");
    TORCH_CHECK(packed_indices.scalar_type() == torch::kUInt8, "packed_indices must be uint8");
    TORCH_CHECK(rotated_query.dim() == 2, "rotated_query must be 2D");
    TORCH_CHECK(packed_indices.dim() == 2, "packed_indices must be 2D");
    TORCH_CHECK(norms.dim() == 1, "norms must be 1D");
    TORCH_CHECK(codebook.dim() == 1, "codebook must be 1D");
    TORCH_CHECK(rotated_query.size(1) == dim, "rotated_query dim mismatch");
    TORCH_CHECK(norms.size(0) == packed_indices.size(0), "norms/key rows mismatch");
    TORCH_CHECK(bits >= 1 && bits <= 8, "bits must be in [1, 8]");

    const auto q_rows = static_cast<int>(rotated_query.size(0));
    const auto k_rows = static_cast<int>(packed_indices.size(0));
    const auto packed_len = static_cast<int>(packed_indices.size(1));

    auto out = torch::empty({q_rows, k_rows}, rotated_query.options().dtype(torch::kFloat32));
    dim3 grid(k_rows, q_rows);
    constexpr int threads = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        rotated_query.scalar_type(),
        "qk_scores_mse_cuda",
        [&] {
            qk_scores_mse_kernel<scalar_t><<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                rotated_query.data_ptr<scalar_t>(),
                packed_indices.data_ptr<uint8_t>(),
                norms.data_ptr<scalar_t>(),
                codebook.data_ptr<float>(),
                out.data_ptr<float>(),
                q_rows,
                k_rows,
                packed_len,
                static_cast<int>(dim),
                static_cast<int>(bits));
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
