/**
 * cute_kernels_ext.cpp
 * PyTorch pybind11 extension exposing CUDA kernels as cutezimage._C.
 *
 * Exposed functions:
 *   rms_norm(x, weight?, eps)         -> Tensor
 *   fused_silu_gate(x1, x3)           -> Tensor
 *   fused_qk_norm(q, k, qw, kw, eps)  -> (Tensor, Tensor)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "cute_rms_norm.cuh"
#include "cute_silu_gate.cuh"
#include "cute_qk_norm.cuh"

// ---------------------------------------------------------------------------
// rms_norm
// ---------------------------------------------------------------------------

torch::Tensor rms_norm_cuda(
        torch::Tensor x,
        c10::optional<torch::Tensor> weight,
        double eps)
{
    TORCH_CHECK(x.is_cuda(), "rms_norm: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "rms_norm: input must be contiguous");

    const int64_t N       = x.size(-1);
    const int64_t num_rows = x.numel() / N;
    const bool has_weight = weight.has_value();

    if (has_weight) {
        TORCH_CHECK(weight->is_contiguous(), "rms_norm: weight must be contiguous");
        TORCH_CHECK(weight->size(0) == N,
                    "rms_norm: weight size ", weight->size(0), " != N ", N);
    }

    auto y = torch::empty_like(x);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_rms_norm(
        x.data_ptr(),
        has_weight ? weight->data_ptr() : nullptr,
        y.data_ptr(),
        num_rows, N,
        (float)eps, has_weight,
        x.scalar_type(),
        stream);

    return y;
}

// ---------------------------------------------------------------------------
// fused_silu_gate
// ---------------------------------------------------------------------------

torch::Tensor fused_silu_gate_cuda(torch::Tensor x1, torch::Tensor x3)
{
    TORCH_CHECK(x1.is_cuda() && x3.is_cuda(),
                "fused_silu_gate: inputs must be CUDA tensors");
    TORCH_CHECK(x1.is_contiguous() && x3.is_contiguous(),
                "fused_silu_gate: inputs must be contiguous");
    TORCH_CHECK(x1.sizes() == x3.sizes(),
                "fused_silu_gate: shape mismatch");
    TORCH_CHECK(x1.scalar_type() == x3.scalar_type(),
                "fused_silu_gate: dtype mismatch");

    auto out = torch::empty_like(x1);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_silu_gate(
        x1.data_ptr(), x3.data_ptr(), out.data_ptr(),
        x1.numel(),
        x1.scalar_type(),
        stream);

    return out;
}

// ---------------------------------------------------------------------------
// fused_qk_norm
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> fused_qk_norm_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor q_weight,
        torch::Tensor k_weight,
        double eps)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda(),
                "fused_qk_norm: Q/K must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),
                "fused_qk_norm: Q/K must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes(), "fused_qk_norm: Q/K shape mismatch");
    TORCH_CHECK(q.dim() == 4, "fused_qk_norm: expected 4-D input (B, S, H, D)");

    const int64_t B = q.size(0), S = q.size(1), H = q.size(2), D = q.size(3);

    if (!q_weight.is_contiguous()) q_weight = q_weight.contiguous();
    if (!k_weight.is_contiguous()) k_weight = k_weight.contiguous();

    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_fused_qk_norm(
        q.data_ptr(), k.data_ptr(),
        q_weight.data_ptr(), k_weight.data_ptr(),
        q_out.data_ptr(), k_out.data_ptr(),
        B, S, H, D,
        (float)eps,
        q.scalar_type(),
        stream);

    return {q_out, k_out};
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CuteZImage CUDA kernels – compiled for A40/H100/RTX5090";

    m.def("rms_norm",
          &rms_norm_cuda,
          "Vectorized RMS LayerNorm (CUDA)",
          py::arg("x"),
          py::arg("weight") = py::none(),
          py::arg("eps") = 1e-5);

    m.def("fused_silu_gate",
          &fused_silu_gate_cuda,
          "Fused SiLU-gate multiply: silu(x1) * x3 (CUDA)",
          py::arg("x1"),
          py::arg("x3"));

    m.def("fused_qk_norm",
          &fused_qk_norm_cuda,
          "Fused per-head RMS norm for Q and K (CUDA)",
          py::arg("q"),
          py::arg("k"),
          py::arg("q_weight"),
          py::arg("k_weight"),
          py::arg("eps") = 1e-5);
}
