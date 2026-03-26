#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "qk_scores.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TurboQuant CUDA kernels";

    m.def(
        "qk_scores_mse",
        &qk_scores_mse_cuda,
        "Packed low-bit decode + QK^T for TurboQuant MSE keys (CUDA)",
        py::arg("rotated_query"),
        py::arg("packed_indices"),
        py::arg("norms"),
        py::arg("codebook"),
        py::arg("dim"),
        py::arg("bits"));
}
