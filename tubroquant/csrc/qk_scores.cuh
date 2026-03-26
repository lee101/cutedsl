#pragma once

#include <torch/extension.h>

torch::Tensor qk_scores_mse_cuda(
    torch::Tensor rotated_query,
    torch::Tensor packed_indices,
    torch::Tensor norms,
    torch::Tensor codebook,
    int64_t dim,
    int64_t bits);
