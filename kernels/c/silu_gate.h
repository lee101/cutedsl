/**
 * CuteDSL: Pure C SiLU-gated FFN
 *
 * Computes: out = silu(x1) * x3
 * where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Used in Z-Image's gated FFN: w2(silu(w1(x)) * w3(x))
 */

#ifndef CUTEDSL_SILU_GATE_H
#define CUTEDSL_SILU_GATE_H

#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fused SiLU + elementwise gating (float32).
 *
 * out[i] = silu(x1[i]) * x3[i]
 *
 * @param out  Output buffer, shape (n,)
 * @param x1   First input (gate projection output), shape (n,)
 * @param x3   Second input (up projection output), shape (n,)
 * @param n    Number of elements
 */
static inline void fused_silu_gate_f32(
    float* out,
    const float* x1,
    const float* x3,
    size_t n
) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float v = x1[i];
        float silu_v = v / (1.0f + expf(-v));
        out[i] = silu_v * x3[i];
    }
}

/**
 * Full gated FFN: out = w2 @ (silu(w1 @ x) * (w3 @ x))
 *
 * @param out       Output buffer, shape (M, dim)
 * @param x         Input, shape (M, dim)
 * @param w1        Gate weight, shape (hidden_dim, dim), row-major
 * @param w2        Down weight, shape (dim, hidden_dim), row-major
 * @param w3        Up weight, shape (hidden_dim, dim), row-major
 * @param M         Number of tokens (rows)
 * @param dim       Model dimension
 * @param hidden_dim FFN hidden dimension
 * @param tmp1      Scratch buffer, shape (M, hidden_dim)
 * @param tmp3      Scratch buffer, shape (M, hidden_dim)
 */
static inline void gated_ffn_f32(
    float* out,
    const float* x,
    const float* w1,
    const float* w2,
    const float* w3,
    size_t M,
    size_t dim,
    size_t hidden_dim,
    float* tmp1,
    float* tmp3
) {
    /* tmp1 = x @ w1^T */
    #pragma omp parallel for
    for (size_t m = 0; m < M; m++) {
        for (size_t h = 0; h < hidden_dim; h++) {
            float acc = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                acc += x[m * dim + d] * w1[h * dim + d];
            }
            tmp1[m * hidden_dim + h] = acc;
        }
    }

    /* tmp3 = x @ w3^T */
    #pragma omp parallel for
    for (size_t m = 0; m < M; m++) {
        for (size_t h = 0; h < hidden_dim; h++) {
            float acc = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                acc += x[m * dim + d] * w3[h * dim + d];
            }
            tmp3[m * hidden_dim + h] = acc;
        }
    }

    /* tmp1 = silu(tmp1) * tmp3 (fused) */
    fused_silu_gate_f32(tmp1, tmp1, tmp3, M * hidden_dim);

    /* out = tmp1 @ w2^T */
    #pragma omp parallel for
    for (size_t m = 0; m < M; m++) {
        for (size_t d = 0; d < dim; d++) {
            float acc = 0.0f;
            for (size_t h = 0; h < hidden_dim; h++) {
                acc += tmp1[m * hidden_dim + h] * w2[d * hidden_dim + h];
            }
            out[m * dim + d] = acc;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CUTEDSL_SILU_GATE_H */
