/**
 * CuteDSL: Pure C RMS LayerNorm
 *
 * T5-style RMS normalization: out = weight * (x * rsqrt(mean(x^2) + eps))
 * No mean subtraction, no bias.
 *
 * This is the reference C implementation for building lightweight inference
 * binaries without PyTorch/Triton dependencies. Compile with:
 *
 *   gcc -O3 -march=native -fopenmp -c rms_norm.c -o rms_norm.o
 *   nvcc -O3 rms_norm.cu -o rms_norm_cuda.o  # for CUDA version
 *
 * The same logic is available as:
 * - cutechronos/triton_kernels/rms_layernorm.py (Triton, GPU)
 * - cutezimage/triton_kernels/rms_norm.py (Triton, GPU)
 * - kernels/c/rms_norm.h (this file, pure C, CPU/CUDA)
 */

#ifndef CUTEDSL_RMS_NORM_H
#define CUTEDSL_RMS_NORM_H

#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMS LayerNorm forward pass (float32).
 *
 * @param out     Output buffer, shape (num_rows, dim)
 * @param x       Input buffer, shape (num_rows, dim)
 * @param weight  Scale parameter, shape (dim,). NULL for no scaling.
 * @param num_rows Number of rows to process
 * @param dim     Hidden dimension (last axis)
 * @param eps     Epsilon for numerical stability (typically 1e-5 or 1e-6)
 */
static inline void rms_norm_f32(
    float* out,
    const float* x,
    const float* weight,
    size_t num_rows,
    size_t dim,
    float eps
) {
    #pragma omp parallel for
    for (size_t row = 0; row < num_rows; row++) {
        const float* x_row = x + row * dim;
        float* out_row = out + row * dim;

        /* Compute sum of squares */
        float sq_sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            sq_sum += x_row[i] * x_row[i];
        }

        /* RMS inverse */
        float rrms = 1.0f / sqrtf(sq_sum / (float)dim + eps);

        /* Normalize and optionally scale */
        if (weight != NULL) {
            for (size_t i = 0; i < dim; i++) {
                out_row[i] = x_row[i] * rrms * weight[i];
            }
        } else {
            for (size_t i = 0; i < dim; i++) {
                out_row[i] = x_row[i] * rrms;
            }
        }
    }
}

/**
 * RMS LayerNorm forward pass (bfloat16 input/output, float32 accumulation).
 *
 * bfloat16 is stored as uint16_t. Variance computed in float32.
 */
static inline void rms_norm_bf16(
    unsigned short* out,
    const unsigned short* x,
    const unsigned short* weight,
    size_t num_rows,
    size_t dim,
    float eps
) {
    #pragma omp parallel for
    for (size_t row = 0; row < num_rows; row++) {
        const unsigned short* x_row = x + row * dim;
        unsigned short* out_row = out + row * dim;

        /* Convert bf16 to f32, compute sum of squares */
        float sq_sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            /* bf16 -> f32: shift left by 16 bits */
            union { unsigned int u; float f; } conv;
            conv.u = ((unsigned int)x_row[i]) << 16;
            float val = conv.f;
            sq_sum += val * val;
        }

        float rrms = 1.0f / sqrtf(sq_sum / (float)dim + eps);

        for (size_t i = 0; i < dim; i++) {
            /* bf16 -> f32 */
            union { unsigned int u; float f; } conv_x, conv_w;
            conv_x.u = ((unsigned int)x_row[i]) << 16;
            float val = conv_x.f * rrms;

            if (weight != NULL) {
                conv_w.u = ((unsigned int)weight[i]) << 16;
                val *= conv_w.f;
            }

            /* f32 -> bf16: round and truncate */
            union { float f; unsigned int u; } conv_out;
            conv_out.f = val;
            /* Round to nearest even */
            conv_out.u += 0x7FFF + ((conv_out.u >> 16) & 1);
            out_row[i] = (unsigned short)(conv_out.u >> 16);
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CUTEDSL_RMS_NORM_H */
