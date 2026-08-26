#ifndef CUTEDSL_CHRONOS_PREPROCESS_H
#define CUTEDSL_CHRONOS_PREPROCESS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUTE_CHRONOS_OK = 0,
    CUTE_CHRONOS_INVALID_ARGUMENT = 1,
    CUTE_CHRONOS_SIZE_OVERFLOW = 2
} cute_chronos_status;

size_t cute_chronos_patch_count(
    size_t input_length, size_t patch_size, size_t context_length);

/*
 * Output layout is [batch, patches, 3 * patch_size]: time encoding,
 * normalized values, validity mask. The caller owns every buffer.
 */
cute_chronos_status cute_chronos_preprocess_f32(
    float *patched,
    float *attention_mask,
    float *location,
    float *scale,
    const float *input,
    size_t batch,
    size_t input_length,
    size_t patch_size,
    size_t context_length,
    int use_arcsinh);

#ifdef __cplusplus
}
#endif

#endif
