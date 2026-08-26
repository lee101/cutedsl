#include "chronos_preprocess.h"

#include <math.h>
#include <stdint.h>

static int size_mul(size_t left, size_t right, size_t *out) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *out = left * right;
    return 1;
}

size_t cute_chronos_patch_count(
    size_t input_length, size_t patch_size, size_t context_length
) {
    if (patch_size == 0 || context_length == 0 || input_length == 0) return 0;
    size_t used = input_length < context_length ? input_length : context_length;
    return used / patch_size + (used % patch_size != 0);
}

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
    int use_arcsinh
) {
    if (!patched || !attention_mask || !location || !scale || !input ||
        batch == 0 || input_length == 0 || patch_size == 0 || context_length == 0) {
        return CUTE_CHRONOS_INVALID_ARGUMENT;
    }

    size_t patches = cute_chronos_patch_count(input_length, patch_size, context_length);
    size_t used_length = input_length < context_length ? input_length : context_length;
    size_t padded_length;
    size_t features;
    size_t output_items;
    size_t input_items;
    if (!size_mul(batch, input_length, &input_items) ||
        !size_mul(patches, patch_size, &padded_length) ||
        !size_mul(3, patch_size, &features) ||
        !size_mul(batch, patches, &output_items) ||
        !size_mul(output_items, features, &output_items)) {
        return CUTE_CHRONOS_SIZE_OVERFLOW;
    }
    (void)input_items;
    (void)output_items;

    size_t source_start = input_length - used_length;
    size_t left_padding = padded_length - used_length;
    for (size_t b = 0; b < batch; ++b) {
        const float *row = input + b * input_length + source_start;
        double sum = 0.0;
        size_t count = 0;
        for (size_t i = 0; i < used_length; ++i) {
            if (!isnan(row[i])) {
                sum += row[i];
                ++count;
            }
        }

        float mean = count ? (float)(sum / (double)count) : 0.0f;
        double squared_sum = 0.0;
        for (size_t i = 0; i < used_length; ++i) {
            if (!isnan(row[i])) {
                double difference = (double)row[i] - mean;
                squared_sum += difference * difference;
            }
        }
        float standard_deviation = count
            ? sqrtf((float)(squared_sum / (double)count)) : 1.0f;
        if (standard_deviation == 0.0f) standard_deviation = 1e-5f;
        location[b] = mean;
        scale[b] = standard_deviation;

        for (size_t p = 0; p < patches; ++p) {
            float *out = patched + (b * patches + p) * features;
            int any_valid = 0;
            for (size_t t = 0; t < patch_size; ++t) {
                size_t padded_position = p * patch_size + t;
                float normalized = 0.0f;
                float valid = 0.0f;
                if (padded_position >= left_padding) {
                    size_t source_position = padded_position - left_padding;
                    float value = row[source_position];
                    if (!isnan(value)) {
                        normalized = (value - mean) / standard_deviation;
                        if (use_arcsinh) normalized = asinhf(normalized);
                        valid = 1.0f;
                        any_valid = 1;
                    }
                }
                out[t] = ((float)padded_position - (float)padded_length) /
                         (float)context_length;
                out[patch_size + t] = normalized;
                out[2 * patch_size + t] = valid;
            }
            attention_mask[b * patches + p] = any_valid ? 1.0f : 0.0f;
        }
    }
    return CUTE_CHRONOS_OK;
}
