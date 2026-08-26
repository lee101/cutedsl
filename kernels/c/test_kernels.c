/**
 * CuteDSL C Kernel Test Suite
 *
 * Build and run:
 *   gcc -O3 -march=native -fopenmp -lm test_kernels.c -o test_kernels
 *   ./test_kernels
 *
 * Or with clang:
 *   clang -O3 -march=native -fopenmp -lm test_kernels.c -o test_kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "rms_norm.h"
#include "silu_gate.h"
#include "chronos_preprocess.h"

#define ASSERT_CLOSE(a, b, tol, msg) \
    if (fabsf((a) - (b)) > (tol)) { \
        printf("FAIL: %s: expected %.6f, got %.6f (diff=%.6e)\n", msg, (b), (a), fabsf((a)-(b))); \
        failures++; \
    }

static int failures = 0;
static int tests = 0;

static void test_rms_norm_basic(void) {
    tests++;
    printf("test_rms_norm_basic... ");

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    size_t dim = 4;

    rms_norm_f32(out, x, weight, 1, dim, 1e-5f);

    /* RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
    /* rrms ≈ 0.3651 */
    float rms = sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-5f);
    float rrms = 1.0f / rms;

    ASSERT_CLOSE(out[0], 1.0f * rrms, 1e-5f, "rms_norm[0]");
    ASSERT_CLOSE(out[1], 2.0f * rrms, 1e-5f, "rms_norm[1]");
    ASSERT_CLOSE(out[2], 3.0f * rrms, 1e-5f, "rms_norm[2]");
    ASSERT_CLOSE(out[3], 4.0f * rrms, 1e-5f, "rms_norm[3]");

    printf("ok\n");
}

static void test_rms_norm_no_weight(void) {
    tests++;
    printf("test_rms_norm_no_weight... ");

    float x[] = {1.0f, 0.0f, -1.0f, 0.0f};
    float out[4];

    rms_norm_f32(out, x, NULL, 1, 4, 1e-5f);

    float rms = sqrtf((1.0f + 0.0f + 1.0f + 0.0f) / 4.0f + 1e-5f);
    float rrms = 1.0f / rms;

    ASSERT_CLOSE(out[0], 1.0f * rrms, 1e-5f, "no_weight[0]");
    ASSERT_CLOSE(out[1], 0.0f, 1e-5f, "no_weight[1]");
    ASSERT_CLOSE(out[2], -1.0f * rrms, 1e-5f, "no_weight[2]");

    printf("ok\n");
}

static void test_rms_norm_multi_row(void) {
    tests++;
    printf("test_rms_norm_multi_row... ");

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float weight[] = {2.0f, 0.5f, 1.0f};
    float out[6];

    rms_norm_f32(out, x, weight, 2, 3, 1e-5f);

    /* Row 0: rms = sqrt((1+4+9)/3) */
    float rms0 = sqrtf((1.0f + 4.0f + 9.0f) / 3.0f + 1e-5f);
    ASSERT_CLOSE(out[0], 1.0f / rms0 * 2.0f, 1e-5f, "row0[0]");
    ASSERT_CLOSE(out[1], 2.0f / rms0 * 0.5f, 1e-5f, "row0[1]");

    /* Row 1: rms = sqrt((16+25+36)/3) */
    float rms1 = sqrtf((16.0f + 25.0f + 36.0f) / 3.0f + 1e-5f);
    ASSERT_CLOSE(out[3], 4.0f / rms1 * 2.0f, 1e-5f, "row1[0]");

    printf("ok\n");
}

static void test_silu_gate_basic(void) {
    tests++;
    printf("test_silu_gate_basic... ");

    float x1[] = {0.0f, 1.0f, -1.0f, 2.0f};
    float x3[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    fused_silu_gate_f32(out, x1, x3, 4);

    /* silu(0) * 1 = 0 */
    ASSERT_CLOSE(out[0], 0.0f, 1e-5f, "silu_gate[0]");

    /* silu(1) * 1 = 1 * sigmoid(1) ≈ 0.7311 */
    float expected = 1.0f / (1.0f + expf(-1.0f));
    ASSERT_CLOSE(out[1], expected, 1e-5f, "silu_gate[1]");

    /* silu(-1) * 1 = -1 * sigmoid(-1) ≈ -0.2689 */
    float expected_neg = -1.0f / (1.0f + expf(1.0f));
    ASSERT_CLOSE(out[2], expected_neg, 1e-5f, "silu_gate[2]");

    printf("ok\n");
}

static void test_silu_gate_gating(void) {
    tests++;
    printf("test_silu_gate_gating... ");

    float x1[] = {1.0f, 1.0f, 1.0f};
    float x3[] = {0.0f, 2.0f, -1.0f};
    float out[3];

    fused_silu_gate_f32(out, x1, x3, 3);

    float silu_1 = 1.0f / (1.0f + expf(-1.0f));
    ASSERT_CLOSE(out[0], silu_1 * 0.0f, 1e-5f, "gating[0]");
    ASSERT_CLOSE(out[1], silu_1 * 2.0f, 1e-5f, "gating[1]");
    ASSERT_CLOSE(out[2], silu_1 * -1.0f, 1e-5f, "gating[2]");

    printf("ok\n");
}

static void test_chronos_preprocess_padding_and_nan(void) {
    tests++;
    printf("test_chronos_preprocess_padding_and_nan... ");
    float input[] = {1.0f, NAN, 3.0f};
    float patched[12];
    float attention[2];
    float location[1];
    float scale[1];
    cute_chronos_status status = cute_chronos_preprocess_f32(
        patched, attention, location, scale, input, 1, 3, 2, 8, 0);
    if (status != CUTE_CHRONOS_OK) {
        printf("FAIL: status=%d\n", status);
        failures++;
        return;
    }
    ASSERT_CLOSE(location[0], 2.0f, 1e-6f, "chronos location");
    ASSERT_CLOSE(scale[0], 1.0f, 1e-6f, "chronos scale");
    ASSERT_CLOSE(patched[0], -0.5f, 1e-6f, "left-pad time");
    ASSERT_CLOSE(patched[2], 0.0f, 1e-6f, "left-pad value");
    ASSERT_CLOSE(patched[4], 0.0f, 1e-6f, "left-pad mask");
    ASSERT_CLOSE(patched[3], -1.0f, 1e-6f, "first normalized value");
    ASSERT_CLOSE(patched[5], 1.0f, 1e-6f, "first value mask");
    ASSERT_CLOSE(patched[8], 0.0f, 1e-6f, "nan normalized value");
    ASSERT_CLOSE(patched[10], 0.0f, 1e-6f, "nan mask");
    ASSERT_CLOSE(patched[9], 1.0f, 1e-6f, "last normalized value");
    ASSERT_CLOSE(attention[0], 1.0f, 1e-6f, "first patch attention");
    ASSERT_CLOSE(attention[1], 1.0f, 1e-6f, "second patch attention");
    printf("ok\n");
}

static void test_chronos_preprocess_all_nan_and_constant(void) {
    tests++;
    printf("test_chronos_preprocess_all_nan_and_constant... ");
    float input[] = {NAN, NAN, 4.0f, 4.0f};
    float patched[12];
    float attention[2];
    float location[2];
    float scale[2];
    cute_chronos_status status = cute_chronos_preprocess_f32(
        patched, attention, location, scale, input, 2, 2, 2, 2, 1);
    if (status != CUTE_CHRONOS_OK) {
        printf("FAIL: status=%d\n", status);
        failures++;
        return;
    }
    ASSERT_CLOSE(location[0], 0.0f, 1e-6f, "all-nan location");
    ASSERT_CLOSE(scale[0], 1.0f, 1e-6f, "all-nan scale");
    ASSERT_CLOSE(attention[0], 0.0f, 1e-6f, "all-nan attention");
    ASSERT_CLOSE(location[1], 4.0f, 1e-6f, "constant location");
    ASSERT_CLOSE(scale[1], 1e-5f, 1e-8f, "constant scale");
    ASSERT_CLOSE(attention[1], 1.0f, 1e-6f, "constant attention");
    printf("ok\n");
}

static void test_chronos_preprocess_truncates_tail(void) {
    tests++;
    printf("test_chronos_preprocess_truncates_tail... ");
    float input[] = {100.0f, 200.0f, 1.0f, 3.0f};
    float patched[6];
    float attention[1];
    float location[1];
    float scale[1];
    cute_chronos_status status = cute_chronos_preprocess_f32(
        patched, attention, location, scale, input, 1, 4, 2, 2, 1);
    if (status != CUTE_CHRONOS_OK) {
        printf("FAIL: status=%d\n", status);
        failures++;
        return;
    }
    ASSERT_CLOSE(location[0], 2.0f, 1e-6f, "truncated location");
    ASSERT_CLOSE(scale[0], 1.0f, 1e-6f, "truncated scale");
    ASSERT_CLOSE(patched[2], asinhf(-1.0f), 1e-6f, "truncated first");
    ASSERT_CLOSE(patched[3], asinhf(1.0f), 1e-6f, "truncated last");
    printf("ok\n");
}

static void test_chronos_preprocess_rejects_bad_sizes(void) {
    tests++;
    printf("test_chronos_preprocess_rejects_bad_sizes... ");
    float value = 0.0f;
    cute_chronos_status status = cute_chronos_preprocess_f32(
        &value, &value, &value, &value, &value, 1, 1, 0, 1, 0);
    if (status != CUTE_CHRONOS_INVALID_ARGUMENT) {
        printf("FAIL: zero patch status=%d\n", status);
        failures++;
        return;
    }
    status = cute_chronos_preprocess_f32(
        &value, &value, &value, &value, &value, SIZE_MAX, 2, 1, 1, 0);
    if (status != CUTE_CHRONOS_SIZE_OVERFLOW) {
        printf("FAIL: overflow status=%d\n", status);
        failures++;
        return;
    }
    printf("ok\n");
}

int main(void) {
    printf("CuteDSL C Kernel Tests\n");
    printf("======================\n\n");

    test_rms_norm_basic();
    test_rms_norm_no_weight();
    test_rms_norm_multi_row();
    test_silu_gate_basic();
    test_silu_gate_gating();
    test_chronos_preprocess_padding_and_nan();
    test_chronos_preprocess_all_nan_and_constant();
    test_chronos_preprocess_truncates_tail();
    test_chronos_preprocess_rejects_bad_sizes();

    printf("\n%d tests, %d failures\n", tests, failures);
    return failures > 0 ? 1 : 0;
}
