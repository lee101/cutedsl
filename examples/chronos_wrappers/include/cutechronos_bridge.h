#ifndef CUTECHRONOS_BRIDGE_H
#define CUTECHRONOS_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cutechronos_init_pipeline(
    const char *model_id,
    const char *backend,
    const char *device,
    const char *dtype_name,
    const char *compile_mode,
    int *out_handle,
    char *error_buffer,
    size_t error_buffer_size
);

int cutechronos_predict_median(
    int handle,
    const float *context,
    int context_length,
    int prediction_length,
    float *out_values,
    int out_capacity,
    int *out_length,
    double *out_latency_ms,
    char *error_buffer,
    size_t error_buffer_size
);

int cutechronos_destroy_pipeline(
    int handle,
    char *error_buffer,
    size_t error_buffer_size
);

#ifdef __cplusplus
}
#endif

#endif
