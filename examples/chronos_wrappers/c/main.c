#include "cutechronos_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ERROR_BUFFER_SIZE 4096

typedef struct {
    const char *model_id;
    const char *backend;
    const char *device;
    const char *dtype_name;
    const char *compile_mode;
    const char *context_csv;
    const char *actual_csv;
    int prediction_length;
    int runs;
    int warmup;
} Options;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int count_values(const char *csv) {
    int count = 1;
    for (const char *p = csv; *p != '\0'; ++p) {
        if (*p == ',') {
            count++;
        }
    }
    return count;
}

static float *parse_csv_floats(const char *csv, int *out_len) {
    char *copy = strdup(csv);
    char *saveptr = NULL;
    int count = count_values(csv);
    float *values = (float *)calloc((size_t)count, sizeof(float));
    int idx = 0;

    if (copy == NULL || values == NULL) {
        free(copy);
        free(values);
        return NULL;
    }

    char *token = strtok_r(copy, ",", &saveptr);
    while (token != NULL) {
        values[idx++] = strtof(token, NULL);
        token = strtok_r(NULL, ",", &saveptr);
    }

    free(copy);
    *out_len = idx;
    return values;
}

static double compute_mae(const float *pred, int pred_len, const float *actual, int actual_len) {
    int length = pred_len < actual_len ? pred_len : actual_len;
    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < length; ++i) {
        if (isnan(pred[i]) || isnan(actual[i])) {
            continue;
        }
        sum += fabs((double)pred[i] - (double)actual[i]);
        valid++;
    }
    return valid > 0 ? sum / (double)valid : NAN;
}

static double compute_mape_pct(const float *pred, int pred_len, const float *actual, int actual_len) {
    int length = pred_len < actual_len ? pred_len : actual_len;
    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < length; ++i) {
        if (isnan(pred[i]) || isnan(actual[i]) || fabs((double)actual[i]) < 1e-12) {
            continue;
        }
        sum += fabs(((double)pred[i] - (double)actual[i]) / (double)actual[i]) * 100.0;
        valid++;
    }
    return valid > 0 ? sum / (double)valid : NAN;
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --context 1,2,3 --actual 4,5,6 [options]\n"
        "  --model-id amazon/chronos-2\n"
        "  --backend cute|original\n"
        "  --device cuda|cpu\n"
        "  --dtype bfloat16|float16|float32\n"
        "  --compile-mode reduce-overhead\n"
        "  --prediction-length 3\n"
        "  --runs 5\n"
        "  --warmup 1\n",
        argv0
    );
}

static int parse_args(int argc, char **argv, Options *opts) {
    opts->model_id = "amazon/chronos-2";
    opts->backend = "cute";
    opts->device = "cuda";
    opts->dtype_name = "bfloat16";
    opts->compile_mode = "";
    opts->context_csv = NULL;
    opts->actual_csv = NULL;
    opts->prediction_length = 3;
    opts->runs = 5;
    opts->warmup = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            opts->model_id = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            opts->backend = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            opts->device = argv[++i];
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            opts->dtype_name = argv[++i];
        } else if (strcmp(argv[i], "--compile-mode") == 0 && i + 1 < argc) {
            opts->compile_mode = argv[++i];
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            opts->context_csv = argv[++i];
        } else if (strcmp(argv[i], "--actual") == 0 && i + 1 < argc) {
            opts->actual_csv = argv[++i];
        } else if (strcmp(argv[i], "--prediction-length") == 0 && i + 1 < argc) {
            opts->prediction_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            opts->runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts->warmup = atoi(argv[++i]);
        } else {
            return -1;
        }
    }

    return (opts->context_csv != NULL && opts->actual_csv != NULL) ? 0 : -1;
}

static void print_json_array(const float *values, int len) {
    printf("[");
    for (int i = 0; i < len; ++i) {
        if (i > 0) {
            printf(", ");
        }
        printf("%.6f", values[i]);
    }
    printf("]");
}

int main(int argc, char **argv) {
    Options opts;
    if (parse_args(argc, argv, &opts) != 0) {
        print_usage(argv[0]);
        return 2;
    }

    int context_len = 0;
    int actual_len = 0;
    float *context = parse_csv_floats(opts.context_csv, &context_len);
    float *actual = parse_csv_floats(opts.actual_csv, &actual_len);
    if (context == NULL || actual == NULL) {
        fprintf(stderr, "failed to parse context or actual arrays\n");
        free(context);
        free(actual);
        return 1;
    }

    int handle = 0;
    char error_buffer[ERROR_BUFFER_SIZE];
    if (cutechronos_init_pipeline(
            opts.model_id,
            opts.backend,
            opts.device,
            opts.dtype_name,
            opts.compile_mode,
            &handle,
            error_buffer,
            sizeof(error_buffer)) != 0) {
        fprintf(stderr, "init failed: %s\n", error_buffer);
        free(context);
        free(actual);
        return 1;
    }

    float *forecast = (float *)calloc((size_t)opts.prediction_length, sizeof(float));
    if (forecast == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(context);
        free(actual);
        return 1;
    }

    for (int i = 0; i < opts.warmup; ++i) {
        int warmup_len = 0;
        double warmup_latency = 0.0;
        if (cutechronos_predict_median(
                handle,
                context,
                context_len,
                opts.prediction_length,
                forecast,
                opts.prediction_length,
                &warmup_len,
                &warmup_latency,
                error_buffer,
                sizeof(error_buffer)) != 0) {
            fprintf(stderr, "warmup failed: %s\n", error_buffer);
            free(context);
            free(actual);
            free(forecast);
            return 1;
        }
    }

    double total_outer_ms = 0.0;
    double total_inner_ms = 0.0;
    int forecast_len = 0;
    for (int i = 0; i < opts.runs; ++i) {
        double start_ms = now_ms();
        double inner_ms = 0.0;
        if (cutechronos_predict_median(
                handle,
                context,
                context_len,
                opts.prediction_length,
                forecast,
                opts.prediction_length,
                &forecast_len,
                &inner_ms,
                error_buffer,
                sizeof(error_buffer)) != 0) {
            fprintf(stderr, "predict failed: %s\n", error_buffer);
            free(context);
            free(actual);
            free(forecast);
            return 1;
        }
        total_outer_ms += now_ms() - start_ms;
        total_inner_ms += inner_ms;
    }

    double mae = compute_mae(forecast, forecast_len, actual, actual_len);
    double mape_pct = compute_mape_pct(forecast, forecast_len, actual, actual_len);
    printf("{\n");
    printf("  \"language\": \"c\",\n");
    printf("  \"backend\": \"%s\",\n", opts.backend);
    printf("  \"model_id\": \"%s\",\n", opts.model_id);
    printf("  \"device\": \"%s\",\n", opts.device);
    printf("  \"prediction_length\": %d,\n", opts.prediction_length);
    printf("  \"runs\": %d,\n", opts.runs);
    printf("  \"warmup\": %d,\n", opts.warmup);
    printf("  \"avg_outer_latency_ms\": %.6f,\n", total_outer_ms / (double)opts.runs);
    printf("  \"avg_inner_latency_ms\": %.6f,\n", total_inner_ms / (double)opts.runs);
    if (isnan(mae)) {
        printf("  \"mae\": null,\n");
    } else {
        printf("  \"mae\": %.6f,\n", mae);
    }
    if (isnan(mape_pct)) {
        printf("  \"mape_pct\": null,\n");
    } else {
        printf("  \"mape_pct\": %.6f,\n", mape_pct);
    }
    printf("  \"forecast\": ");
    print_json_array(forecast, forecast_len);
    printf(",\n  \"actual\": ");
    print_json_array(actual, actual_len);
    printf("\n}\n");

    if (cutechronos_destroy_pipeline(handle, error_buffer, sizeof(error_buffer)) != 0) {
        fprintf(stderr, "destroy warning: %s\n", error_buffer);
    }

    free(context);
    free(actual);
    free(forecast);
    return 0;
}
