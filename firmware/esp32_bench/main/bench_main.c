/* SPDX-License-Identifier: MIT
 *
 * On-device TFLite Micro benchmark for the classic ESP32.
 *
 * Measures Invoke() latency over BENCH_ITERATIONS runs and reports the tensor
 * arena actually used. Prints one parseable BENCH_JSON line that matches the
 * BenchmarkResult schema used by tools/gen_report.py.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#if defined(__has_include)
#  if !__has_include("model_data.h")
#    error "model_data.h not found. Generate it: python tools/export_tflite_header.py <model.tflite> firmware/esp32_bench/main/model_data.h"
#  endif
#endif
#include "model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef MODEL_NAME
#define MODEL_NAME "model"
#endif
#ifndef MODEL_PRECISION
#define MODEL_PRECISION "unknown"
#endif
#ifndef BENCH_ARENA_BYTES
#define BENCH_ARENA_BYTES (32 * 1024)
#endif
#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 100
#endif
#ifndef BENCH_WARMUP
#define BENCH_WARMUP 10
#endif

static const char *TAG = "bench";

static uint8_t s_tensor_arena[BENCH_ARENA_BYTES] __attribute__((aligned(16)));

static int compare_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static uint32_t percentile_u32(const uint32_t *sorted, int n, int p)
{
    if (n <= 0) {
        return 0;
    }
    if (p <= 0) {
        return sorted[0];
    }
    if (p >= 100) {
        return sorted[n - 1];
    }
    int rank = (p * n + 99) / 100 - 1; /* nearest-rank, matches host impl */
    if (rank < 0) {
        rank = 0;
    }
    if (rank >= n) {
        rank = n - 1;
    }
    return sorted[rank];
}

static void zero_input(TfLiteTensor *input)
{
    if (input->type == kTfLiteInt8) {
        memset(input->data.int8, 0, (size_t)input->bytes);
    } else if (input->type == kTfLiteFloat32) {
        for (int i = 0; i < input->bytes / (int)sizeof(float); i++) {
            input->data.f[i] = 0.0f;
        }
    } else {
        memset(input->data.int8, 0, (size_t)input->bytes);
    }
}

void app_main(void)
{
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter *error_reporter = &micro_error_reporter;

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %lu != runtime %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        model, resolver, s_tensor_arena, sizeof(s_tensor_arena), error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed - increase BENCH_ARENA_BYTES");
        return;
    }

    zero_input(interpreter.input(0));

    for (int i = 0; i < BENCH_WARMUP; i++) {
        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "warmup Invoke failed");
            return;
        }
    }

    static uint32_t samples[BENCH_ITERATIONS];
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        int64_t t0 = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed at iteration %d", i);
            return;
        }
        samples[i] = (uint32_t)(esp_timer_get_time() - t0);
    }

    qsort(samples, BENCH_ITERATIONS, sizeof(uint32_t), compare_u32);

    uint64_t sum = 0;
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        sum += samples[i];
    }
    double mean_ms = (double)sum / BENCH_ITERATIONS / 1000.0;
    double p50_ms = percentile_u32(samples, BENCH_ITERATIONS, 50) / 1000.0;
    double p95_ms = percentile_u32(samples, BENCH_ITERATIONS, 95) / 1000.0;
    double min_ms = samples[0] / 1000.0;
    double max_ms = samples[BENCH_ITERATIONS - 1] / 1000.0;
    size_t arena_used = interpreter.arena_used_bytes();

    ESP_LOGI(TAG, "%s / %s: p50=%.3f ms p95=%.3f ms arena=%u B size=%u B",
             MODEL_NAME, MODEL_PRECISION, p50_ms, p95_ms,
             (unsigned)arena_used, (unsigned)model_data_len);

    printf("BENCH_JSON {\"device\":\"esp32\",\"model\":\"%s\","
           "\"precision\":\"%s\",\"size_bytes\":%u,\"ram_bytes\":%u,"
           "\"iterations\":%d,\"latency_ms\":{\"mean\":%.3f,\"p50\":%.3f,"
           "\"p95\":%.3f,\"min\":%.3f,\"max\":%.3f}}\n",
           MODEL_NAME, MODEL_PRECISION, (unsigned)model_data_len,
           (unsigned)arena_used, BENCH_ITERATIONS,
           mean_ms, p50_ms, p95_ms, min_ms, max_ms);
}
