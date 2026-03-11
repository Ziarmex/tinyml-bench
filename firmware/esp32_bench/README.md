# On-device ESP32 benchmark

Measures **latency** and **tensor-arena RAM** for a TFLite Micro model on a real
classic ESP32, and prints a line that can be merged into `results/results.json`.

## 1. Generate the model header

From the repository root, after training (see top-level README):

```bash
python tools/export_tflite_header.py \
    results/autoencoder_int8_ptq.tflite \
    firmware/esp32_bench/main/model_data.h \
    --name model_data
```

`bench_main.c` includes `model_data.h`; if it is missing the build stops with a
clear message.

## 2. Build and flash

```bash
cd firmware/esp32_bench
idf.py set-target esp32
idf.py -DMODEL_NAME="autoencoder" -DMODEL_PRECISION="int8 (PTQ)" build flash monitor
```

Optionally tune the tensor arena:

```bash
idf.py -DMODEL_NAME="autoencoder" -DMODEL_PRECISION="int8 (PTQ)" \
       -DBENCH_ARENA_BYTES=32768 build flash monitor
```

## 3. Read the result

The monitor prints a human summary and a parseable line:

```
BENCH_JSON {"device":"esp32","model":"autoencoder","precision":"int8 (PTQ)",
"size_bytes":24832,"ram_bytes":20160,"iterations":100,
"latency_ms":{"mean":3.12,"p50":3.08,"p95":3.41,"min":3.01,"max":4.02}}
```

Add that JSON object to `results/results.json` (it matches the `BenchmarkResult`
schema), then regenerate the README table:

```bash
python tools/gen_report.py
```

Measure latency at a fixed CPU frequency for reproducible numbers, and keep the
console quiet during the run (this harness does not use Wi-Fi).
