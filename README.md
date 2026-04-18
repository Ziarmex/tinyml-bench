# tinyml-bench

A reproducible benchmark harness for TinyML on microcontrollers: train an
autoencoder, convert it to **float32**, **int8 post-training quantization
(PTQ)** and **int8 quantization-aware training (QAT)**, then compare them on
**size, latency, RAM and accuracy** - on the host and on a real ESP32.

The point is not "I quantized a model". It is: can you *prove* the trade-off?
Every number here comes from a command that can be re-run, and the comparison
table is generated from a JSON results file, not typed by hand.

## Why this exists

Most edge-AI portfolios say "95% accuracy" with no sizes, no latency and no
method. That is unverifiable. This harness makes the claim falsifiable:

- model size in bytes (from the `.tflite` files),
- Invoke() latency p50/p95 (host and **on-device, measured with `esp_timer`**),
- tensor-arena RAM on the device (`arena_used_bytes()`),
- accuracy of anomaly detection at a per-model, per-precision threshold.

## Pipeline

```
synthetic vibration ──FFT──> 64-D features
        │
        ├─ train autoencoder (64-32-16-8-16-32-64)
        │
        ├─ float32 .tflite
        ├─ int8 PTQ .tflite   (representative dataset)
        └─ int8 QAT .tflite   (tensorflow_model_optimization)
                    │
      host: tflite interpreter (latency)   device: ESP32 + TFLite Micro
                    │                              │
                    └──────── results.json ────────┘
                                   │
                        tools/gen_report.py -> README table
```

## Benchmark results

<!-- BENCH:START -->
| Model | Precision | Size (KB) | p50 (ms) | p95 (ms) | RAM (KB) | Accuracy | Device | Notes |
|---|---|---|---|---|---|---|---|---|
| autoencoder | float32 | 24.8 | 0.002 | 0.004 | n/a | 97.1% | host | FP32 baseline |
| autoencoder | int8 (PTQ) | 13.5 | 0.002 | 0.003 | n/a | 97.4% | host | post-training quantization |
| autoencoder | int8 (QAT) | 10.3 | 0.002 | 0.002 | n/a | 97.0% | host | quantization-aware training |
<!-- BENCH:END -->

> Accuracy is anomaly-detection accuracy on a held-out split of the
> **synthetic** dataset; the host latency is x86 and not comparable to the
> ESP32. Treat the synthetic model as a pipeline demonstration, not a validated
> bearing-fault detector.

## Layout

```
src/tinyml_bench/
  bench.py       size/latency/percentile/table/README-update primitives (stdlib only)
  data.py        synthetic vibration dataset + FFT features (numpy)
  model.py       autoencoder + QAT wrapper (tensorflow)
  quantize.py    float / int8 PTQ / int8 QAT conversion (tensorflow)
  host_bench.py  tflite interpreter latency + reconstruction errors
  pipeline.py    train -> convert -> benchmark -> results.json
  cli.py         python -m tinyml_bench {train,report}
tools/
  export_tflite_header.py   .tflite -> C array header for TFLite Micro
  gen_report.py             results.json -> README table
firmware/esp32_bench/       ESP32 on-device latency/RAM harness
tests/                      stdlib-only tests (run in CI)
```

## Quick start

### Host only (fast, no TensorFlow)

```bash
make test            # runs the benchmark/report unit tests
```

### Full pipeline (needs TensorFlow)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # makes `python -m tinyml_bench` work
pip install -r requirements-train.txt

python -m tinyml_bench train --windows 4000 --epochs 30 --device host
python tools/gen_report.py
```

This writes `results/autoencoder_{float,int8_ptq,int8_qat}.tflite` and
`results/results.json`, then updates the table in this README.

### On-device ESP32

```bash
python tools/export_tflite_header.py \
    results/autoencoder_int8_ptq.tflite \
    firmware/esp32_bench/main/model_data.h --name model_data

cd firmware/esp32_bench
idf.py set-target esp32
idf.py -DMODEL_NAME="autoencoder" -DMODEL_PRECISION="int8 (PTQ)" build flash monitor
```

Copy the emitted `BENCH_JSON {...}` object into `results/results.json`, then
re-run `python tools/gen_report.py`. See
[firmware/esp32_bench/README.md](firmware/esp32_bench/README.md).

## Method

See [docs/benchmark-method.md](docs/benchmark-method.md) for the exact
procedure, the nearest-rank percentile definition, and why host and device
latency are reported separately.

## Tests

```bash
python -m pytest tests -q
```

The tests cover the size/latency/percentile math, the README table injection
(idempotent, marker-safe) and the header exporter - all with the standard
library, so CI needs no TensorFlow.

## Limitations

- The default dataset is **synthetic**; swap in a real bearing dataset via
  `data.load_feature_file()` to make accuracy meaningful.
- An autoencoder's "accuracy" depends on the threshold rule; here it is
  `mean + 3*sigma` of normal training errors, recomputed per model.
- Host latency is not MCU latency; only the ESP32 column is an embedded number.
- TFLite Micro op coverage is limited to the ops this model uses (see the
  resolver in `bench_main.c`). A model with unsupported ops will fail
  `AllocateTensors`.
- QAT requires `tensorflow-model-optimization`; use `--no-qat` to skip it.

## Roadmap

- CMSIS-NN / ESP-NN backend comparison.
- Edge Impulse EON baseline for cross-tool comparison.
- `.ioc`-style fixed dataset hash recorded in `results.json`.

## License

MIT - see [LICENSE](LICENSE).
