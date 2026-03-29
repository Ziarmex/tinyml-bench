# Benchmark method

## Reproducibility contract

Every published number must come from a command in this repository and carry
its provenance. The results schema is intentionally small:

```json
{
  "name": "autoencoder",
  "precision": "int8 (PTQ)",
  "model_bytes": 24832,
  "latency": {"count": 100, "mean_ms": 3.12, "p50_ms": 3.08,
               "p95_ms": 3.41, "min_ms": 3.01, "max_ms": 4.02},
  "device": "esp32",
  "ram_bytes": 20160,
  "accuracy": 0.93,
  "notes": "post-training quantization"
}
```

`tools/gen_report.py` turns a list of these into the README table. The README
region between `<!-- BENCH:START -->` and `<!-- BENCH:END -->` is machine
generated; do not hand-edit it.

## Latency

- **Host**: `tensorflow.lite.Interpreter`, one `invoke()` per timed sample, no
  input mutation (embedded inference latency on an MCU is data-independent for
  dense layers, and keeping input fixed removes allocator noise).
- **Device**: `esp_timer_get_time()` around `MicroInterpreter::Invoke()`, fixed
  CPU frequency, console quiet, warmup runs excluded.
- **Warmup**: 10-20 runs discarded before sampling.
- **Statistics**: `count`, `mean`, `p50`, `p95`, `min`, `max`. Tail latency
  (p95) matters more than the mean for hard-period systems.

### Percentile definition

Nearest-rank on the sorted samples: `rank = ceil(p/100 * n) - 1`, clamped to
`[0, n-1]`. The device harness implements the same rule so host and device
numbers agree. For `n=100`, p50 = sample[49], p95 = sample[94].

## Size and RAM

- **Model size**: exact file size of the `.tflite` flatbuffer in bytes.
- **RAM (device)**: `MicroInterpreter::arena_used_bytes()` after
  `AllocateTensors()`. This is the tensor-arena high-water mark, the number that
  actually constrains SRAM. It is not the same as model size.
- **RAM (host)**: reported as `n/a`; the full TFLite interpreter does not expose
  a comparable arena.

## Accuracy

The autoencoder is trained only on normal windows. Per model and precision:

1. compute reconstruction MSE on the normal training set,
2. threshold = `mean + 3 * std`,
3. flag test windows with error above the threshold as anomalies,
4. accuracy = fraction of correct binary labels on the held-out split.

Recomputing the threshold per model is deliberate: comparing float and int8
accuracy using a single float threshold would understate quantization
degradation.

## What is deliberately not claimed

- No accuracy claim on real machinery (synthetic data by default).
- No power measurement here (see the FreeRTOS sensor node repo for that).
- No cross-MCU comparison until the same model is run on more than one target.
