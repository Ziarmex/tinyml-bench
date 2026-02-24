"""End-to-end pipeline: train -> convert -> benchmark -> results.json.

Run with TensorFlow installed (requirements-train.txt). The benchmark table is
regenerated separately by tools/gen_report.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import host_bench, model as model_mod, quantize
from .bench import BenchmarkResult, results_to_json

DEFAULT_OUT = Path("results")


def _module_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


@dataclass
class PipelineConfig:
    windows: int = 4000
    epochs: int = 30
    seed: int = 0
    device: str = "host"
    do_qat: bool = True
    out_dir: Path = DEFAULT_OUT


def _detect_accuracy(errors_fn, normal_train, test_x, test_y,
                     k: float = 3.0) -> tuple[float, float]:
    train_errors = errors_fn(normal_train)
    threshold = float(train_errors.mean() + k * train_errors.std())
    test_errors = errors_fn(test_x)
    predictions = (test_errors > threshold).astype(np.int32)
    accuracy = float((predictions == test_y).mean())
    return accuracy, threshold


def run(config: PipelineConfig) -> list[BenchmarkResult]:
    # tensorflow-model-optimization still targets the legacy Keras API.
    # The variable must be set before TensorFlow is first imported.
    if config.do_qat:
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, labels = data_mod.make_dataset(n_windows=config.windows,
                                             seed=config.seed)
    train_x, train_y, test_x, test_y = data_mod.split(features, labels,
                                                      seed=config.seed)
    normal_train = train_x[train_y == 0]

    print(f"dataset: {len(features)} windows, "
          f"{len(normal_train)} normal training samples")

    ae = model_mod.build_autoencoder()
    ae.fit(normal_train, normal_train, epochs=config.epochs, batch_size=32,
           verbose=2)

    float_acc, _ = _detect_accuracy(
        lambda x: model_mod.reconstruction_errors(ae, x),
        normal_train, test_x, test_y)

    results: list[BenchmarkResult] = []

    float_path = out_dir / "autoencoder_float.tflite"
    quantize.convert_float(ae, float_path)
    results.append(BenchmarkResult(
        name="autoencoder", precision="float32",
        model_bytes=float_path.stat().st_size,
        latency=host_bench.benchmark_tflite(float_path),
        device=config.device, accuracy=float_acc, notes="FP32 baseline"))

    ptq_path = out_dir / "autoencoder_int8_ptq.tflite"
    quantize.convert_int8_ptq(ae, normal_train, ptq_path)
    ptq_acc, _ = _detect_accuracy(
        lambda x: host_bench.tflite_reconstruction_errors(ptq_path, x),
        normal_train, test_x, test_y)
    results.append(BenchmarkResult(
        name="autoencoder", precision="int8 (PTQ)",
        model_bytes=ptq_path.stat().st_size,
        latency=host_bench.benchmark_tflite(ptq_path),
        device=config.device, accuracy=ptq_acc,
        notes="post-training quantization"))

    if config.do_qat:
        qat_model = model_mod.build_qat_autoencoder()
        qat_model.fit(normal_train, normal_train, epochs=config.epochs,
                      batch_size=32, verbose=2)
        qat_path = out_dir / "autoencoder_int8_qat.tflite"
        quantize.convert_int8_qat(qat_model, normal_train, qat_path)
        qat_acc, _ = _detect_accuracy(
            lambda x: host_bench.tflite_reconstruction_errors(qat_path, x),
            normal_train, test_x, test_y)
        results.append(BenchmarkResult(
            name="autoencoder", precision="int8 (QAT)",
            model_bytes=qat_path.stat().st_size,
            latency=host_bench.benchmark_tflite(qat_path),
            device=config.device, accuracy=qat_acc,
            notes="quantization-aware training"))

    results_path = out_dir / "results.json"
    results_path.write_text(results_to_json(results))

    config_path = out_dir / "run-config.json"
    config_path.write_text(json.dumps({
        "windows": config.windows,
        "epochs": config.epochs,
        "seed": config.seed,
        "device": config.device,
        "do_qat": config.do_qat,
        "dataset": "synthetic vibration (data.py)",
        "tensorflow": _module_version("tensorflow"),
        "numpy": _module_version("numpy"),
    }, indent=2) + "\n")

    print(f"wrote {results_path} and {config_path}")
    return results
