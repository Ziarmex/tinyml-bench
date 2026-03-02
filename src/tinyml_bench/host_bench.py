"""Host-side benchmarking of TFLite models via the TensorFlow interpreter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bench import LatencyStats, measure_latency


def load_interpreter(path: str | Path) -> Any:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def benchmark_tflite(path: str | Path, iterations: int = 200,
                     warmup: int = 20) -> LatencyStats:
    interpreter = load_interpreter(path)
    detail = interpreter.get_input_details()[0]
    data = np.zeros(detail["shape"], dtype=detail["dtype"])

    def invoke() -> None:
        interpreter.set_tensor(detail["index"], data)
        interpreter.invoke()

    return measure_latency(invoke, iterations=iterations, warmup=warmup)


def _quantize_input(row: np.ndarray, detail: dict) -> np.ndarray:
    if detail["dtype"] == np.int8:
        scale, zero_point = detail["quantization"]
        q = np.round(row / scale + zero_point)
        q = np.clip(q, -128, 127).astype(np.int8)
    else:
        q = row.astype(detail["dtype"])
    return q.reshape(detail["shape"])


def _dequantize_output(data: np.ndarray, detail: dict) -> np.ndarray:
    flat = data.flatten().astype(np.float32)
    if detail["dtype"] == np.int8:
        scale, zero_point = detail["quantization"]
        return (flat - zero_point) * scale
    return flat


def tflite_reconstruction_errors(path: str | Path,
                                 features: np.ndarray) -> np.ndarray:
    interpreter = load_interpreter(path)
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    errors = np.empty(len(features), dtype=np.float32)
    for i, row in enumerate(features):
        interpreter.set_tensor(in_detail["index"], _quantize_input(row, in_detail))
        interpreter.invoke()
        reconstructed = _dequantize_output(
            interpreter.get_tensor(out_detail["index"]), out_detail)
        errors[i] = float(np.mean((row - reconstructed) ** 2))
    return errors
