"""TFLite conversion: float32, int8 post-training quantization, int8 QAT."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


def representative_generator(features: np.ndarray, count: int = 100) -> Iterable[list]:
    """Yield samples for PTQ calibration, one at a time."""
    subset = features[: min(count, len(features))]
    for row in subset:
        yield [row.reshape(1, -1).astype(np.float32)]


def _convert(converter: Any) -> bytes:
    return converter.convert()


def convert_float(model: Any, out_path: str | Path) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    data = _convert(converter)
    Path(out_path).write_bytes(data)
    return data


def convert_int8_ptq(model: Any, features: np.ndarray,
                     out_path: str | Path) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_generator(features)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = _convert(converter)
    Path(out_path).write_bytes(data)
    return data


def convert_int8_qat(qat_model: Any, features: np.ndarray,
                     out_path: str | Path) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_generator(features)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = _convert(converter)
    Path(out_path).write_bytes(data)
    return data
