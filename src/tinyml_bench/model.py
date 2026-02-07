"""Autoencoder definition and quantization-aware training wrapper.

TensorFlow is imported lazily so the rest of the package (and its host tests)
does not depend on it.
"""
from __future__ import annotations

from typing import Any

from .data import N_FEATURES


def build_autoencoder(input_dim: int = N_FEATURES, latent_dim: int = 8) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(latent)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear",
                                    name="reconstruction")(x)

    model = tf.keras.Model(inputs, outputs, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_qat_autoencoder(input_dim: int = N_FEATURES, latent_dim: int = 8) -> Any:
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    model = build_autoencoder(input_dim, latent_dim)
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(optimizer="adam", loss="mse")
    return qat_model


def reconstruction_errors(model: Any, features: Any) -> Any:
    import numpy as np

    reconstructed = model.predict(features, verbose=0)
    return np.mean(np.square(features - reconstructed), axis=1)
