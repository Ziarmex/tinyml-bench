"""Synthetic vibration dataset for predictive maintenance.

The default dataset is generated on the fly so results are reproducible without
shipping data. A real bearing dataset (e.g. CWRU) can be dropped in by
implementing ``load_feature_file``; nothing else changes.

Honesty note: models trained on synthetic impulses are a *pipeline
demonstration*, not a validated bearing-fault detector. Report them as such.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_FEATURES = 64


def generate_signal(window: int, fs: int, fault: bool,
                    rng: np.random.Generator) -> np.ndarray:
    t = np.arange(window) / float(fs)
    signal = np.sin(2.0 * np.pi * 50.0 * t) + 0.5 * np.sin(2.0 * np.pi * 120.0 * t)
    signal = signal + 0.1 * rng.standard_normal(window)

    if fault:
        n_impulses = max(1, window // 200)
        idx = rng.integers(0, window, size=n_impulses)
        signal[idx] += rng.uniform(1.0, 3.0, size=n_impulses)
    return signal.astype(np.float32)


def fft_features(signal: np.ndarray, n_features: int = N_FEATURES) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(signal))[:n_features]
    if spectrum.size < n_features:
        spectrum = np.pad(spectrum, (0, n_features - spectrum.size))
    return np.log1p(spectrum).astype(np.float32)


def make_dataset(n_windows: int = 4000, window: int = 128, fs: int = 1000,
                 fault_ratio: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = np.empty((n_windows, N_FEATURES), dtype=np.float32)
    labels = np.empty((n_windows,), dtype=np.int32)

    n_fault = int(round(n_windows * fault_ratio))
    fault_flags = np.zeros(n_windows, dtype=bool)
    fault_flags[:n_fault] = True
    rng.shuffle(fault_flags)

    for i in range(n_windows):
        signal = generate_signal(window, fs, bool(fault_flags[i]), rng)
        features[i] = fft_features(signal)
        labels[i] = 1 if fault_flags[i] else 0

    return features, labels


def split(features: np.ndarray, labels: np.ndarray, test_ratio: float = 0.2,
          seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    n_test = int(len(features) * test_ratio)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return (features[train_idx], labels[train_idx],
            features[test_idx], labels[test_idx])


def load_feature_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a precomputed (features, labels) .npz file.

    Expected keys: ``features`` (N, N_FEATURES) and ``labels`` (N,).
    """
    data = np.load(Path(path))
    return data["features"].astype(np.float32), data["labels"].astype(np.int32)
