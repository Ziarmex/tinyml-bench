"""Reproducible benchmarking primitives for TinyML models.

This module has no third-party dependencies so it can be unit-tested on the
host and reused by the training pipeline and the report generator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

BENCH_START = "<!-- BENCH:START -->"
BENCH_END = "<!-- BENCH:END -->"


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on an already-sorted sequence.

    p is in [0, 100]. Empty input yields NaN.
    """
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    rank = math.ceil(p / 100.0 * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return float(sorted_values[rank])


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples_ms(cls, samples: Iterable[float]) -> "LatencyStats":
        values = sorted(float(s) for s in samples)
        if not values:
            return cls(0, float("nan"), float("nan"), float("nan"),
                       float("nan"), float("nan"))
        return cls(
            count=len(values),
            mean_ms=sum(values) / len(values),
            p50_ms=percentile(values, 50),
            p95_ms=percentile(values, 95),
            min_ms=values[0],
            max_ms=values[-1],
        )


@dataclass
class BenchmarkResult:
    name: str
    precision: str
    model_bytes: int
    latency: LatencyStats
    device: str = "host"
    ram_bytes: int | None = None
    accuracy: float | None = None  # fraction in [0, 1]
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        latency = data["latency"]
        if isinstance(latency, dict):
            latency = LatencyStats(**latency)
        return cls(
            name=data["name"],
            precision=data["precision"],
            model_bytes=int(data["model_bytes"]),
            latency=latency,
            device=data.get("device", "host"),
            ram_bytes=data.get("ram_bytes"),
            accuracy=data.get("accuracy"),
            notes=data.get("notes", ""),
        )


def measure_latency(fn: Callable[[], None], iterations: int = 100,
                    warmup: int = 10,
                    clock: Callable[[], float] = time.perf_counter) -> LatencyStats:
    """Time `fn` in milliseconds after a warmup phase."""
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    for _ in range(max(0, warmup)):
        fn()
    samples = []
    for _ in range(iterations):
        start = clock()
        fn()
        samples.append((clock() - start) * 1000.0)
    return LatencyStats.from_samples_ms(samples)


def measure_file_size(path: str | Path) -> int:
    return Path(path).stat().st_size


def _fmt_kb(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1024.0:.1f}"


def _fmt_ms(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def results_to_markdown(results: Sequence[BenchmarkResult]) -> str:
    header = ("| Model | Precision | Size (KB) | p50 (ms) | p95 (ms) | "
              "RAM (KB) | Accuracy | Device | Notes |")
    separator = "|" + "---|" * 9
    lines = [header, separator]
    for r in results:
        accuracy = "n/a" if r.accuracy is None else f"{r.accuracy * 100:.1f}%"
        lines.append(
            f"| {r.name} | {r.precision} | {_fmt_kb(r.model_bytes)} | "
            f"{_fmt_ms(r.latency.p50_ms)} | {_fmt_ms(r.latency.p95_ms)} | "
            f"{_fmt_kb(r.ram_bytes)} | {accuracy} | {r.device} | {r.notes} |"
        )
    return "\n".join(lines)


def update_readme(readme_path: str | Path, table: str,
                  start: str = BENCH_START, end: str = BENCH_END) -> None:
    """Replace the region between `start` and `end` with `table`."""
    path = Path(readme_path)
    text = path.read_text()
    if start not in text or end not in text:
        raise ValueError(f"markers {start!r}/{end!r} not found in {path}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(f"{before}{start}\n{table}\n{end}{after}")


def results_to_json(results: Sequence[BenchmarkResult]) -> str:
    return json.dumps([r.to_dict() for r in results], indent=2) + "\n"


def results_from_json(text: str) -> list[BenchmarkResult]:
    return [BenchmarkResult.from_dict(d) for d in json.loads(text)]
