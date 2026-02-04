"""tinyml-bench: reproducible edge-AI benchmarking."""
from .bench import (
    BENCH_END,
    BENCH_START,
    BenchmarkResult,
    LatencyStats,
    measure_file_size,
    measure_latency,
    percentile,
    results_from_json,
    results_to_json,
    results_to_markdown,
    update_readme,
)

__all__ = [
    "BENCH_END",
    "BENCH_START",
    "BenchmarkResult",
    "LatencyStats",
    "measure_file_size",
    "measure_latency",
    "percentile",
    "results_from_json",
    "results_to_json",
    "results_to_markdown",
    "update_readme",
]

__version__ = "0.1.0"
