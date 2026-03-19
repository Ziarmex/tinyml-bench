import math

import pytest

from tinyml_bench import (
    BenchmarkResult,
    LatencyStats,
    measure_file_size,
    measure_latency,
    percentile,
    results_from_json,
    results_to_json,
    results_to_markdown,
)


def test_percentile_nearest_rank():
    values = list(range(1, 11))  # 1..10
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile(values, 100) == 10


def test_percentile_empty_is_nan():
    assert math.isnan(percentile([], 50))


def test_latency_stats_from_samples():
    stats = LatencyStats.from_samples_ms([1, 2, 3, 4, 5])
    assert stats.count == 5
    assert stats.mean_ms == pytest.approx(3.0)
    assert stats.p50_ms == 3
    assert stats.min_ms == 1
    assert stats.max_ms == 5


def test_measure_latency_counts_calls():
    calls = {"n": 0}

    def work():
        calls["n"] += 1

    stats = measure_latency(work, iterations=20, warmup=5)
    assert calls["n"] == 25
    assert stats.count == 20
    assert stats.mean_ms >= 0


def test_measure_latency_rejects_bad_iterations():
    with pytest.raises(ValueError):
        measure_latency(lambda: None, iterations=0)


def test_measure_file_size(tmp_path):
    path = tmp_path / "model.tflite"
    path.write_bytes(b"x" * 2048)
    assert measure_file_size(path) == 2048


def _sample_result():
    return BenchmarkResult(
        name="autoencoder",
        precision="int8",
        model_bytes=25600,
        latency=LatencyStats(100, 1.0, 0.9, 1.4, 0.8, 2.0),
        device="esp32",
        ram_bytes=30720,
        accuracy=0.93,
        notes="PTQ",
    )


def test_json_roundtrip():
    result = _sample_result()
    restored = results_from_json(results_to_json([result]))
    assert len(restored) == 1
    assert restored[0].name == result.name
    assert restored[0].latency.p95_ms == pytest.approx(result.latency.p95_ms)
    assert restored[0].ram_bytes == result.ram_bytes
    assert restored[0].accuracy == pytest.approx(0.93)


def test_markdown_contains_expected_cells():
    table = results_to_markdown([_sample_result()])
    assert "| Model | Precision |" in table
    assert "25.0" in table        # 25600 bytes -> 25.0 KB
    assert "30.0" in table        # 30720 bytes -> 30.0 KB
    assert "93.0%" in table
    assert "esp32" in table


def test_markdown_handles_missing_optional_fields():
    result = BenchmarkResult(
        name="float",
        precision="float32",
        model_bytes=1024,
        latency=LatencyStats(1, 2.0, 2.0, 2.0, 2.0, 2.0),
    )
    table = results_to_markdown([result])
    assert table.count("n/a") >= 2
