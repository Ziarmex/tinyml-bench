import pytest

from tinyml_bench import (
    BENCH_END,
    BENCH_START,
    BenchmarkResult,
    LatencyStats,
    results_to_markdown,
    update_readme,
)


def _result(precision="int8", size=1024, p50=1.0):
    return BenchmarkResult(
        name="autoencoder",
        precision=precision,
        model_bytes=size,
        latency=LatencyStats(10, p50, p50, p50, p50, p50),
        device="esp32",
    )


def test_update_readme_replaces_region(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\nIntro\n\n"
        f"{BENCH_START}\nold table\n{BENCH_END}\n\nFooter\n"
    )
    table = results_to_markdown([_result()])
    update_readme(readme, table)
    text = readme.read_text()
    assert "old table" not in text
    assert "Footer" in text
    assert "| Model | Precision |" in text
    # markers must still be present and in order
    assert text.index(BENCH_START) < text.index(BENCH_END)


def test_update_readme_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"a\n{BENCH_START}\n{BENCH_END}\nb\n")
    table = results_to_markdown([_result()])
    update_readme(readme, table)
    once = readme.read_text()
    update_readme(readme, table)
    assert readme.read_text() == once


def test_update_readme_missing_markers_raises(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")
    with pytest.raises(ValueError):
        update_readme(readme, "table")
