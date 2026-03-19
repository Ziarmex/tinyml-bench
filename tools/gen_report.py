#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate the benchmark table in README.md from results/results.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml_bench import results_from_json, results_to_markdown, update_readme  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results" / "results.json"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"no results file at {results_path}", file=sys.stderr)
        return 1

    results = results_from_json(results_path.read_text())
    update_readme(args.readme, results_to_markdown(results))
    print(f"Updated {args.readme} with {len(results)} result(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
