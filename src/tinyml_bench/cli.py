"""Command-line interface: python -m tinyml_bench ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    from .pipeline import PipelineConfig, run

    config = PipelineConfig(
        windows=args.windows,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        do_qat=not args.no_qat,
        out_dir=Path(args.out_dir),
    )
    run(config)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .bench import results_from_json, results_to_markdown, update_readme

    results = results_from_json(Path(args.results).read_text())
    update_readme(args.readme, results_to_markdown(results))
    print(f"Updated {args.readme}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyml_bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train, quantize and benchmark")
    train.add_argument("--windows", type=int, default=4000)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="host")
    train.add_argument("--out-dir", default="results")
    train.add_argument("--no-qat", action="store_true",
                       help="skip quantization-aware training")
    train.set_defaults(func=_cmd_train)

    report = sub.add_parser("report", help="regenerate README benchmark table")
    report.add_argument("--results", default="results/results.json")
    report.add_argument("--readme", default="README.md")
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
