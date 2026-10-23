"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark_stats, load_benchmark
from .config import ExperimentConfig
from .experiment import ExperimentRunner
from .llm import BudgetExceeded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experience-memory agent pilot")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate configuration and benchmark")
    validate.add_argument("--config", default="config/experiment_v0.json")

    run = sub.add_parser("run", help="run the complete answer-correct-learn-retrieve loop")
    run.add_argument("--config", default="config/experiment_v0.json")
    run.add_argument("--output", default=None)
    run.add_argument(
        "--offline",
        action="store_true",
        help="use a deterministic mock model; no API calls",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    config = ExperimentConfig.load(root / args.config)
    benchmark = root / config.benchmark
    if args.command == "validate":
        items = load_benchmark(benchmark)
        print(json.dumps(benchmark_stats(items), indent=2))
        return 0
    if args.command == "run":
        try:
            runner = ExperimentRunner(
                root=root,
                config_path=args.config,
                output_dir=args.output,
                offline=args.offline,
            )
            summary = runner.run()
        except BudgetExceeded as exc:
            print(f"BUDGET STOP: {exc}")
            return 2
        print(f"Run directory: {runner.output_dir}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary.get("status") == "completed" else 3
    return 1

