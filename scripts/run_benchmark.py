"""
scripts/run_benchmark.py — Run the model benchmark suite.

Strategies
----------
  - council + meta
  - configured single model + meta
  - one-layer CNN weak learned baseline
  - random trading baseline over multiple seeds

Usage
-----
  uv run --env-file .env.run scripts/run_benchmark.py --config pipeline.toml
  uv run --env-file .env.run scripts/run_benchmark.py --config pipeline.toml --benchmark-id final_test
"""
from __future__ import annotations

import argparse

from kvant.benchmarking import run_benchmark
from kvant.utils.pipeline_config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suite for the current kvant setup.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    parser.add_argument("--benchmark-id", default=None, help="Output id under prepared/<experiment>/benchmark/.")
    parser.add_argument("--random-seeds", type=int, default=None, help="Number of random baseline seeds.")
    parser.add_argument(
        "--no-train-shallow",
        action="store_true",
        help="Fail if shallow_cnn is missing instead of training it automatically.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV outputs but skip benchmark-level figures.",
    )
    args = parser.parse_args()

    cfg, cfg_path = load_pipeline_config(args.config)
    print(f"Config: {cfg_path}")

    result = run_benchmark(
        cfg,
        benchmark_id=args.benchmark_id,
        random_seeds=args.random_seeds,
        train_shallow=not args.no_train_shallow,
        make_plots=not args.no_plots,
    )

    print("\nBenchmark complete")
    print(f"  Directory : {result.benchmark_dir}")
    print(f"  Summary   : {result.summary_csv}")
    print(f"  Equity    : {result.equity_comparison_csv}")
    print(f"  Figures   : {result.figures_dir}")


if __name__ == "__main__":
    main()
