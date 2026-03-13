"""
scripts/run_predict.py — Entry-point for inference + evaluation statistics.

Loads a prepared experiment from disk, runs inference with a saved model
checkpoint, computes all statistics, and writes CSV outputs.

Usage
-----
  python scripts/run_predict.py \\
      --exp-dir   prepared/<experiment_id> \\
      --checkpoint checkpoints/<run>/     \\
      --model     conv1d                  \\
      --out-dir   results/<run>/          \\
      --split     test

Available model names: conv1d, conv3d, resnls, tsb
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained kvant model and save evaluation CSVs."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Path to a prepared experiment directory (contains train/val/test sub-dirs).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved model checkpoint (directory produced by model.save()).",
    )
    parser.add_argument(
        "--model",
        default="conv1d",
        help="Model architecture key — one of: conv1d, conv3d, resnls, tsb.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory where evaluation CSVs will be written. "
            "Defaults to <exp-dir>/eval/<model>_<split>/"
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate (default: test).",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        nargs="+",
        help="Optional: evaluate only these tickers (e.g. --tickers AAPL MSFT).",
    )
    args = parser.parse_args()

    # Resolve model class from registry
    from kvant.models import MODEL_REGISTRY
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{args.model}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[args.model]

    exp_dir    = Path(args.exp_dir)
    checkpoint = Path(args.checkpoint)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = exp_dir / "eval" / f"{args.model}_{args.split}"

    from kvant.evaluation import evaluate_experiment
    evaluate_experiment(
        exp_dir    = exp_dir,
        model_path = checkpoint,
        model_cls  = model_cls,
        out_dir    = out_dir,
        split      = args.split,
        tickers    = args.tickers,
    )


if __name__ == "__main__":
    main()
