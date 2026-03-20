"""
scripts/run_predict.py — Entry-point for inference + evaluation statistics.

Loads a prepared experiment from disk, runs inference with a saved model
checkpoint, computes all statistics, and writes CSV outputs.

Usage
-----
  # Use defaults (last experiment, matching checkpoint):
  python scripts/run_predict.py

  # Override specific options:
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

from kvant.utils.pipeline_config import list_from_config, load_pipeline_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PREPARED_ROOT = _PROJECT_ROOT / "prepared"
_CHECKPOINTS_ROOT = _PROJECT_ROOT / "checkpoints"


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Path to pipeline TOML config.")
    pre_args, remaining = pre_parser.parse_known_args()
    cfg, cfg_path = load_pipeline_config(pre_args.config)

    default_tickers = list_from_config(cfg["predict"].get("tickers"))
    if default_tickers == []:
        default_tickers = None

    prepared_root = Path(cfg["paths"].get("prepared_root", str(_PREPARED_ROOT)))
    checkpoints_root = Path(cfg["paths"].get("checkpoints_root", str(_CHECKPOINTS_ROOT)))

    parser = argparse.ArgumentParser(
        description="Run inference with a trained kvant model and save evaluation CSVs.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--experiment-id",
        default=cfg["predict"].get("experiment_id", "last"),
        help="Prepared experiment ID under configured prepared_root, or 'last'.",
    )
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Path to a prepared experiment directory. Defaults to the last prepared experiment.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a saved model checkpoint. Defaults to checkpoints/<experiment-id>/<model>/.",
    )
    parser.add_argument(
        "--model",
        default=cfg["predict"].get("model", "conv1d"),
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
        default=cfg["predict"].get("split", "test"),
        choices=["train", "val", "test"],
        help="Which data split to evaluate (default: test).",
    )
    parser.add_argument(
        "--tickers",
        default=default_tickers,
        nargs="+",
        help="Optional: evaluate only these tickers (e.g. --tickers AAPL MSFT).",
    )
    args = parser.parse_args(remaining)

    # Resolve experiment directory
    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir)
    else:
        exp_id = args.experiment_id
        if exp_id == "last":
            last_file = prepared_root / "last_experiment.txt"
            if not last_file.exists():
                raise SystemExit(f"No last_experiment.txt found in {prepared_root}. Pass --exp-dir explicitly.")
            exp_id = last_file.read_text().strip()
            print(f"Auto-detected experiment: {exp_id}")
        exp_dir = prepared_root / exp_id

    # Resolve checkpoint
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
    else:
        checkpoint = checkpoints_root / exp_dir.name / args.model
        if not (checkpoint / "weights.pt").exists():
            raise SystemExit(
                f"No checkpoint found at {checkpoint}/weights.pt. "
                f"Train first with: uv run --env-file .env.run scripts/run_train.py --experiment-id {exp_dir.name}"
            )
        print(f"Auto-detected checkpoint: {checkpoint}")

    print(f"Config: {cfg_path}")

    # Resolve model class from registry
    from kvant.models import MODEL_REGISTRY
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{args.model}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[args.model]

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
