"""
scripts/run_predict.py — Entry-point for inference + evaluation statistics.

Loads a prepared experiment from disk, runs inference with a saved model
checkpoint, computes all statistics, and writes CSV outputs.

Usage
-----
  python scripts/run_predict.py
  python scripts/run_predict.py --config pipeline.toml

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
    parser = argparse.ArgumentParser(
        description="Run inference with a trained kvant model and save evaluation CSVs."
    )
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()
    cfg, cfg_path = load_pipeline_config(args.config)

    prepared_root = Path(cfg["paths"].get("prepared_root", str(_PREPARED_ROOT)))
    checkpoints_root = Path(cfg["paths"].get("checkpoints_root", str(_CHECKPOINTS_ROOT)))
    predict_cfg = cfg["predict"]

    exp_id = str(predict_cfg.get("experiment_id", "last"))
    if exp_id == "last":
        last_file = prepared_root / "last_experiment.txt"
        if not last_file.exists():
            raise SystemExit(f"No last_experiment.txt found in {prepared_root}.")
        exp_id = last_file.read_text().strip()
        print(f"Auto-detected experiment: {exp_id}")
    exp_dir = prepared_root / exp_id

    model_name = str(predict_cfg.get("model", "conv1d"))
    split = str(predict_cfg.get("split", "test"))
    tickers = list_from_config(predict_cfg.get("tickers"))
    tickers = tickers if tickers else None
    required_buy_probability = float(predict_cfg.get("required_buy_probability", 0.0))
    required_sell_probability = float(predict_cfg.get("required_sell_probability", 0.0))

    checkpoint = checkpoints_root / exp_dir.name / model_name
    if not (checkpoint / "weights.pt").exists():
        raise SystemExit(
            f"No checkpoint found at {checkpoint}/weights.pt. "
            f"Train first with: uv run --env-file .env.run scripts/run_train.py"
        )
    print(f"Auto-detected checkpoint: {checkpoint}")

    print(f"Config: {cfg_path}")

    # Resolve model class from registry
    from kvant.models import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[model_name]

    out_dir = exp_dir / "eval" / f"{model_name}_{split}"

    from kvant.evaluation import evaluate_experiment
    evaluate_experiment(
        exp_dir    = exp_dir,
        model_path = checkpoint,
        model_cls  = model_cls,
        out_dir    = out_dir,
        split      = split,
        tickers    = tickers,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
    )


if __name__ == "__main__":
    main()
