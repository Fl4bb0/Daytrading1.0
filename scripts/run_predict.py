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

from kvant.utils.ensemble import ensemble_slug, normalize_model_names
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
    ensemble_cfg = cfg.get("ensemble", {})

    exp_id = str(predict_cfg.get("experiment_id", "last"))
    if exp_id == "last":
        last_file = prepared_root / "last_experiment.txt"
        if not last_file.exists():
            raise SystemExit(f"No last_experiment.txt found in {prepared_root}.")
        exp_id = last_file.read_text().strip()
        print(f"Auto-detected experiment: {exp_id}")
    exp_dir = prepared_root / exp_id

    required_buy_probability = float(predict_cfg.get("required_buy_probability", 0.0))
    required_sell_probability = float(predict_cfg.get("required_sell_probability", 0.0))
    execution_priority = str(predict_cfg.get("execution_priority", "model_confidence"))
    top_k_raw = predict_cfg.get("top_k_per_timestamp")
    top_k_per_timestamp = None if top_k_raw in (None, "", 0) else int(top_k_raw)
    ticker_cooldown_minutes = int(predict_cfg.get("ticker_cooldown_minutes", 0))

    model_names = normalize_model_names(ensemble_cfg.get("models"))
    use_ensemble = bool(model_names)
    if use_ensemble:
        active_model_name = ensemble_slug(model_names)
    else:
        active_model_name = str(predict_cfg.get("model", "conv1d"))
        model_names = [active_model_name]

    from kvant.models import MODEL_REGISTRY
    from kvant.models.ensemble import AveragingEnsembleModel

    member_models = []
    member_paths = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            raise SystemExit(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        checkpoint = checkpoints_root / exp_dir.name / model_name
        if not (checkpoint / "weights.pt").exists():
            raise SystemExit(
                f"No checkpoint found at {checkpoint}/weights.pt. "
                f"Train first with: uv run --env-file .env.run scripts/run_train.py"
            )
        member_models.append(MODEL_REGISTRY[model_name].load(checkpoint))
        member_paths.append(checkpoint)

    if len(member_models) == 1 and not use_ensemble:
        model = member_models[0]
        model_path = member_paths[0]
        model_cls = type(model)
    else:
        model = AveragingEnsembleModel(
            member_models,
            member_names=model_names,
            member_paths=member_paths,
            name=active_model_name,
        )
        model_path = checkpoints_root / exp_dir.name / active_model_name
        model.save(model_path)
        model_cls = type(member_models[0])

    print(f"Models: {model_names}")
    print(f"Config: {cfg_path}")
    print(f"Auto-detected checkpoint: {model_path}")

    out_dir = exp_dir / "eval" / f"{model.name}_{str(predict_cfg.get('split', 'test'))}"

    from kvant.evaluation import evaluate_experiment
    evaluate_experiment(
        exp_dir    = exp_dir,
        model_path = model_path,
        model_cls  = model_cls,
        out_dir    = out_dir,
        split      = str(predict_cfg.get("split", "test")),
        tickers    = list_from_config(predict_cfg.get("tickers")) or None,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        execution_priority=execution_priority,
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
        model=model,
    )


if __name__ == "__main__":
    main()
