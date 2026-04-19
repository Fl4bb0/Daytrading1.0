"""
scripts/run_train_meta.py — Train the minimal meta ranking model on validation predictions.

Usage
-----
  python scripts/run_train_meta.py
  python scripts/run_train_meta.py --config pipeline.toml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvant.utils.ensemble import ensemble_slug, normalize_model_names
from kvant.utils.pipeline_config import list_from_config, load_pipeline_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PREPARED_ROOT = _PROJECT_ROOT / "prepared"
_CHECKPOINTS_ROOT = _PROJECT_ROOT / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a meta regression model on base-model validation predictions."
    )
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()
    cfg, cfg_path = load_pipeline_config(args.config)

    prepared_root = Path(cfg["paths"].get("prepared_root", str(_PREPARED_ROOT)))
    checkpoints_root = Path(cfg["paths"].get("checkpoints_root", str(_CHECKPOINTS_ROOT)))
    predict_cfg = cfg["predict"]
    ensemble_cfg = cfg.get("ensemble", {})
    meta_cfg = cfg.get("meta", {})

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
    requested_tickers = list_from_config(predict_cfg.get("tickers")) or None
    meta_train_split = str(meta_cfg.get("train_split", "val"))
    meta_alpha = float(meta_cfg.get("alpha", 1.0))
    meta_shrinkage_k = float(meta_cfg.get("shrinkage_k", 10.0))

    model_names = normalize_model_names(ensemble_cfg.get("models"))
    use_ensemble = bool(model_names)
    if use_ensemble and "conv3d" in model_names:
        raise SystemExit(
            "conv3d cannot be used in ensemble mode in the current pipeline. "
            "Use non-conv3d models together, or run conv3d as a standalone model"
        )
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
        model_cls = type(member_models[0])

    print(f"Models           : {model_names}")
    print(f"Config           : {cfg_path}")
    print(f"Base checkpoint  : {model_path}")
    print(f"Meta train split : {meta_train_split}")

    from kvant.evaluation import build_prediction_frame
    from kvant.meta import META_FEATURE_COLUMNS, RidgeMetaModel, build_meta_training_frame

    pred_df = build_prediction_frame(
        exp_dir=exp_dir,
        model_path=model_path,
        model_cls=model_cls,
        split=meta_train_split,
        tickers=requested_tickers,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        model=model,
    )
    meta_train_df = build_meta_training_frame(
        pred_df,
        shrinkage_k=meta_shrinkage_k,
    )
    if meta_train_df.empty:
        raise SystemExit(
            "No valid directional rows were available to train the meta model. "
            "Check that the base model produces BUY/SHORT predictions with probabilities."
        )

    meta_model = RidgeMetaModel(alpha=meta_alpha)
    metrics = meta_model.fit(meta_train_df)

    meta_dir = checkpoints_root / exp_dir.name / active_model_name / "meta"
    meta_model.save(meta_dir)
    (meta_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "base_model_name": active_model_name,
                "base_member_models": model_names,
                "train_split": meta_train_split,
                "n_rows_total": int(len(pred_df)),
                "n_rows_trainable": int(len(meta_train_df)),
                "feature_columns": META_FEATURE_COLUMNS,
                "metrics": metrics,
            },
            indent=2,
        )
    )

    print(f"Meta checkpoint  : {meta_dir}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
