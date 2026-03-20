"""
scripts/run_train.py — Train configured model on prepared data.

Usage
-----
  python scripts/run_train.py
  python scripts/run_train.py --config pipeline.toml
"""

import argparse
import json
from pathlib import Path

from kvant.utils.pipeline_config import load_pipeline_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PREPARED_ROOT = _PROJECT_ROOT / "prepared"
_CHECKPOINTS_ROOT = _PROJECT_ROOT / "checkpoints"


def _load_split(exp_dir: Path, index, lookback_L: int):
    from kvant.utils.split_loader import load_split_from_index

    loaded = load_split_from_index(
        exp_dir=exp_dir,
        index=index,
        lookback_L=lookback_L,
        include_timestamps=False,
        include_metadata=False,
    )
    return loaded.X, loaded.y


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser(description="Train a kvant model.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()

    pipeline_cfg, cfg_path = load_pipeline_config(args.config)
    train_cfg = pipeline_cfg["train"]
    paths_cfg = pipeline_cfg["paths"]

    prepared_root = Path(paths_cfg.get("prepared_root", str(_PREPARED_ROOT)))
    exp_id = str(train_cfg.get("experiment_id", "last"))
    if exp_id == "last":
        last_file = prepared_root / "last_experiment.txt"
        if not last_file.exists():
            raise SystemExit(f"No last_experiment.txt found in {prepared_root}")
        exp_id = last_file.read_text().strip()

    exp_dir = prepared_root / exp_id
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    model_name = str(train_cfg.get("model", "conv1d"))
    checkpoint_dir = Path(paths_cfg.get("checkpoints_root", str(_CHECKPOINTS_ROOT))) / exp_id / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {exp_dir}")
    print(f"Config     : {cfg_path}")
    print(f"Model      : {model_name}")
    print(f"Checkpoint : {checkpoint_dir}")
    print(f"Device     : {train_cfg.get('device', 'cpu')}\n")

    cfg_data = json.loads((exp_dir / "config.json").read_text())
    lookback_L = int(cfg_data["lookback_L"])

    index_train = np.load(exp_dir / "index_train.npy")
    index_val = np.load(exp_dir / "index_val.npy")
    index_test = np.load(exp_dir / "index_test.npy")

    X_train, y_train = _load_split(exp_dir, index_train, lookback_L)
    X_val, y_val = _load_split(exp_dir, index_val, lookback_L)
    X_test, y_test = _load_split(exp_dir, index_test, lookback_L)

    n_features = X_train.shape[1]
    n_classes = int(y_train.max()) + 1

    from kvant.models import MODEL_REGISTRY

    if model_name not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[model_name](
        n_features=n_features,
        n_classes=n_classes,
        device=str(train_cfg.get("device", "cpu")),
    )

    from kvant.training.pytorch_trainer import PytorchTrainer
    from kvant.training.trainer import TrainConfig

    cfg = TrainConfig(
        epochs=int(train_cfg.get("epochs", 50)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        early_stopping_patience=int(train_cfg.get("patience", 10)),
        checkpoint_dir=checkpoint_dir,
    )
    trainer = PytorchTrainer(model, cfg)

    history = trainer.fit(X_train, y_train, X_val, y_val)
    print(f"Best val accuracy : {history['best_val_accuracy']:.4f} (epoch {history['best_epoch']})")

    test_metrics = trainer.evaluate(X_test, y_test)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    model.save(checkpoint_dir)
    print(f"Checkpoint saved -> {checkpoint_dir}")


if __name__ == "__main__":
    main()
