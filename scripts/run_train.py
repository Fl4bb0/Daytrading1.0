"""
scripts/run_train.py — Entry-point for model training.

Loads a prepared experiment from disk, builds train/val/test numpy arrays
from the index files, instantiates the requested model and trainer, and runs
the full training loop.

Usage
-----
  python scripts/run_train.py --experiment-id <id> --model conv1d
  python scripts/run_train.py --experiment-id last --model resnls --epochs 100
"""
import argparse
from pathlib import Path

from kvant.utils.pipeline_config import load_pipeline_config


# Where prepare_experiment writes its output
_PREPARED_ROOT = Path(__file__).resolve().parents[1] / "prepared"
# Where trained checkpoints are saved
_CHECKPOINTS_ROOT = Path(__file__).resolve().parents[1] / "checkpoints"


def _load_split(exp_dir: Path, index: "np.ndarray", lookback_L: int) -> "tuple[np.ndarray, np.ndarray]":
    """
    Given an index array of shape (n, 2) with columns (tid, position),
    slice a rolling window of length lookback_L immediately before each target
    position p (i.e. bars [p-lookback_L, p))
    and return (X, y) with X shaped (n, n_features, lookback_L).
    """
    from kvant.utils.split_loader import load_split_from_index

    loaded = load_split_from_index(
        exp_dir=exp_dir,
        index=index,
        lookback_L=lookback_L,
        include_timestamps=False,
        include_metadata=False,
    )
    return loaded.X, loaded.y


def main():
    import numpy as np

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Path to pipeline TOML config.")
    pre_args, remaining = pre_parser.parse_known_args()
    pipeline_cfg, cfg_path = load_pipeline_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train a kvant model.", parents=[pre_parser])
    parser.add_argument(
        "--experiment-id", default=pipeline_cfg["train"].get("experiment_id", "last"),
        help="Prepared experiment ID (sub-directory name under prepared/), or 'last'.",
    )
    parser.add_argument("--model", default=pipeline_cfg["train"].get("model", "conv1d"), help="Model key: conv1d, conv3d, resnls, tsb.")
    parser.add_argument("--epochs", type=int, default=int(pipeline_cfg["train"].get("epochs", 50)))
    parser.add_argument("--batch-size", type=int, default=int(pipeline_cfg["train"].get("batch_size", 256)))
    parser.add_argument("--lr", type=float, default=float(pipeline_cfg["train"].get("learning_rate", 1e-3)))
    parser.add_argument("--patience", type=int, default=int(pipeline_cfg["train"].get("patience", 10)), help="Early-stopping patience.")
    parser.add_argument("--device", default=pipeline_cfg["train"].get("device", "cpu"), help="torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--prepared-root", default=str(Path(pipeline_cfg["paths"].get("prepared_root", str(_PREPARED_ROOT)))),
        help=f"Root directory for prepared experiments. Default: {_PREPARED_ROOT}",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Where to save the best checkpoint. Default: checkpoints/<experiment-id>/<model>/",
    )
    args = parser.parse_args(remaining)

    prepared_root = Path(args.prepared_root)

    # Resolve experiment directory
    exp_id = args.experiment_id
    if exp_id == "last":
        last_file = prepared_root / "last_experiment.txt"
        if not last_file.exists():
            raise SystemExit(f"No last_experiment.txt found in {prepared_root}")
        exp_id = last_file.read_text().strip()

    exp_dir = prepared_root / exp_id
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    # Resolve checkpoint directory
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(pipeline_cfg["paths"].get("checkpoints_root", str(_CHECKPOINTS_ROOT))) / exp_id / args.model
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {exp_dir}")
    print(f"Config     : {cfg_path}")
    print(f"Model      : {args.model}")
    print(f"Checkpoint : {checkpoint_dir}")
    print(f"Device     : {args.device}\n")

    # ------------------------------------------------------------------
    # Load index arrays and build X / y splits
    # ------------------------------------------------------------------
    import json
    cfg_data   = json.loads((exp_dir / "config.json").read_text())
    lookback_L = int(cfg_data["lookback_L"])

    index_train = np.load(exp_dir / "index_train.npy")
    index_val   = np.load(exp_dir / "index_val.npy")
    index_test  = np.load(exp_dir / "index_test.npy")

    print("Loading splits from disk…")
    X_train, y_train = _load_split(exp_dir, index_train, lookback_L)
    X_val,   y_val   = _load_split(exp_dir, index_val,   lookback_L)
    X_test,  y_test  = _load_split(exp_dir, index_test,  lookback_L)

    # Features are stored as (n, n_features, lookback_L) — keep as-is for Conv models.
    # n_features is dim 1.
    n_features = X_train.shape[1]
    n_classes  = int(y_train.max()) + 1

    print(f"  train : {len(X_train):,} samples  |  shape {X_train.shape}")
    print(f"  val   : {len(X_val):,} samples")
    print(f"  test  : {len(X_test):,} samples")
    print(f"  n_features={n_features}  n_classes={n_classes}\n")

    # ------------------------------------------------------------------
    # Instantiate model
    # ------------------------------------------------------------------
    from kvant.models import MODEL_REGISTRY
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[args.model](
        n_features=n_features,
        n_classes=n_classes,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    from kvant.training.pytorch_trainer import PytorchTrainer
    from kvant.training.trainer import TrainConfig

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = PytorchTrainer(model, cfg)

    print("Training…")
    history = trainer.fit(X_train, y_train, X_val, y_val)
    print(f"\nBest val accuracy : {history['best_val_accuracy']:.4f}  (epoch {history['best_epoch']})")

    # ------------------------------------------------------------------
    # Evaluate on test split
    # ------------------------------------------------------------------
    print("\nEvaluating on test split…")
    test_metrics = trainer.evaluate(X_test, y_test)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save final checkpoint explicitly (best weights already restored by trainer)
    model.save(checkpoint_dir)
    print(f"\nCheckpoint saved → {checkpoint_dir}")


if __name__ == "__main__":
    main()
