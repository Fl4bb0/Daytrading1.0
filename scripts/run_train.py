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


# Where prepare_experiment writes its output
_PREPARED_ROOT = Path(__file__).resolve().parents[1] / "prepared"
# Where trained checkpoints are saved
_CHECKPOINTS_ROOT = Path(__file__).resolve().parents[1] / "checkpoints"


def _load_split(exp_dir: Path, index: "np.ndarray", lookback_L: int) -> "tuple[np.ndarray, np.ndarray]":
    """
    Given an index array of shape (n, 2) with columns (tid, position),
    slice a rolling window of length lookback_L ending at each position
    and return (X, y) with X shaped (n, n_features, lookback_L).
    """
    import numpy as np
    from collections import defaultdict

    tickers_root = exp_dir / "tickers"
    ticker_dirs  = sorted(tickers_root.iterdir())
    tid_to_dir   = {i: d for i, d in enumerate(ticker_dirs)}

    by_tid: dict = defaultdict(list)
    for tid, pos in index:
        by_tid[int(tid)].append(int(pos))

    X_parts, y_parts = [], []
    for tid, positions in sorted(by_tid.items()):
        tdir     = tid_to_dir[tid]
        features = np.load(tdir / "features.npy", mmap_mode="r")  # (total_bars, n_features)
        labels   = np.load(tdir / "labels.npy",   mmap_mode="r")  # (total_bars,)

        windows = np.stack(
            [features[p - lookback_L + 1 : p + 1] for p in positions],
            axis=0,
        )  # (n, lookback_L, n_features)

        # Transpose to (n, n_features, lookback_L) — expected by all Conv/LSTM models
        windows = windows.transpose(0, 2, 1)

        X_parts.append(windows)
        y_parts.append(labels[np.array(positions)].astype(np.int64))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="Train a kvant model.")
    parser.add_argument(
        "--experiment-id", required=True,
        help="Prepared experiment ID (sub-directory name under prepared/), or 'last'.",
    )
    parser.add_argument("--model",    default="conv1d", help="Model key: conv1d, conv3d, resnls, tsb.")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=10, help="Early-stopping patience.")
    parser.add_argument("--device",   default="cpu", help="torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--prepared-root", default=str(_PREPARED_ROOT),
        help=f"Root directory for prepared experiments. Default: {_PREPARED_ROOT}",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Where to save the best checkpoint. Default: checkpoints/<experiment-id>/<model>/",
    )
    args = parser.parse_args()

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
        else _CHECKPOINTS_ROOT / exp_id / args.model
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {exp_dir}")
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
