"""
scripts/run_train.py — Entry-point for model training.

Thin wrapper: parses CLI args, instantiates the requested KvantModel
and Trainer, then calls trainer.fit().  No logic lives here.

Usage
-----
  python scripts/run_train.py --experiment-id <id> --model conv1d
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train a kvant model.")
    parser.add_argument("--experiment-id", required=True, help="ID of a prepared experiment.")
    parser.add_argument("--model", default="conv1d", help="Model key (conv1d, transformer, ...).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # TODO: resolve model key → KvantModel subclass, instantiate Trainer, call fit.
    raise NotImplementedError("run_train.py: wire up model registry and Trainer.")


if __name__ == "__main__":
    main()
