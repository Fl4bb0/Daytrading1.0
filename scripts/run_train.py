"""scripts/run_train.py — Train configured model on prepared data."""

import argparse

from kvant.pipeline_runtime import resolve_experiment_dir, train_experiment
from kvant.utils.pipeline_config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a kvant model.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()

    pipeline_cfg, cfg_path = load_pipeline_config(args.config)
    exp_id = str(pipeline_cfg["train"].get("experiment_id", "last"))
    exp_dir = resolve_experiment_dir(exp_id, pipeline_cfg)

    print(f"Experiment : {exp_dir}")
    print(f"Config     : {cfg_path}")
    for artifact in train_experiment(exp_dir, pipeline_cfg):
        print(f"Model      : {artifact.model_name}")
        print(f"Checkpoint : {artifact.checkpoint_dir}")
        print(
            f"Best val accuracy : {artifact.best_val_accuracy:.4f} "
            f"(epoch {artifact.best_epoch})"
        )
        for key, value in artifact.test_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"Checkpoint saved -> {artifact.checkpoint_dir}\n")


if __name__ == "__main__":
    main()
