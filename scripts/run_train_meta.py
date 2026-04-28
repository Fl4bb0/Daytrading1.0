"""scripts/run_train_meta.py — Train the minimal meta ranking model on validation predictions."""

import argparse

from kvant.pipeline_runtime import resolve_experiment_dir, train_meta_experiment
from kvant.utils.pipeline_config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a meta regression model on base-model validation predictions."
    )
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()
    cfg, cfg_path = load_pipeline_config(args.config)

    meta_cfg = cfg.get("meta", {})
    exp_id = str(cfg["predict"].get("experiment_id", "last"))
    exp_dir = resolve_experiment_dir(exp_id, cfg)
    meta_train_split = str(meta_cfg.get("train_split", "val"))

    print(f"Config           : {cfg_path}")
    print(f"Experiment       : {exp_dir}")
    print(f"Meta train split : {meta_train_split}")
    meta_dir = train_meta_experiment(exp_dir, cfg)
    print(f"Meta checkpoint  : {meta_dir}")


if __name__ == "__main__":
    main()
