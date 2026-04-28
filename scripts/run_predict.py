"""scripts/run_predict.py — Entry-point for inference + evaluation statistics."""

import argparse

from kvant.pipeline_runtime import load_runtime_model, predict_experiment, resolve_experiment_dir
from kvant.utils.pipeline_config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained kvant model and save evaluation CSVs."
    )
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()
    cfg, cfg_path = load_pipeline_config(args.config)

    exp_id = str(cfg["predict"].get("experiment_id", "last"))
    exp_dir = resolve_experiment_dir(exp_id, cfg)
    runtime = load_runtime_model(exp_dir, cfg)
    print(f"Models: {runtime.model_names}")
    print(f"Config: {cfg_path}")
    print(f"Checkpoint: {runtime.model_path}")
    predict_experiment(exp_dir, cfg)


if __name__ == "__main__":
    main()
