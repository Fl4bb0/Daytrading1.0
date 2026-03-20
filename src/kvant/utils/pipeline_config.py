from __future__ import annotations

import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "pipeline.toml"

DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "paths": {
        "store": "data/1m",
        "prepared_root": "prepared",
        "checkpoints_root": "checkpoints",
        "sync_state": "sync_state.json",
    },
    "data": {
        "interval": "1m",
        "symbols": [],
        "prepost": False,
        "daily_budget": 25,
    },
    "prepare": {
        "val_frac": 0.15,
        "test_frac": 0.15,
        "lookback": 20,
        "width_minutes": 20,
        "height_pct": 0.5,
        "target_bars_per_day": 195,
        "volatility_scaled_barrier": True,
        "vol_scale_min": 0.5,
        "vol_scale_max": 2.0,
    },
    "train": {
        "experiment_id": "last",
        "model": "conv1d",
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "patience": 10,
        "device": "cpu",
    },
    "predict": {
        "experiment_id": "last",
        "model": "conv1d",
        "split": "test",
        "tickers": [],
        "required_buy_probability": 0.6,
        "required_sell_probability": 0.6,
    },
    "daily": {
        "roll_forward": True,
        "recent_days": 7,
    },
    "plot": {
        "experiment_id": "last",
        "model": "conv1d",
        "split": "test",
        "show": False,
        "out_dir": None,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_pipeline_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv("KVANT_PIPELINE_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return _DEFAULT_CONFIG_PATH


def load_pipeline_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    path = resolve_pipeline_config_path(config_path)
    if not path.exists():
        raise SystemExit(f"Pipeline config not found: {path}")

    data = tomllib.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid pipeline config format: {path}")

    cfg = _deep_merge(DEFAULT_PIPELINE_CONFIG, data)
    _validate_config(cfg, path)
    return cfg, path


def _validate_config(cfg: dict[str, Any], path: Path) -> None:
    for key in ("val_frac", "test_frac"):
        value = float(cfg["prepare"][key])
        if value <= 0 or value >= 1:
            raise SystemExit(f"{key} must be between 0 and 1 in {path}")

    if float(cfg["prepare"]["val_frac"]) + float(cfg["prepare"]["test_frac"]) >= 1:
        raise SystemExit(f"val_frac + test_frac must be < 1 in {path}")

    if int(cfg["prepare"]["lookback"]) <= 0:
        raise SystemExit(f"lookback must be > 0 in {path}")

    if int(cfg["prepare"]["width_minutes"]) <= 0:
        raise SystemExit(f"width_minutes must be > 0 in {path}")

    if int(cfg["prepare"]["target_bars_per_day"]) <= 0:
        raise SystemExit(f"target_bars_per_day must be > 0 in {path}")

    if str(cfg["predict"]["split"]) not in {"train", "val", "test"}:
        raise SystemExit(f"predict.split must be one of train|val|test in {path}")

    for key in ("required_buy_probability", "required_sell_probability"):
        value = float(cfg["predict"].get(key, 0.0))
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"predict.{key} must be between 0 and 1 in {path}")


def list_from_config(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise SystemExit("Expected a list or string in pipeline config")

