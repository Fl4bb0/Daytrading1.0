from __future__ import annotations

import math
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
    "trading": {
        "brokerage_fee": 0.0008,
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
    "ensemble": {
        "models": [],
    },
    "train": {
        "experiment_id": "last",
        "model": "conv1d",
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "patience": 10,
        "device": "auto",
    },
    "predict": {
        "experiment_id": "last",
        "model": "conv1d",
        "split": "test",
        "tickers": [],
        "required_buy_probability": 0.6,
        "required_sell_probability": 0.6,
        "execution_priority": "model_confidence",
        "top_k_per_timestamp": None,
        "ticker_cooldown_minutes": 0,
    },
    "meta": {
        "enabled": False,
        "train_split": "val",
        "alpha": 1.0,
        "shrinkage_k": 10.0,
        "min_score_buy": None,
        "min_score_short": None,
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
    "benchmark": {
        "benchmark_id": "benchmark_test",
        "single_model": "resnls",
        "random_seeds": 50,
        "random_seed_start": 0,
        "random_trade_probability": None,
        "random_fallback_trade_probability": 0.03,
        "shallow_epochs": 20,
        "shallow_patience": 5,
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
    brokerage_fee = float(cfg.get("trading", {}).get("brokerage_fee", 0.0008))
    if not math.isfinite(brokerage_fee) or brokerage_fee < 0.0:
        raise SystemExit(f"trading.brokerage_fee must be a finite number >= 0 in {path}")

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

    ensemble_models = list_from_config(cfg.get("ensemble", {}).get("models")) or []

    from kvant.models import MODEL_REGISTRY

    single_train_model = str(cfg["train"].get("model", "conv1d"))
    if single_train_model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown train.model '{single_train_model}' in {path}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    single_predict_model = str(cfg["predict"].get("model", "conv1d"))
    if single_predict_model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown predict.model '{single_predict_model}' in {path}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    if ensemble_models:
        if len(set(ensemble_models)) != len(ensemble_models):
            raise SystemExit(f"ensemble.models must not contain duplicates in {path}")

        unknown = [name for name in ensemble_models if name not in MODEL_REGISTRY]
        if unknown:
            raise SystemExit(
                f"Unknown model(s) in ensemble.models {unknown} in {path}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        unsupported_in_ensemble = [name for name in ensemble_models if name == "conv3d"]
        if unsupported_in_ensemble:
            raise SystemExit(
                f"ensemble.models contains unsupported model(s) {unsupported_in_ensemble} in {path}. "
                "conv3d requires multi-timeframe tensors and cannot be mixed in the current ensemble pipeline"
            )

    if str(cfg["predict"]["split"]) not in {"train", "val", "test"}:
        raise SystemExit(f"predict.split must be one of train|val|test in {path}")

    for key in ("required_buy_probability", "required_sell_probability"):
        value = float(cfg["predict"].get(key, 0.0))
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"predict.{key} must be between 0 and 1 in {path}")

    execution_priority = str(cfg["predict"].get("execution_priority", "model_confidence"))
    if execution_priority not in {"first_seen", "model_confidence", "meta_score"}:
        raise SystemExit(
            "predict.execution_priority must be one of "
            "first_seen|model_confidence|meta_score in "
            f"{path}"
        )

    top_k_per_timestamp = cfg["predict"].get("top_k_per_timestamp")
    if top_k_per_timestamp not in (None, "", 0):
        if int(top_k_per_timestamp) <= 0:
            raise SystemExit(f"predict.top_k_per_timestamp must be > 0 in {path}")

    ticker_cooldown_minutes = int(cfg["predict"].get("ticker_cooldown_minutes", 0))
    if ticker_cooldown_minutes < 0:
        raise SystemExit(f"predict.ticker_cooldown_minutes must be >= 0 in {path}")

    meta_train_split = str(cfg.get("meta", {}).get("train_split", "val"))
    if meta_train_split not in {"train", "val"}:
        raise SystemExit(f"meta.train_split must be one of train|val in {path}")

    meta_alpha = float(cfg.get("meta", {}).get("alpha", 1.0))
    if meta_alpha <= 0.0:
        raise SystemExit(f"meta.alpha must be > 0 in {path}")

    meta_shrinkage_k = float(cfg.get("meta", {}).get("shrinkage_k", 10.0))
    if meta_shrinkage_k < 0.0:
        raise SystemExit(f"meta.shrinkage_k must be >= 0 in {path}")

    for key in ("min_score_buy", "min_score_short"):
        value = cfg.get("meta", {}).get(key)
        if value in (None, ""):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise SystemExit(f"meta.{key} must be a finite number in {path}")

    if execution_priority == "meta_score" and not bool(cfg.get("meta", {}).get("enabled", False)):
        raise SystemExit(
            "predict.execution_priority=meta_score requires meta.enabled=true "
            f"in {path}"
        )

    benchmark_cfg = cfg.get("benchmark", {})
    benchmark_single = str(benchmark_cfg.get("single_model", cfg["predict"].get("model", "conv1d")))
    if benchmark_single not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown benchmark.single_model '{benchmark_single}' in {path}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    if int(benchmark_cfg.get("random_seeds", 50)) <= 0:
        raise SystemExit(f"benchmark.random_seeds must be > 0 in {path}")

    random_trade_probability = benchmark_cfg.get("random_trade_probability")
    if random_trade_probability not in (None, ""):
        value = float(random_trade_probability)
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"benchmark.random_trade_probability must be between 0 and 1 in {path}")

    random_fallback_trade_probability = float(benchmark_cfg.get("random_fallback_trade_probability", 0.03))
    if random_fallback_trade_probability < 0.0 or random_fallback_trade_probability > 1.0:
        raise SystemExit(f"benchmark.random_fallback_trade_probability must be between 0 and 1 in {path}")

    if int(benchmark_cfg.get("shallow_epochs", 20)) <= 0:
        raise SystemExit(f"benchmark.shallow_epochs must be > 0 in {path}")

    if int(benchmark_cfg.get("shallow_patience", 5)) <= 0:
        raise SystemExit(f"benchmark.shallow_patience must be > 0 in {path}")


def list_from_config(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise SystemExit("Expected a list or string in pipeline config")
