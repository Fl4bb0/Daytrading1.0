from __future__ import annotations

from typing import Any, Sequence

from kvant.utils.pipeline_config import list_from_config


def normalize_model_names(value: Any) -> list[str]:
    names = list_from_config(value) or []
    return [str(name).strip() for name in names if str(name).strip()]


def ensemble_slug(model_names: Sequence[str]) -> str:
    cleaned = [str(name).strip() for name in model_names if str(name).strip()]
    if not cleaned:
        raise ValueError("ensemble_slug requires at least one model name")
    return "ensemble_" + "-".join(cleaned)

