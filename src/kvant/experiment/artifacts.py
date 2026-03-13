"""
experiment.artifacts — Low-level helpers for persisting per-ticker artifacts.

Extracted from prepare_experiment.py so prepare.py stays focused on
orchestration logic only.

save_ticker_artifacts(tdir, X, y, ts, meta, label_metadata=None)
save_label_metadata_jsonl(tdir, metadata)
json_default(x)   — shared fallback serialiser for json.dumps
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def json_default(x):
    """Fallback serialiser for json.dumps — handles numpy + pandas scalars."""
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.datetime64):
        return str(pd.Timestamp(x))
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return str(x)


def save_label_metadata_jsonl(tdir: Path, metadata: List[Optional[dict]]) -> None:
    """Write one JSON value per line, aligned positionally with features/labels."""
    path = tdir / "label_metadata.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, default=json_default))
            f.write("\n")


def save_ticker_artifacts(
    tdir: Path,
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    meta: dict,
    label_metadata: Optional[List[Optional[dict]]] = None,
) -> None:
    """Persist features, labels, timestamps, metadata, and optional label metadata."""
    tdir.mkdir(parents=True, exist_ok=True)
    np.save(tdir / "features.npy", X.astype(np.float32))
    np.save(tdir / "labels.npy", y.astype(np.int8))
    np.save(tdir / "timestamps.npy", ts.astype("datetime64[ns]"))
    (tdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if label_metadata is not None:
        if len(label_metadata) != len(y):
            raise RuntimeError(
                f"label_metadata length {len(label_metadata)} != labels length {len(y)}"
            )
        save_label_metadata_jsonl(tdir, label_metadata)
