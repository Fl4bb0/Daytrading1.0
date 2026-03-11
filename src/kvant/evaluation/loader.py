"""
evaluation.loader — Load prepared experiment artifacts from disk for a given split.

load_split(exp_dir, split, tickers=None)
    → (X, y, timestamps, tids, metas, ticker_map)

Directory layout expected (produced by experiment.artifacts):
    <exp_dir>/
        <split>/
            <TICKER>/
                features.npy
                labels.npy
                timestamps.npy
                meta.json
                label_metadata.jsonl   (optional)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_split(
    exp_dir: Path,
    split: str,
    tickers: Optional[List[str]] = None,
) -> Tuple[
    np.ndarray,          # X       : float32, shape (n, n_features, lookback_L)
    np.ndarray,          # y       : int8,    shape (n,)
    np.ndarray,          # timestamps: datetime64[ns], shape (n,)
    np.ndarray,          # tids    : int64,   shape (n,)  — monotonic ticker id
    List[Optional[dict]],# metas   : per-sample label metadata (None if missing)
    Dict[int, str],      # ticker_map: tid → ticker symbol
]:
    """
    Load all artifacts for *split* from *exp_dir* and concatenate them.

    Parameters
    ----------
    exp_dir  : Path to a prepared experiment directory.
    split    : One of ``"train"``, ``"val"``, ``"test"``.
    tickers  : Optional allowlist; if given, only these tickers are loaded.

    Returns
    -------
    X, y, timestamps, tids, metas, ticker_map
    """
    split_dir = Path(exp_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Expected layout: <exp_dir>/<split>/<TICKER>/features.npy …"
        )

    ticker_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not ticker_dirs:
        raise FileNotFoundError(f"No ticker sub-directories found in {split_dir}")

    if tickers is not None:
        tickers_set = set(tickers)
        ticker_dirs = [p for p in ticker_dirs if p.name in tickers_set]
        if not ticker_dirs:
            raise ValueError(
                f"None of the requested tickers {tickers} found in {split_dir}"
            )

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    tss: List[np.ndarray] = []
    tid_arrs: List[np.ndarray] = []
    metas_all: List[Optional[dict]] = []
    ticker_map: Dict[int, str] = {}

    reference_shape: Optional[Tuple[int, ...]] = None  # (n_features, lookback_L)

    for tid, tdir in enumerate(ticker_dirs):
        features_path = tdir / "features.npy"
        labels_path   = tdir / "labels.npy"
        ts_path       = tdir / "timestamps.npy"

        if not features_path.exists() or not labels_path.exists():
            # Skip tickers with missing core artifacts (e.g. too few samples)
            continue

        X_t  = np.load(features_path)          # (n, n_features, lookback_L) or (n, n_features)
        y_t  = np.load(labels_path).astype(np.int64)
        ts_t = np.load(ts_path) if ts_path.exists() else np.full(len(y_t), np.datetime64("NaT"))

        # Validate / record feature shape
        if reference_shape is None:
            reference_shape = X_t.shape[1:]
        elif X_t.shape[1:] != reference_shape:
            raise ValueError(
                f"Feature shape mismatch for ticker {tdir.name}: "
                f"expected {reference_shape}, got {X_t.shape[1:]}"
            )

        n = len(y_t)
        ticker_map[tid] = tdir.name

        # Load per-sample label metadata
        meta_path = tdir / "label_metadata.jsonl"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                ticker_metas: List[Optional[dict]] = [
                    json.loads(line) if line.strip() not in ("", "null") else None
                    for line in f
                ]
            # Guard against length mismatches
            if len(ticker_metas) != n:
                ticker_metas = [None] * n
        else:
            ticker_metas = [None] * n

        Xs.append(X_t)
        ys.append(y_t)
        tss.append(ts_t)
        tid_arrs.append(np.full(n, tid, dtype=np.int64))
        metas_all.extend(ticker_metas)

    if not Xs:
        raise RuntimeError(
            f"No valid ticker artifacts could be loaded from {split_dir}"
        )

    X   = np.concatenate(Xs,       axis=0)
    y   = np.concatenate(ys,       axis=0)
    ts  = np.concatenate(tss,      axis=0)
    tids = np.concatenate(tid_arrs, axis=0)

    return X, y, ts, tids, metas_all, ticker_map
