"""
evaluation.loader — Load prepared experiment artifacts from disk for a given split.

load_split(exp_dir, split, tickers=None)
    → (X, y, timestamps, tids, metas, ticker_map)

Actual on-disk layout (produced by prepare_experiment):
    <exp_dir>/
        config.json
        index_train.npy      ← (n, 2) arrays of (tid, position)
        index_val.npy
        index_test.npy
        tickers_<split>.json
        tickers/
            <TICKER>/
                features.npy         (total_bars, n_features)
                labels.npy           (total_bars,)
                timestamps.npy       (total_bars,)
                label_metadata.jsonl (total_bars lines)
                meta.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_split(
    exp_dir: Path,
    split: str,
    tickers: Optional[List[str]] = None,
) -> Tuple[
    np.ndarray,           # X          : float32, shape (n, n_features, lookback_L)
    np.ndarray,           # y          : int64,   shape (n,)
    np.ndarray,           # timestamps : datetime64[ns], shape (n,)
    np.ndarray,           # tids       : int64,   shape (n,)
    List[Optional[dict]], # metas      : per-sample label metadata
    Dict[int, str],       # ticker_map : tid → ticker symbol
]:
    """
    Load all artifacts for *split* from *exp_dir*, apply the lookback window,
    and return concatenated arrays ready for model inference.
    """
    exp_dir = Path(exp_dir)

    # ------------------------------------------------------------------
    # Resolve which tickers belong to this split
    # ------------------------------------------------------------------
    tickers_file = exp_dir / f"tickers_{split}.json"
    if not tickers_file.exists():
        raise FileNotFoundError(f"Tickers list not found: {tickers_file}")
    split_tickers: List[str] = json.loads(tickers_file.read_text())

    if tickers is not None:
        missing = [t for t in tickers if t not in split_tickers]
        if missing:
            raise ValueError(f"Requested tickers not in {split} split: {missing}")
        split_tickers = [t for t in split_tickers if t in set(tickers)]

    if not split_tickers:
        raise RuntimeError(f"No tickers available for split '{split}'")

    # ------------------------------------------------------------------
    # Build tid → ticker mapping (consistent with prepare_experiment order)
    # ------------------------------------------------------------------
    all_tickers: List[str] = json.loads((exp_dir / "tickers_all.json").read_text())
    tid_to_ticker: Dict[int, str] = {i: t for i, t in enumerate(all_tickers)}
    ticker_to_tid: Dict[str, int] = {t: i for i, t in enumerate(all_tickers)}

    # ------------------------------------------------------------------
    # Load lookback_L from config
    # ------------------------------------------------------------------
    cfg       = json.loads((exp_dir / "config.json").read_text())
    lookback_L = int(cfg["lookback_L"])

    # ------------------------------------------------------------------
    # Load index array for this split: shape (n, 2) → (tid, position)
    # ------------------------------------------------------------------
    index_path = exp_dir / f"index_{split}.npy"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    index = np.load(index_path)  # (n, 2)

    if len(index) == 0:
        empty = np.asarray([], dtype=np.int64)
        return (
            np.empty((0, 1, lookback_L), dtype=np.float32),
            empty, empty, empty, [], {},
        )

    # Filter index to requested tickers
    allowed_tids = {ticker_to_tid[t] for t in split_tickers}
    mask  = np.isin(index[:, 0], list(allowed_tids))
    index = index[mask]

    # ------------------------------------------------------------------
    # Group positions by tid, load features/labels/timestamps/metas
    # ------------------------------------------------------------------
    tickers_root = exp_dir / "tickers"
    by_tid: Dict[int, List[int]] = defaultdict(list)
    for tid, pos in index:
        by_tid[int(tid)].append(int(pos))

    X_parts:    List[np.ndarray] = []
    y_parts:    List[np.ndarray] = []
    ts_parts:   List[np.ndarray] = []
    tid_parts:  List[np.ndarray] = []
    metas_all:  List[Optional[dict]] = []
    ticker_map: Dict[int, str] = {}

    for tid in sorted(by_tid):
        ticker   = tid_to_ticker[tid]
        tdir     = tickers_root / ticker
        positions = np.array(by_tid[tid])

        features   = np.load(tdir / "features.npy",   mmap_mode="r")  # (total, n_features)
        labels     = np.load(tdir / "labels.npy",     mmap_mode="r")  # (total,)
        ts_arr     = np.load(tdir / "timestamps.npy", mmap_mode="r")  # (total,)

        # Build rolling windows: (n, lookback_L, n_features) → transpose → (n, n_features, lookback_L)
        windows = np.stack(
            [features[p - lookback_L + 1: p + 1] for p in positions],
            axis=0,
        ).transpose(0, 2, 1).astype(np.float32)

        # Load label metadata
        meta_path = tdir / "label_metadata.jsonl"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                all_metas = [
                    json.loads(line) if line.strip() not in ("", "null") else None
                    for line in f
                ]
            ticker_metas = [
                all_metas[p] if p < len(all_metas) else None
                for p in positions
            ]
        else:
            ticker_metas = [None] * len(positions)

        ticker_map[tid] = ticker
        X_parts.append(windows)
        y_parts.append(labels[positions].astype(np.int64))
        ts_parts.append(ts_arr[positions])
        tid_parts.append(np.full(len(positions), tid, dtype=np.int64))
        metas_all.extend(ticker_metas)

    X          = np.concatenate(X_parts,   axis=0)
    y          = np.concatenate(y_parts,   axis=0)
    timestamps = np.concatenate(ts_parts,  axis=0)
    tids       = np.concatenate(tid_parts, axis=0)

    return X, y, timestamps, tids, metas_all, ticker_map
