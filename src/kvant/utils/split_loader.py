"""
utils.split_loader — Shared split loading utilities for training and evaluation.

Provides a single implementation for turning an index array of (tid, position)
into causal model windows and aligned labels, with optional timestamps/metadata.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LoadedSplit:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    tids: np.ndarray
    metas: List[Optional[dict]]
    ticker_map: Dict[int, str]


def _load_ticker_maps(exp_dir: Path) -> tuple[Dict[int, str], Dict[str, int]]:
    all_tickers: List[str] = json.loads((exp_dir / "tickers_all.json").read_text())
    tid_to_ticker = {i: t for i, t in enumerate(all_tickers)}
    ticker_to_tid = {t: i for i, t in enumerate(all_tickers)}
    return tid_to_ticker, ticker_to_tid


def load_split_from_index(
    exp_dir: Path,
    index: np.ndarray,
    lookback_L: int,
    *,
    allowed_tickers: Optional[List[str]] = None,
    include_timestamps: bool = False,
    include_metadata: bool = False,
) -> LoadedSplit:
    """Load causal windows and aligned targets from a prepared split index."""
    exp_dir = Path(exp_dir)
    tid_to_ticker, ticker_to_tid = _load_ticker_maps(exp_dir)

    if allowed_tickers is not None:
        unknown = [t for t in allowed_tickers if t not in ticker_to_tid]
        if unknown:
            raise ValueError(f"Unknown ticker(s) requested: {unknown}")
        allowed_tids = {ticker_to_tid[t] for t in allowed_tickers}
        mask = np.isin(index[:, 0], list(allowed_tids))
        index = index[mask]

    if len(index) == 0:
        empty = np.asarray([], dtype=np.int64)
        return LoadedSplit(
            X=np.empty((0, 1, lookback_L), dtype=np.float32),
            y=empty,
            timestamps=empty,
            tids=empty,
            metas=[],
            ticker_map={},
        )

    by_tid: Dict[int, List[int]] = defaultdict(list)
    for tid, pos in index:
        by_tid[int(tid)].append(int(pos))

    tickers_root = exp_dir / "tickers"
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    ts_parts: List[np.ndarray] = []
    tid_parts: List[np.ndarray] = []
    metas_all: List[Optional[dict]] = []
    ticker_map: Dict[int, str] = {}

    for tid in sorted(by_tid):
        if tid not in tid_to_ticker:
            raise ValueError(
                f"Found unknown tid={tid} in index for {exp_dir.name}. "
                f"Known tids: 0..{len(tid_to_ticker) - 1}"
            )

        ticker = tid_to_ticker[tid]
        tdir = tickers_root / ticker
        if not tdir.exists():
            raise FileNotFoundError(f"Ticker directory missing for tid={tid}: {tdir}")

        positions = np.asarray(by_tid[tid], dtype=np.int64)
        features = np.load(tdir / "features.npy", mmap_mode="r")
        labels = np.load(tdir / "labels.npy", mmap_mode="r")

        bad_mask = (positions < lookback_L) | (positions >= len(features))
        if np.any(bad_mask):
            bad = positions[bad_mask]
            raise ValueError(
                "Invalid target positions for causal windows: "
                f"ticker={ticker} tid={tid} lookback_L={lookback_L} total_bars={len(features)} "
                f"bad_count={len(bad)} min_bad={int(bad.min())} max_bad={int(bad.max())}"
            )

        windows = np.stack(
            [features[p - lookback_L : p] for p in positions],
            axis=0,
        ).transpose(0, 2, 1).astype(np.float32)

        X_parts.append(windows)
        y_parts.append(labels[positions].astype(np.int64))
        tid_parts.append(np.full(len(positions), tid, dtype=np.int64))

        if include_timestamps:
            ts_arr = np.load(tdir / "timestamps.npy", mmap_mode="r")
            ts_parts.append(ts_arr[positions])

        if include_metadata:
            meta_path = tdir / "label_metadata.jsonl"
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    all_metas = [
                        json.loads(line) if line.strip() not in ("", "null") else None
                        for line in f
                    ]
                metas_all.extend(
                    all_metas[int(p)] if int(p) < len(all_metas) else None
                    for p in positions
                )
            else:
                metas_all.extend([None] * len(positions))

        ticker_map[tid] = ticker

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    tids = np.concatenate(tid_parts, axis=0)

    if include_timestamps:
        timestamps = np.concatenate(ts_parts, axis=0)
    else:
        timestamps = np.asarray([], dtype=np.int64)

    return LoadedSplit(
        X=X,
        y=y,
        timestamps=timestamps,
        tids=tids,
        metas=metas_all if include_metadata else [],
        ticker_map=ticker_map,
    )

