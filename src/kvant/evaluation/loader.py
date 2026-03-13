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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from kvant.utils.split_loader import load_split_from_index


def load_split(
    exp_dir: Path,
    split: str,
    tickers: Optional[List[str]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Optional[dict]],
    Dict[int, str],
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

    # Validate requested split tickers against known ids before loading
    _ = {ticker_to_tid[t] for t in split_tickers}

    loaded = load_split_from_index(
        exp_dir=exp_dir,
        index=index,
        lookback_L=lookback_L,
        allowed_tickers=split_tickers,
        include_timestamps=True,
        include_metadata=True,
    )
    return loaded.X, loaded.y, loaded.timestamps, loaded.tids, loaded.metas, loaded.ticker_map
