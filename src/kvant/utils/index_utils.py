"""
utils.index_utils — Sample-index helpers for train/val/test splitting.

Extracted from ml_prepare_data/prepare_experiment.py.

valid_target_positions(labels, lookback_L)  → np.ndarray of int positions
in_split(ts, split, val_start, test_start)  → bool
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from kvant.utils.time_utils import as_dt64_utc_naive


def valid_target_positions(labels: np.ndarray, lookback_L: int) -> np.ndarray:
    """
    Return positions where a label exists (≠ -1) and enough lookback history
    is available (position ≥ lookback_L).
    """
    pos = np.arange(len(labels))
    return pos[(labels != -1) & (pos >= lookback_L)]


def in_split(
    ts,
    split: str,
    val_start,
    test_start,
    test_end=None,
) -> bool:
    """
    Return True if timestamp `ts` belongs to `split` given the boundaries.

    Parameters
    ----------
    ts         : any timestamp-like value.
    split      : "train" | "val" | "test"
    val_start  : first timestamp of the validation split (or None).
    test_start : first timestamp of the test split (or None).
    test_end   : exclusive end of the test split (or None for no upper bound).
                 Lets callers append a label-lookahead buffer of bars after
                 the nominal test window without those buffer bars becoming
                 eligible trade entries themselves.
    """
    ts = as_dt64_utc_naive(ts)
    val_start = as_dt64_utc_naive(val_start)
    test_start = as_dt64_utc_naive(test_start)
    test_end = as_dt64_utc_naive(test_end)

    if split == "train":
        cut = val_start if val_start is not None else test_start
        return True if cut is None else bool(ts < cut)

    if split == "val":
        if val_start is None:
            return False
        if test_start is None:
            return bool(ts >= val_start)
        return bool((ts >= val_start) and (ts < test_start))

    if split == "test":
        if test_start is None:
            return False
        if test_end is None:
            return bool(ts >= test_start)
        return bool((ts >= test_start) and (ts < test_end))

    raise ValueError(f"Unknown split: {split!r}. Expected 'train', 'val', or 'test'.")
