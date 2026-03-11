"""
sampling._utils — Internal helpers shared across all sampler implementations.

Not part of the public API — import from the concrete sampler modules instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kvant.utils.time_utils import ensure_utc_sorted_index


# ---------------------------------------------------------------------------
# OHLCV aggregation
# ---------------------------------------------------------------------------

def aggregate_ohlcv(df: pd.DataFrame, ends: np.ndarray) -> pd.DataFrame:
    """
    Aggregate raw minute bars into OHLCV bars closing at each index in *ends*.

    Parameters
    ----------
    df   : UTC-sorted OHLCV DataFrame.
    ends : 0-based row indices at which bars close (from any bar-end algorithm).

    Returns
    -------
    New DataFrame with one row per bar, indexed by the close-row timestamp.
    """
    if len(df) == 0:
        return df.copy()

    if len(ends) == 0:
        seg = df.iloc[:]
        bar: dict = {}
        if "open"   in seg.columns: bar["open"]   = float(seg["open"].iloc[0])
        if "high"   in seg.columns: bar["high"]   = float(seg["high"].max())
        if "low"    in seg.columns: bar["low"]    = float(seg["low"].min())
        if "close"  in seg.columns: bar["close"]  = float(seg["close"].iloc[-1])
        if "volume" in seg.columns: bar["volume"] = float(seg["volume"].sum())
        return pd.DataFrame([bar], index=pd.DatetimeIndex([df.index[-1]]))

    ends = np.unique(np.clip(ends, 0, len(df) - 1))
    rows, idx = [], []
    start = 0
    for end in ends:
        if end < start:
            continue
        seg = df.iloc[start : end + 1]
        bar = {}
        if "open"   in seg.columns: bar["open"]   = float(seg["open"].iloc[0])
        if "high"   in seg.columns: bar["high"]   = float(seg["high"].max())
        if "low"    in seg.columns: bar["low"]    = float(seg["low"].min())
        if "close"  in seg.columns: bar["close"]  = float(seg["close"].iloc[-1])
        if "volume" in seg.columns: bar["volume"] = float(seg["volume"].sum())
        rows.append(bar)
        idx.append(df.index[end])
        start = end + 1

    return ensure_utc_sorted_index(pd.DataFrame(rows, index=pd.DatetimeIndex(idx)))


# ---------------------------------------------------------------------------
# Flow helpers
# ---------------------------------------------------------------------------

def flow_per_day(df: pd.DataFrame, flow: np.ndarray) -> float:
    """Return average total flow per calendar trading day."""
    idx = ensure_utc_sorted_index(df).index
    n_days = int(idx.normalize().nunique())
    if n_days == 0:
        return 0.0
    return float(flow.sum()) / n_days


def ticks_per_day(df: pd.DataFrame) -> float:
    """Return average number of rows per calendar trading day."""
    if len(df) == 0:
        return 0.0
    idx = ensure_utc_sorted_index(df).index
    n_days = int(idx.normalize().nunique())
    return len(df) / n_days if n_days > 0 else 0.0


def tick_directions(close: np.ndarray) -> np.ndarray:
    """
    Apply the tick rule to a close-price array.

    Returns +1.0 / -1.0 per row (never 0). First tick is +1.
    """
    n = len(close)
    b = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            b[i] = 1.0
        elif diff < 0.0:
            b[i] = -1.0
        else:
            b[i] = b[i - 1]
    return b
