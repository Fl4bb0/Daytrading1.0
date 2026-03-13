"""
utils.time_utils — UTC datetime normalisation helpers.

Extracted from ml_prepare_data/prepare_experiment.py and
ml_prepare_data/dataset_preparation_utils.py so every sub-package
can import without circular dependencies.

as_dt64_utc_naive(x)           → np.datetime64[ns], UTC-naive
ensure_utc_sorted_index(df)    → df with tz-naive UTC DatetimeIndex, sorted asc
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def as_dt64_utc_naive(x) -> Optional[np.datetime64]:
    """Convert any timestamp-like value to UTC-naive np.datetime64[ns]."""
    if x is None:
        return None
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[ns]")
    if isinstance(x, pd.Timestamp):
        if x.tz is not None:
            x = x.tz_convert("UTC").tz_localize(None)
        return x.to_datetime64().astype("datetime64[ns]")
    return np.datetime64(
        pd.Timestamp(x, tz="UTC").tz_localize(None)
    ).astype("datetime64[ns]")


def ensure_utc_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that df has a monotonically increasing, UTC-naive DatetimeIndex.
    Converts tz-aware → UTC → tz-localize(None) then sorts.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    elif idx.tz is not None:
        df = df.copy()
        df.index = idx.tz_convert("UTC").tz_localize(None)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df
