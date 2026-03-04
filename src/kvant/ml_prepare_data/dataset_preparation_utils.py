from __future__ import annotations

import pandas as pd


def ensure_utc_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - df indexed by DatetimeIndex, or
      - df with a 'timestamp' column convertible to datetime.

    Returns a copy (if needed) indexed by UTC DatetimeIndex, sorted increasing.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        out = df
    else:
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column.")
        out = df.copy()
        out.index = pd.to_datetime(out["timestamp"], utc=True)

    out = out.sort_index()
    if out.index.tz is None:
        out = out.copy()
        out.index = out.index.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out
