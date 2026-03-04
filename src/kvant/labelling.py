import dataclasses
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.kvant.kmarket_info.is_nyse_open import nyse_trade_window_is_valid


@dataclasses.dataclass(frozen=True)
class TripleBarLabel:
    bar_open_time: pd.Timestamp
    bar_close_time: pd.Timestamp
    label: int           # 0 = stop-loss (DOWN), 1 = vertical/time exit, 2 = take-profit (UP)
    pnl_fraction: float  # (exit - entry) / entry
    pnl_absolute: float  # (exit - entry) in price units (e.g., $ per share)


def _to_utc_ts(x: Union[pd.Timestamp, str]) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def tripple_bar_label(
    data: pd.DataFrame,
    time_start: Union[pd.Timestamp, str],
    width: int,
    height: float,
) -> Optional[TripleBarLabel]:
    """
    Triple-barrier label for the bar at/after `time_start`.

    Parameters
    ----------
    data : pd.DataFrame
        UTC time-indexed OHLCV-like dataframe with columns at least: open, high, low, close.
    time_start : Timestamp-like
        UTC timestamp (or coercible to pd.Timestamp).
    width : int
        Vertical barrier in minutes (max time in position).
    height : float
        Fractional barrier size (e.g. 0.01 means +/- 1% from entry).

    Returns
    -------
    TripleBarLabel | None
        Returns None if:
        - data is empty / timestamps not found
        - prices are invalid
        - entry/exit timestamps are not within the exchange trading window
    """
    if data is None or data.empty:
        return None

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"data must contain columns {sorted(required_cols)}")


    # df = data.sort_index()
    df = data
    ts0 = _to_utc_ts(time_start)


    # Find the first bar at/after time_start
    pos = df.index.searchsorted(ts0, side="left")
    if pos >= len(df.index):
        return None

    entry_ts = df.index[pos]

    # Vertical barrier target and last available bar <= it
    end_target = entry_ts + pd.Timedelta(minutes=int(width))
    end_pos = df.index.searchsorted(end_target, side="right") - 1
    if end_pos < pos:
        return None

    exit_ts_vertical = df.index[end_pos]

    # If entry/vertical-exit can't be executed in allowed market window, reject immediately
    if not nyse_trade_window_is_valid(entry_ts, exit_ts_vertical):
        return None

    entry_price = float(df.iloc[pos]["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    h = float(height)
    upper = entry_price * (1.0 + h)
    lower = entry_price * (1.0 - h)

    path = df.iloc[pos : end_pos + 1]

    hit_up = path["high"] >= upper
    hit_dn = path["low"] <= lower

    up_times = path.index[hit_up]
    dn_times = path.index[hit_dn]

    # Decide earliest barrier hit (if any)
    if len(up_times) == 0 and len(dn_times) == 0:
        # Vertical barrier exit
        label = 1
        exit_ts = exit_ts_vertical
        exit_price = float(path.loc[exit_ts, "close"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            return None
    else:
        first_up = up_times[0] if len(up_times) else None
        first_dn = dn_times[0] if len(dn_times) else None

        # If both hit in the same bar, choose stop-loss (conservative)
        if first_dn is not None and (first_up is None or first_dn <= first_up):
            label = 0
            exit_ts = first_dn
            exit_price = lower
        else:
            label = 2
            exit_ts = first_up
            exit_price = upper

    # Enforce market-window availability for the actual realized exit timestamp too
    if not nyse_trade_window_is_valid(entry_ts, exit_ts):
        return None

    pnl_abs = float(exit_price - entry_price)
    pnl_frac = float(pnl_abs / entry_price)

    return TripleBarLabel(
        bar_open_time=entry_ts,
        bar_close_time=exit_ts,
        label=int(label),
        pnl_fraction=pnl_frac,
        pnl_absolute=pnl_abs,
    )