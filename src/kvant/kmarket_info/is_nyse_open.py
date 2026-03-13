"""
kmarket_info.is_nyse_open — NYSE trading calendar and market-hours utilities.

is_nyse_available(dt, minutes_after_open, minutes_before_close) -> bool
nyse_trade_window_is_valid(entry_ts_utc, exit_ts_utc)           -> bool
nyse_market_close_today(now_utc)                                 -> pd.Timestamp | None
is_nyse_trading_day(date)                                        -> bool
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal

_NYSE_CAL = mcal.get_calendar("NYSE")
_NY_TZ = "America/New_York"

# Build once (covers 2010-01-01 .. 2035-01-01)
_NYSE_SCHED = _NYSE_CAL.schedule(start_date="2010-01-01", end_date="2035-01-01")

# Fast lookup: python date -> (market_open_utc, market_close_utc)
_NYSE_OPEN_CLOSE_BY_DATE: dict[date, tuple[pd.Timestamp, pd.Timestamp]] = {
    idx.date(): (row["market_open"], row["market_close"])
    for idx, row in _NYSE_SCHED.iterrows()
}


def is_nyse_trading_day(d: date) -> bool:
    """Return True if *d* is a regular or early-close NYSE trading day."""
    return d in _NYSE_OPEN_CLOSE_BY_DATE


def nyse_market_close_today(now_utc: Optional[pd.Timestamp] = None) -> Optional[pd.Timestamp]:
    """
    Return the NYSE market-close timestamp (UTC) for today, or None if today
    is not a trading day.

    Parameters
    ----------
    now_utc : pd.Timestamp (tz-aware UTC) or None
        Defaults to the current UTC time if not supplied.
    """
    if now_utc is None:
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    today_ny = now_utc.tz_convert(_NY_TZ).date()
    oc = _NYSE_OPEN_CLOSE_BY_DATE.get(today_ny)
    return oc[1] if oc is not None else None


def nyse_trade_window_is_valid(
    entry_ts_utc: pd.Timestamp,
    exit_ts_utc: pd.Timestamp,
) -> bool:
    """Return True if both entry and exit are within valid trading windows."""
    return bool(is_nyse_available(entry_ts_utc) and is_nyse_available(exit_ts_utc))


def is_nyse_available(
    dt,
    minutes_after_open: int = 10,
    minutes_before_close: int = 10,
) -> bool:
    """
    Return True if *dt* falls within the NYSE trading window, excluding the
    first *minutes_after_open* and last *minutes_before_close* minutes.

    Parameters
    ----------
    dt                   : pd.Timestamp (tz-aware, UTC)
    minutes_after_open   : grace period after open (default 10)
    minutes_before_close : grace period before close (default 10)
    """
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        raise ValueError("dt must be timezone-aware (UTC).")
    ts = ts.tz_convert("UTC")

    session_date_ny = ts.tz_convert(_NY_TZ).date()
    oc = _NYSE_OPEN_CLOSE_BY_DATE.get(session_date_ny)
    if oc is None:
        return False  # weekend / holiday / outside precomputed range

    market_open, market_close = oc
    earliest_ok = market_open + pd.Timedelta(minutes=minutes_after_open)
    latest_ok = market_close - pd.Timedelta(minutes=minutes_before_close)

    if earliest_ok > latest_ok:
        return False  # edge case: very short early-close session

    return bool((ts >= earliest_ok) and (ts <= latest_ok))
