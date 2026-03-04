import pandas as pd
import pandas_market_calendars as mcal

_NYSE_CAL = mcal.get_calendar("NYSE")
_NY_TZ = "America/New_York"

# Build once (covers 2010-01-01 .. 2026-01-01)
_NYSE_SCHED = _NYSE_CAL.schedule(start_date="2010-01-01", end_date="2026-01-01")

# Fast lookup: python-date -> (market_open_utc, market_close_utc)
# schedule columns are tz-aware UTC timestamps already
_NYSE_OPEN_CLOSE_BY_DATE = {
    idx.date(): (row["market_open"], row["market_close"])
    for idx, row in _NYSE_SCHED.iterrows()
}


def nyse_trade_window_is_valid(entry_ts_utc: pd.Timestamp, exit_ts_utc: pd.Timestamp) -> bool:
    """Uses is_nyse_available(ts) (default args) at entry and exit."""
    return bool(is_nyse_available(entry_ts_utc) and is_nyse_available(exit_ts_utc))


def is_nyse_available(dt, minutes_after_open: int = 10, minutes_before_close: int = 10) -> bool:
    """
    Parameters
    ----------
    dt : pd.Timestamp (tz-aware, UTC)
    minutes_after_open : int
    minutes_before_close : int
    """
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        raise ValueError("dt must be timezone-aware (UTC).")
    ts = ts.tz_convert("UTC")

    session_date_ny = ts.tz_convert(_NY_TZ).date()  # python date in NY time

    oc = _NYSE_OPEN_CLOSE_BY_DATE.get(session_date_ny)
    if oc is None:
        return False  # weekend/holiday/outside precomputed range

    market_open, market_close = oc  # UTC tz-aware pd.Timestamps

    earliest_ok = market_open + pd.Timedelta(minutes=minutes_after_open)
    latest_ok = market_close - pd.Timedelta(minutes=minutes_before_close)

    if earliest_ok > latest_ok:
        return False  # e.g., too strict on early-close day

    return (ts >= earliest_ok) and (ts <= latest_ok)