"""
kdata.yahoo_retriever — Low-level yfinance helpers.

These are intentionally plain functions with no ABC dependency so they
can be tested and used in isolation. YahooRetriever (retriever.py)
calls these internally.

get_history(symbol, *, start, end, period, interval, prepost, actions, as_pandas)
get_current_price(symbol, *, interval, include_prepost, realtime_threshold_seconds)
get_price_at(symbol, when, *, interval, window_seconds)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import yfinance as yf


# ---------------------------------------------------------------------------
# Internal timestamp helpers
# ---------------------------------------------------------------------------

def _to_utc_dt(ts: Union[str, int, float, datetime]) -> datetime:
    """Convert any timestamp-like value to a timezone-aware UTC datetime."""
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts / 1000.0 if ts > 1e12 else ts)
    elif isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            from dateutil import parser
            dt = parser.parse(ts)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _to_date_str(ts: Optional[Union[str, int, float, datetime]]) -> Optional[str]:
    """Return 'YYYY-MM-DD' string or None, for use as yfinance start/end args."""
    return None if ts is None else _to_utc_dt(ts).strftime("%Y-%m-%d")


def _index_to_utc_dt(ix: Any) -> datetime:
    """Convert a single pandas DatetimeIndex entry to a UTC-aware datetime."""
    dt = ix.to_pydatetime()
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_history(
    symbol: str,
    *,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    period: Optional[str] = None,
    interval: str = "1d",
    prepost: bool = False,
    actions: bool = False,
    as_pandas: bool = True,
    **kwargs,
) -> Any:
    """
    Fetch OHLCV history for *symbol* from Yahoo Finance.

    Supply either ``start``/``end`` or ``period`` (e.g. ``'6mo'``, ``'1y'``).

    Parameters
    ----------
    as_pandas : bool
        If True (default) return a ``pd.DataFrame``; otherwise a list of dicts
        with ISO-formatted UTC timestamps.
    """
    t = yf.Ticker(symbol)
    hist = t.history(
        start=_to_date_str(start),
        end=_to_date_str(end),
        period=period,
        interval=interval,
        prepost=prepost,
        actions=actions,
        **kwargs,
    )

    if as_pandas:
        return hist

    out: List[Dict[str, Any]] = []
    for ix, row in hist.iterrows():
        row_dt = _index_to_utc_dt(ix)
        out.append({
            "timestamp": row_dt.isoformat(),
            "open":   None if row.get("Open")   is None else float(row["Open"]),
            "high":   None if row.get("High")   is None else float(row["High"]),
            "low":    None if row.get("Low")    is None else float(row["Low"]),
            "close":  None if row.get("Close")  is None else float(row["Close"]),
            "volume": None if row.get("Volume") is None else int(row["Volume"]),
        })
    return out


def get_current_price(
    symbol: str,
    *,
    interval: str = "1m",
    include_prepost: bool = True,
    realtime_threshold_seconds: int = 120,
) -> Dict[str, Any]:
    """
    Return the most-recent price for *symbol* with multiple fallbacks:
      1. ``fast_info.last_price``
      2. Latest bar from ``Ticker.history``
      3. ``info.regularMarketPrice``

    The returned dict always contains ``symbol`` and ``current_price``.
    ``age_seconds`` measures the reported Yahoo timestamp, which may be delayed.
    """
    t = yf.Ticker(symbol)
    out: Dict[str, Any] = {"symbol": symbol}

    # 1) fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi is not None and fi.get("last_price") is not None:
            out.update({"current_price": fi["last_price"], "source": "fast_info"})
    except Exception:
        pass

    # 2) intraday history bar
    try:
        hist = t.history(period="1d", interval=interval, prepost=include_prepost, actions=False)
        if not hist.empty:
            last = hist.iloc[-1]
            ts_utc = _index_to_utc_dt(last.name)
            age = (datetime.now(timezone.utc) - ts_utc).total_seconds()
            bar = {
                "timestamp": ts_utc.isoformat(),
                "open":   float(last.get("Open",   float("nan"))),
                "high":   float(last.get("High",   float("nan"))),
                "low":    float(last.get("Low",    float("nan"))),
                "close":  float(last.get("Close",  float("nan"))),
                "volume": int(last["Volume"]) if last.get("Volume") is not None else None,
            }
            out.setdefault("current_price", bar["close"])
            out.update({
                "bar": bar,
                "age_seconds": int(age),
                "is_recent": age <= realtime_threshold_seconds,
                "source": out.get("source", "history"),
            })
    except Exception:
        pass

    # 3) .info fallback
    try:
        if out.get("current_price") is None:
            info = t.info or {}
            price = info.get("regularMarketPrice") or info.get("previousClose")
            out.update({"current_price": price, "source": out.get("source", "info")})
    except Exception:
        out.setdefault("current_price", None)

    return out


def get_price_at(
    symbol: str,
    when: Union[str, int, float, datetime],
    *,
    interval: str = "1m",
    window_seconds: int = 120,
) -> Optional[Dict[str, Any]]:
    """
    Return the bar for *symbol* closest to *when*.

    Fetches a small window around *when* (±``window_seconds``) at the given
    *interval* and picks the nearest bar. Returns ``None`` if no data.
    """
    dt = _to_utc_dt(when)
    start = _to_date_str(dt - timedelta(seconds=window_seconds))
    end = _to_date_str(dt + timedelta(seconds=window_seconds) + timedelta(days=1))

    t = yf.Ticker(symbol)
    try:
        hist = t.history(start=start, end=end, interval=interval, actions=False)
        if hist.empty:
            return None

        rows = sorted(
            [
                (abs((_index_to_utc_dt(ix) - dt).total_seconds()), _index_to_utc_dt(ix), row)
                for ix, row in hist.iterrows()
            ],
            key=lambda x: x[0],
        )
        diff, row_dt, row = rows[0]
        return {
            "symbol": symbol,
            "requested_timestamp": dt.isoformat(),
            "matched_timestamp": row_dt.isoformat(),
            "age_seconds": int(abs((datetime.now(timezone.utc) - row_dt).total_seconds())),
            "interval": interval,
            "distance_seconds": int(diff),
            "open":   float(row.get("Open",   float("nan"))),
            "high":   float(row.get("High",   float("nan"))),
            "low":    float(row.get("Low",    float("nan"))),
            "close":  float(row.get("Close",  float("nan"))),
            "volume": int(row["Volume"]) if row.get("Volume") is not None else None,
        }
    except Exception:
        return None
