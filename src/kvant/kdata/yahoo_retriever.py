from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Dict, Any, List

import yfinance as yf


def _to_utc_dt(ts: Union[str, int, float, datetime]) -> datetime:
    """Convert a timestamp (datetime, ISO string, or epoch ms/seconds) to a timezone-aware UTC datetime."""
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)):
        # treat as seconds if within reasonable range, otherwise milliseconds
        if ts > 1e12:  # clearly ms
            dt = datetime.fromtimestamp(ts / 1000.0)
        else:
            dt = datetime.fromtimestamp(ts)
    elif isinstance(ts, str):
        # Try ISO formats first
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            # fallback to parsing common formats
            from dateutil import parser

            dt = parser.parse(ts)
    else:
        raise TypeError("Unsupported timestamp type")

    # make timezone-aware (assume naive -> local/UTC? we treat naive as UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _to_date_str(ts: Optional[Union[str, int, float, datetime]]) -> Optional[str]:
    """Convert any supported timestamp to a 'YYYY-MM-DD' string for yfinance start/end arguments."""
    if ts is None:
        return None
    return _to_utc_dt(ts).strftime("%Y-%m-%d")


def _index_to_utc_dt(ix) -> datetime:
    """Convert a pandas DatetimeIndex entry to a timezone-aware UTC datetime."""
    dt = ix.to_pydatetime()
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def get_current_price(symbol: str,
                      interval: str = "1m",
                      include_prepost: bool = True,
                      realtime_threshold_seconds: int = 120) -> Dict[str, Any]:
    """
    Return the most recent price information for `symbol`.

    Tries multiple fallbacks in this order:
    1. Ticker.fast_info.last_price (if available)
    2. Ticker.history with the requested `interval` (defaults to 1m) for the latest bar
    3. Ticker.info regularMarketPrice

    Returns a dict with keys: symbol, current_price, timestamp (ISO UTC), age_seconds, source, and bar fields when available.
    Note: Yahoo data may be delayed. `age_seconds` measures the age of the timestamp returned by Yahoo.
    """
    t = yf.Ticker(symbol)
    out: Dict[str, Any] = {"symbol": symbol}

    # 1) Try fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi and "last_price" in fi and fi["last_price"] is not None:
            out.update({"current_price": fi["last_price"], "source": "fast_info"})
    except Exception:
        pass

    # 2) Try intraday history for timestamped bar
    try:
        hist = t.history(period="1d", interval=interval, prepost=include_prepost, actions=False)
        if not hist.empty:
            last = hist.iloc[-1]
            ts_utc = _index_to_utc_dt(last.name)
            age_seconds = (datetime.now(timezone.utc) - ts_utc).total_seconds()

            bar = {
                "timestamp": ts_utc.isoformat(),
                "open": float(last.get("Open", float("nan"))),
                "high": float(last.get("High", float("nan"))),
                "low": float(last.get("Low", float("nan"))),
                "close": float(last.get("Close", float("nan"))),
                "volume": int(last.get("Volume", 0)) if not last.get("Volume") is None else None,
            }
            out.setdefault("current_price", bar["close"])
            out.update({"bar": bar, "age_seconds": int(age_seconds), "is_recent": age_seconds <= realtime_threshold_seconds, "source": out.get("source", "history")})
    except Exception:
        pass

    # 3) Fallback to .info or ensure we have current_price key
    try:
        if "current_price" not in out or out.get("current_price") is None:
            info = t.info or {}
            price = info.get("regularMarketPrice") or info.get("previousClose")
            out.update({"current_price": price, "source": out.get("source", "info")})
    except Exception:
        out.setdefault("current_price", None)

    return out


def get_price_at(symbol: str,
                 when: Union[str, int, float, datetime],
                 interval: str = "1m",
                 window_seconds: int = 120) -> Optional[Dict[str, Any]]:
    """
    Return the price/bar for `symbol` closest to the provided `when` timestamp.

    `when` can be a datetime, ISO string, or epoch (seconds or ms). The function fetches a small historical window
    around `when` (default +/- `window_seconds`) using the given `interval`, then returns the closest bar.

    Returns None if no data is available in the window.
    """
    dt = _to_utc_dt(when)
    start = _to_date_str(dt - timedelta(seconds=window_seconds))
    end = _to_date_str(dt + timedelta(seconds=window_seconds) + timedelta(days=1))

    t = yf.Ticker(symbol)
    try:
        hist = t.history(start=start, end=end, interval=interval, actions=False)
        if hist.empty:
            return None

        # Find the row with timestamp closest to dt
        rows = []
        for ix, row in hist.iterrows():
            row_dt = _index_to_utc_dt(ix)
            diff = abs((row_dt - dt).total_seconds())
            rows.append((diff, row_dt, row))

        rows.sort(key=lambda x: x[0])
        diff, row_dt, row = rows[0]
        return {
            "symbol": symbol,
            "requested_timestamp": dt.isoformat(),
            "matched_timestamp": row_dt.isoformat(),
            "age_seconds": int(abs((datetime.now(timezone.utc) - row_dt).total_seconds())),
            "interval": interval,
            "distance_seconds": int(diff),
            "open": float(row.get("Open", float("nan"))),
            "high": float(row.get("High", float("nan"))),
            "low": float(row.get("Low", float("nan"))),
            "close": float(row.get("Close", float("nan"))),
            "volume": int(row.get("Volume", 0)) if not row.get("Volume") is None else None,
        }
    except Exception:
        return None


def get_history(symbol: str,
                start: Optional[Union[str, datetime]] = None,
                end: Optional[Union[str, datetime]] = None,
                period: Optional[str] = None,
                interval: str = "1d",
                prepost: bool = False,
                actions: bool = False,
                as_pandas: bool = True,
                **kwargs) -> Union[Any, List[Dict[str, Any]]]:
    """
    Return historical data for `symbol`.

    Either supply `start`/`end` (datetimes or ISO strings) or `period` (e.g. '1mo', '1y').
    By default returns a pandas.DataFrame; set `as_pandas=False` to get a list of dict records with ISO timestamps.
    """
    t = yf.Ticker(symbol)

    s = _to_date_str(start)
    e = _to_date_str(end)

    hist = t.history(start=s, end=e, period=period, interval=interval, prepost=prepost, actions=actions, **kwargs)

    if as_pandas:
        return hist

    # Convert to list of dicts
    out: List[Dict[str, Any]] = []
    try:
        for ix, row in hist.iterrows():
            row_dt = _index_to_utc_dt(ix)
            out.append({
                "timestamp": row_dt.isoformat(),
                "open": None if row.get("Open") is None else float(row.get("Open")),
                "high": None if row.get("High") is None else float(row.get("High")),
                "low": None if row.get("Low") is None else float(row.get("Low")),
                "close": None if row.get("Close") is None else float(row.get("Close")),
                "volume": None if row.get("Volume") is None else int(row.get("Volume")),
            })
    except Exception:
        return []

    return out


def test():
    # Simple demo: fetch AAPL current price and last 3 daily history points
    try:
        print("Demo: fetching AAPL current price...")
        cur = get_current_price("AAPL")
        print(cur)

        # print("\nDemo: fetching AAPL price ~1 day ago (price at timestamp)...")
        # when = datetime.now(timezone.utc) - timedelta(days=1)
        # pa = get_price_at("AAPL", when, interval="1d", window_seconds=60 * 60 * 24)
        # print(pa)

        # print("\nDemo: fetching last 5 daily history entries...")
        # h = get_history("AAPL", period="5d", interval="1d", as_pandas=False)
        # print(h)
    except Exception as e:
        print("Demo failed (likely missing network or yfinance dependency):", e)


def previous_week_test():
    # Test fetching price at a specific timestamp (e.g. 1 week ago)
    try:
        print("\nTest: fetching all AAPL data for the past week...")
        now = _to_utc_dt(datetime.now(timezone.utc))
        one_week_ago = _to_utc_dt(now - timedelta(days=7))
        history = get_history("AAPL", start=one_week_ago, end=now, interval="1m", as_pandas=True)
        print(f"Fetched {len(history)}")
        print(f"Data type: {type(history)}")
    except Exception as e:
        print("Test failed:", e)

def previous_week_download_test():
    # Test fetching price at a specific timestamp (e.g. 1 week ago)
    try:
        print("\nTest: fetching all AAPL data for the past week...")
        now = _to_utc_dt(datetime.now(timezone.utc))
        one_week_ago = _to_utc_dt(now - timedelta(days=7))

    except Exception as e:
        print("Test failed:", e)

if __name__ == "__main__":
    previous_week_test()