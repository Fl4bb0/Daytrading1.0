"""
kdata.alpha_vantage_retriever — Low-level Alpha Vantage helpers.

This module targets the documented ``TIME_SERIES_INTRADAY`` endpoint with the
``month=YYYY-MM`` parameter. Some Alpha Vantage plans do not permit historical
month access; in that case we raise a clear error instead of silently burning
through the daily request budget.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, Optional

import pandas as pd
import requests

_API_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(RuntimeError):
    """Raised when Alpha Vantage returns an API or transport error."""


class AlphaVantagePlanError(AlphaVantageError):
    """Raised when the API key does not permit the requested endpoint."""


def get_intraday_month(
    symbol: str,
    month: str,
    *,
    apikey: str,
    interval: str = "1min",
    adjusted: bool = True,
    extended_hours: bool = True,
    datatype: str = "json",
    entitlement: Optional[str] = None,
    timeout_seconds: int = 30,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch one month of intraday bars for *symbol*.

    Parameters
    ----------
    month : str
        Month slice in ``YYYY-MM`` format.
    interval : str
        Alpha interval string. Use ``1min`` for this project.
    """
    params: Dict[str, Any] = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "month": month,
        "outputsize": "full",
        "extended_hours": str(bool(extended_hours)).lower(),
        "adjusted": str(bool(adjusted)).lower(),
        "apikey": apikey,
        "datatype": datatype,
    }
    if entitlement is not None:
        params["entitlement"] = entitlement

    client = session or requests
    try:
        resp = client.get(_API_URL, params=params, timeout=timeout_seconds)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise AlphaVantageError(f"Alpha Vantage request failed for {symbol} {month}: {exc}") from exc

    body = resp.text.strip()
    lower = body.lower()
    if datatype == "json" or lower.startswith("{"):
        payload = resp.json()
        if "Error Message" in payload:
            raise AlphaVantageError(f"Alpha Vantage error for {symbol} {month}: {payload['Error Message']}")
        if "Note" in payload:
            raise AlphaVantageError(f"Alpha Vantage note for {symbol} {month}: {payload['Note']}")
        if "Information" in payload:
            info = str(payload["Information"])
            if "premium" in info.lower() or "subscribe" in info.lower():
                raise AlphaVantagePlanError(f"Alpha Vantage plan does not allow {symbol} {month}: {info}")
            raise AlphaVantageError(f"Alpha Vantage info for {symbol} {month}: {info}")

        key = f"Time Series ({interval})"
        series = payload.get(key)
        if not isinstance(series, dict):
            for k, v in payload.items():
                if isinstance(k, str) and k.lower().startswith("time series") and isinstance(v, dict):
                    series = v
                    break
        if not isinstance(series, dict):
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(series, orient="index")
        if df.empty:
            return df

        rename_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
        df = df.rename(columns=rename_map)
        idx = pd.to_datetime(list(df.index), errors="coerce")
        if idx.tz is None:
            idx = idx.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        idx = idx.tz_convert("UTC")
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df.index = idx
        df.index.name = "timestamp"

        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        return df.sort_index()

    if datatype != "csv" and not lower.startswith("time,"):
        return pd.DataFrame()

    if "time,open,high,low,close,volume" not in lower:
        if "thank you for using alpha vantage" in lower:
            if "premium" in lower or "subscribe" in lower:
                raise AlphaVantagePlanError(f"Alpha Vantage plan does not allow {symbol} {month}: {body}")
            raise AlphaVantageError(f"Alpha Vantage rate/plan response for {symbol} {month}: {body}")
        return pd.DataFrame()

    rows = list(csv.DictReader(StringIO(body)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.rename(columns={"time": "timestamp"})
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]

    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], errors="coerce"))
    if idx.tz is None:
        # Alpha intraday timestamps are in US/Eastern.
        idx = idx.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    idx = idx.tz_convert("UTC")
    df = df[keep]
    df.index = idx
    df.index.name = "timestamp"

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    return df.sort_index()



def get_current_price(symbol: str, *, apikey: str, timeout_seconds: int = 15) -> Dict[str, Any]:
    """Fetch current quote from Alpha Vantage GLOBAL_QUOTE endpoint."""
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": apikey,
        "datatype": "json",
    }
    try:
        resp = requests.get(_API_URL, params=params, timeout=timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise AlphaVantageError(f"Alpha Vantage quote failed for {symbol}: {exc}") from exc

    quote = payload.get("Global Quote") or {}
    price_raw = quote.get("05. price")
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        price = float(str(price_raw)) if price_raw is not None else None
    except (TypeError, ValueError):
        price = None
    return {
        "symbol": symbol,
        "current_price": price,
        "timestamp": timestamp,
        "source": "AlphaVantage",
    }

