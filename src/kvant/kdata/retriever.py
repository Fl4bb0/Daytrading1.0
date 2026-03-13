"""
kdata.retriever — Unified data-retrieval interface.

Back-ends are subclasses of :class:`DataRetriever`. Only Yahoo Finance
is active; the pattern shows exactly what to implement to add a new one.

Active back-ends
----------------
  YahooRetriever        — Yahoo Finance via yfinance (live/recent reads).
  AlphaVantageRetriever — Alpha Vantage intraday month-slice fetches.
  HybridRetriever       — Yahoo for recent/live, Alpha Vantage for history.

Adding a new back-end
---------------------
  1. Create ``kdata/<name>_retriever.py`` with the raw data-fetch logic.
  2. Subclass ``DataRetriever`` and implement ``get_history`` + ``get_ticker_data``.
  3. Optionally override ``get_current_price`` for a cheaper real-time endpoint.

Typical usage
-------------
    from kvant.kdata.retriever import HybridRetriever

    r = HybridRetriever()
    df   = r.get_history("AAPL")
    data = r.get_ticker_data(["AAPL", "MSFT"])
    cur  = r.get_current_price("AAPL")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import os
from typing import Dict, List, Optional, Union, cast

import pandas as pd
from dotenv import find_dotenv, load_dotenv


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataRetriever(ABC):
    """Abstract interface that every data-retrieval back-end must satisfy."""

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Return a UTC-indexed OHLCV DataFrame for a single *symbol*."""
        ...

    @abstractmethod
    def get_ticker_data(
        self,
        symbols: List[str],
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{ticker: DataFrame}`` for every symbol in *symbols*."""
        ...

    def get_current_price(self, symbol: str, **kwargs) -> dict:
        """
        Return the most-recent price info for *symbol*.

        Default implementation fetches the last bar via :meth:`get_history`.
        Override when the back-end exposes a cheaper real-time endpoint.
        """
        df = self.get_history(symbol, period="1d", interval="1m", **kwargs)
        if df.empty:
            return {"symbol": symbol, "current_price": None, "source": self.__class__.__name__}
        last = df.iloc[-1]
        return {
            "symbol": symbol,
            "current_price": float(last.get("close", last.get("Close", float("nan")))),
            "timestamp": df.index[-1].isoformat(),
            "source": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# Yahoo Finance back-end
# ---------------------------------------------------------------------------

class YahooRetriever(DataRetriever):
    """
    Retrieves OHLCV data from Yahoo Finance via *yfinance*.

    Parameters
    ----------
    interval : str
        Default bar interval (e.g. ``'1m'``, ``'1h'``, ``'1d'``).
        Can be overridden per-call.
    period : str | None
        Default look-back period (e.g. ``'6mo'``, ``'1y'``).
        Used when neither ``start``/``end`` nor a per-call ``period`` is supplied.
    prepost : bool
        Include pre-/post-market bars. Defaults to False.
    """

    def __init__(
        self,
        interval: str = "1d",
        period: Optional[str] = "6mo",
        prepost: bool = False,
    ) -> None:
        self.default_interval = interval
        self.default_period = period
        self.prepost = prepost

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_history(
        self,
        symbol: str,
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return a UTC-indexed, lower-cased-column OHLCV DataFrame.

        Columns: ``open``, ``high``, ``low``, ``close``, ``volume``.

        Daily bars (``interval='1d'``) have their midnight-UTC timestamps
        snapped to 14:30 UTC (= 09:30 ET, NYSE open) so downstream labellers
        that check trading hours don't reject every row.
        """
        from kvant.kdata.yahoo_retriever import get_history as _get_history

        _interval = interval or self.default_interval
        _period = period or (None if (start or end) else self.default_period)

        df: pd.DataFrame = _get_history(
            symbol,
            start=start,
            end=end,
            period=_period,
            interval=_interval,
            prepost=self.prepost,
            as_pandas=True,
            **kwargs,
        )

        if df.empty:
            return df

        # Normalise index to UTC-aware DatetimeIndex
        df.index = pd.to_datetime(df.index, utc=True)

        # Lower-case columns; drop yfinance multi-level artifacts
        df.columns = [c.lower() for c in df.columns]
        for col in ("ticker", "price"):
            if col in df.columns:
                df = df.drop(columns=col)

        # Snap daily midnight timestamps to NYSE open (09:30 ET = 14:30 UTC)
        if _interval == "1d":
            df.index = df.index.normalize() + pd.Timedelta(hours=14, minutes=30)

        return df

    def get_ticker_data(
        self,
        symbols: List[str],
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return ``{ticker: DataFrame}`` for every non-empty symbol.

        Symbols for which Yahoo returns no data are silently omitted.
        """
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.get_history(
                sym, start=start, end=end, period=period, interval=interval, **kwargs
            )
            if not df.empty:
                out[sym] = df
        return out

    # ------------------------------------------------------------------
    # Yahoo-specific extras (not part of the base interface)
    # ------------------------------------------------------------------

    def get_current_price(self, symbol: str, **kwargs) -> dict:
        """Use Yahoo's fast_info / intraday history for a fresher price."""
        from kvant.kdata.yahoo_retriever import get_current_price as _get_current_price
        return _get_current_price(symbol, **kwargs)

    def get_price_at(
        self,
        symbol: str,
        when: Union[str, int, float, datetime],
        **kwargs,
    ) -> Optional[dict]:
        """Return the bar closest to *when*. See yahoo_retriever.get_price_at."""
        from kvant.kdata.yahoo_retriever import get_price_at as _get_price_at
        return _get_price_at(symbol, when, **kwargs)


# ---------------------------------------------------------------------------
# Alpha Vantage back-end
# ---------------------------------------------------------------------------

class AlphaVantageRetriever(DataRetriever):
    """Retrieves OHLCV data from Alpha Vantage intraday month slices."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        interval: str = "1m",
        extended_hours: bool = True,
        adjusted: bool = True,
        datatype: str = "json",
        entitlement: Optional[str] = None,
    ) -> None:
        # Auto-load .env so ALPHAVANTAGE_API_KEY works without shell export.
        load_dotenv(find_dotenv(usecwd=True), override=False)
        key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not key:
            raise ValueError("Alpha Vantage API key missing. Set ALPHAVANTAGE_API_KEY or pass api_key=...")
        self.api_key = key
        self.default_interval = interval
        self.extended_hours = extended_hours
        self.adjusted = adjusted
        self.datatype = datatype
        self.entitlement = entitlement

    def get_history(
        self,
        symbol: str,
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1m",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return historical 1-minute bars.

        Supports either:
          - month="YYYY-MM" in kwargs, or
          - start/end range (fetched month-by-month and concatenated).
        """
        from kvant.kdata.alpha_vantage_retriever import get_intraday_month

        av_interval = "1min" if (interval or self.default_interval) == "1m" else interval
        month = kwargs.pop("month", None)
        out: List[pd.DataFrame] = []

        if month is not None:
            df = get_intraday_month(
                symbol,
                month,
                apikey=self.api_key,
                interval=av_interval,
                adjusted=self.adjusted,
                extended_hours=self.extended_hours,
                datatype=self.datatype,
                entitlement=self.entitlement,
            )
            return self._slice(df, start=start, end=end)

        range_start, range_end = self._resolve_range(start=start, end=end, period=period)
        for m in self._months_between(range_start, range_end):
            part = get_intraday_month(
                symbol,
                m,
                apikey=self.api_key,
                interval=av_interval,
                adjusted=self.adjusted,
                extended_hours=self.extended_hours,
                datatype=self.datatype,
                entitlement=self.entitlement,
            )
            if not part.empty:
                out.append(part)

        if not out:
            return pd.DataFrame()
        full = pd.concat(out).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        return self._slice(full, start=range_start, end=range_end)

    def get_ticker_data(
        self,
        symbols: List[str],
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.get_history(
                sym,
                start=start,
                end=end,
                period=period,
                interval=interval,
                **kwargs,
            )
            if not df.empty:
                out[sym] = df
        return out

    def get_current_price(self, symbol: str, **kwargs) -> dict:
        from kvant.kdata.alpha_vantage_retriever import get_current_price as _get_current_price
        return _get_current_price(symbol, apikey=self.api_key, **kwargs)

    @staticmethod
    def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
        months: List[str] = []
        current = start.to_period("M")
        last = end.to_period("M")
        latest_complete = pd.Timestamp.now(tz="UTC").to_period("M") - 1
        while current <= last:
            if current <= latest_complete:
                months.append(str(current))
            current += 1
        return months

    @staticmethod
    def _slice(
        df: pd.DataFrame,
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        out = df
        if start is not None:
            s = pd.Timestamp(start)
            if s.tz is None:
                s = s.tz_localize("UTC")
            else:
                s = s.tz_convert("UTC")
            out = out[out.index >= s]
        if end is not None:
            e = pd.Timestamp(end)
            if e.tz is None:
                e = e.tz_localize("UTC")
            else:
                e = e.tz_convert("UTC")
            out = out[out.index < e]
        return out.sort_index()

    @staticmethod
    def _resolve_range(
        *,
        start: Optional[Union[str, datetime]],
        end: Optional[Union[str, datetime]],
        period: Optional[str],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        def _to_utc(ts: Union[str, datetime]) -> pd.Timestamp:
            out = pd.Timestamp(ts)
            return cast(pd.Timestamp, out.tz_localize("UTC") if out.tz is None else out.tz_convert("UTC"))

        if start is None and end is None:
            if not period:
                raise ValueError("AlphaVantageRetriever requires start/end, period, or month='YYYY-MM'.")
            num = int(period[:-1])
            unit = period[-1]
            end_ts = pd.Timestamp.now(tz="UTC")
            if unit == "d":
                start_ts = end_ts - pd.Timedelta(days=num)
            elif unit == "w":
                start_ts = end_ts - pd.Timedelta(weeks=num)
            elif unit == "m":
                start_ts = end_ts - pd.DateOffset(months=num)
            elif unit == "y":
                start_ts = end_ts - pd.DateOffset(years=num)
            else:
                raise ValueError(f"Unsupported period format: {period}")
            return cast(pd.Timestamp, start_ts), cast(pd.Timestamp, end_ts)

        end_ts = _to_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
        start_ts = _to_utc(start) if start is not None else end_ts - pd.DateOffset(years=2)
        return cast(pd.Timestamp, start_ts), cast(pd.Timestamp, end_ts)


# ---------------------------------------------------------------------------
# Hybrid back-end
# ---------------------------------------------------------------------------

class HybridRetriever(DataRetriever):
    """Route recent/live data to Yahoo and deeper history to Alpha Vantage."""

    def __init__(
        self,
        *,
        yahoo: Optional[YahooRetriever] = None,
        alpha: Optional[AlphaVantageRetriever] = None,
        recent_days: int = 7,
    ) -> None:
        self.yahoo = yahoo or YahooRetriever(interval="1m", period="7d")
        self.alpha = alpha or AlphaVantageRetriever(interval="1m")
        self.recent_days = recent_days

    def get_history(
        self,
        symbol: str,
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        use_yahoo = self._prefer_yahoo(start=start, end=end, period=period, interval=interval)
        backend: DataRetriever = self.yahoo if use_yahoo else self.alpha
        return backend.get_history(
            symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
            **kwargs,
        )

    def get_ticker_data(
        self,
        symbols: List[str],
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.get_history(
                sym,
                start=start,
                end=end,
                period=period,
                interval=interval,
                **kwargs,
            )
            if not df.empty:
                out[sym] = df
        return out

    def get_current_price(self, symbol: str, **kwargs) -> dict:
        return self.yahoo.get_current_price(symbol, **kwargs)

    def _prefer_yahoo(
        self,
        *,
        start: Optional[Union[str, datetime]],
        end: Optional[Union[str, datetime]],
        period: Optional[str],
        interval: str,
    ) -> bool:
        if interval not in ("1m", "1min"):
            return True
        if period is not None:
            if period.endswith("d"):
                try:
                    return int(period[:-1]) <= self.recent_days
                except ValueError:
                    return False
            return False
        if start is None:
            return False
        start_ts = pd.Timestamp(start)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        return (now - start_ts) <= pd.Timedelta(days=self.recent_days)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DataRetriever",
    "YahooRetriever",
    "AlphaVantageRetriever",
    "HybridRetriever",
]
