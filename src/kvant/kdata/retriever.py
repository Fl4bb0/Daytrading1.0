"""
kdata.retriever — Unified data-retrieval interface.

Back-ends are subclasses of :class:`DataRetriever`. Only Yahoo Finance
is active; the pattern shows exactly what to implement to add a new one.

Active back-ends
----------------
  YahooRetriever  — Yahoo Finance via yfinance (raw helpers in yahoo_retriever.py).

Adding a new back-end
---------------------
  1. Create ``kdata/<name>_retriever.py`` with the raw data-fetch logic.
  2. Subclass ``DataRetriever`` and implement ``get_history`` + ``get_ticker_data``.
  3. Optionally override ``get_current_price`` for a cheaper real-time endpoint.

Typical usage
-------------
    from kvant.kdata.retriever import YahooRetriever

    r = YahooRetriever(interval="1d", period="6mo")
    df   = r.get_history("AAPL")
    data = r.get_ticker_data(["AAPL", "MSFT"])
    cur  = r.get_current_price("AAPL")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd


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

        # Lower-case columns; drop yfinance multi-level artefacts
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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DataRetriever",
    "YahooRetriever",
]
