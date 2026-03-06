"""
retriever.py
============
Unified data-retrieval interface for the kvant pipeline.

Back-ends are implemented as subclasses of :class:`DataRetriever`:

  • :class:`YahooRetriever`         – Yahoo Finance via yfinance.
                                       Best for recent / live data.
  • :class:`HuggingFaceRetriever`   – HuggingFace minute-bar dataset
                                       (mito0o852/OHLCV-1m). Stub – not
                                       active, but ready to extend.

Typical usage
-------------
    from kvant.kdata.retriever import YahooRetriever

    r = YahooRetriever()
    df   = r.get_history("AAPL", period="1mo", interval="1m")
    data = r.get_ticker_data(["AAPL", "MSFT"], period="1mo", interval="1m")
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
    """Abstract interface for all data-retrieval back-ends."""

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
        """Return ``{ticker: DataFrame}`` for each symbol in *symbols*."""

    def get_current_price(self, symbol: str, **kwargs) -> dict:
        """Return the most-recent price info for *symbol*.

        Default implementation fetches the last bar from :meth:`get_history`.
        Override for back-ends that expose a cheaper real-time endpoint.
        """
        df = self.get_history(symbol, period="1d", interval="1m", **kwargs)
        if df.empty:
            return {"symbol": symbol, "current_price": None}
        last = df.iloc[-1]
        return {
            "symbol": symbol,
            "current_price": float(last.get("Close", last.get("close", float("nan")))),
            "timestamp": df.index[-1].isoformat(),
            "source": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# Yahoo Finance back-end
# ---------------------------------------------------------------------------

class YahooRetriever(DataRetriever):
    """Retrieves data from Yahoo Finance via *yfinance*."""

    def __init__(self, prepost: bool = False):
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
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        from kvant.kdata.yahoo_retriever import get_history as _get_history
        df = _get_history(
            symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
            prepost=self.prepost,
            as_pandas=True,
            **kwargs,
        )
        if not df.empty:
            df.index = pd.to_datetime(df.index, utc=True)
        return df

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
            df = self.get_history(sym, start=start, end=end, period=period, interval=interval, **kwargs)
            if not df.empty:
                out[sym] = df
        return out

    # ------------------------------------------------------------------
    # Yahoo-specific extras
    # ------------------------------------------------------------------

    def get_current_price(self, symbol: str, **kwargs) -> dict:
        from kvant.kdata.yahoo_retriever import get_current_price as _get_current_price
        return _get_current_price(symbol, **kwargs)

    def get_price_at(
        self,
        symbol: str,
        when: Union[str, int, float, datetime],
        **kwargs,
    ) -> Optional[dict]:
        from kvant.kdata.yahoo_retriever import get_price_at as _get_price_at
        return _get_price_at(symbol, when, **kwargs)


# ---------------------------------------------------------------------------
# HuggingFace back-end  (stub – not active, ready to extend)
# ---------------------------------------------------------------------------

class HuggingFaceRetriever(DataRetriever):
    """Retrieves data from the HuggingFace minute-bar dataset.

    Not used in the active pipeline, but kept as an extension point.
    Call :meth:`get_splits` to obtain pre-built train/val/test splits.
    """

    def get_history(self, symbol: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(
            "HuggingFaceRetriever does not support single-ticker queries. "
            "Use get_splits() + get_ticker_data() instead."
        )

    def get_ticker_data(
        self,
        symbols: List[str],
        *,
        downloaded_dataset=None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        if downloaded_dataset is None:
            raise ValueError("'downloaded_dataset' is required for HuggingFaceRetriever.")
        from kvant.kdata.hf_minute_data import get_ticker_data as _hf_get_ticker_data
        return _hf_get_ticker_data(downloaded_dataset)

    def get_splits(
        self,
        *,
        n: int = 5,
        warmup_quarters: int = 16,
        blacklisted_tickers: Optional[tuple] = None,
        preset: Optional[str] = None,
    ) -> list:
        from kvant.kdata.hf_minute_data import (
            get_huggingface_top_n_tiny_splits,
            get_huggingface_top_5_small_splits,
            get_huggingface_top_200_splits,
        )
        if preset == "large":
            return get_huggingface_top_200_splits()
        if preset == "small":
            return get_huggingface_top_5_small_splits()
        return get_huggingface_top_n_tiny_splits(
            n=n,
            warmup_quarters=warmup_quarters,
            blacklisted_tickers=blacklisted_tickers,
        )


# ---------------------------------------------------------------------------
# Re-export HF types for convenience (callers that still need them)
# ---------------------------------------------------------------------------
from kvant.kdata.hf_minute_data import (
    DownloadedDatasetSplit,
    DatasetConfiguration,
    available_datasets,
    download_and_create_dataset,
)

__all__ = [
    # Classes
    "DataRetriever",
    "YahooRetriever",
    "HuggingFaceRetriever",
    # HF types
    "DownloadedDatasetSplit",
    "DatasetConfiguration",
    "available_datasets",
    "download_and_create_dataset",
]
