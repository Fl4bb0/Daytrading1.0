"""
sampling.count — Count-based bar samplers: Tick, Volume, Dollar.

Reference: López de Prado, "Advances in Financial Machine Learning", ch. 2.

Each sampler accumulates a running counter of a flow measure and closes a bar
when the counter exceeds a fixed threshold. The threshold is tuned per-ticker
from training data so that bars-per-day ≈ ``target_bars_per_day``.

  TunedTickBarSampler   — counter += 1 per row
  TunedVolumeBarSampler — counter += volume_t
  TunedDollarBarSampler — counter += close_t * volume_t
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from kvant.sampling.base import BarSampler
from kvant.sampling._utils import aggregate_ohlcv, flow_per_day, ticks_per_day
from kvant.utils.time_utils import ensure_utc_sorted_index


# ---------------------------------------------------------------------------
# Core algorithm (shared by all three)
# ---------------------------------------------------------------------------

def _count_bar_ends(counter: np.ndarray, threshold: float) -> np.ndarray:
    """Return 0-based row indices where count-based bars close."""
    if len(counter) == 0 or threshold <= 0.0:
        return np.array([], dtype=np.int64)
    ends = []
    running = 0.0
    for i, val in enumerate(counter):
        running += val
        if running >= threshold:
            ends.append(i)
            running = 0.0
    return np.asarray(ends, dtype=np.int64)


def _check_fitted(sampler_name: str, ticker: str, store: dict) -> None:
    if ticker not in store:
        raise KeyError(
            f"{sampler_name}: no tuned threshold for ticker={ticker!r}. "
            f"Call fit(ticker_dfs_train) first."
        )


# ---------------------------------------------------------------------------
# Tick Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedTickBarSampler(BarSampler):
    """
    Tick Bar sampler — a new bar closes every N ticks.

    N = ticks_per_day / target_bars_per_day (fitted per ticker from training data).
    """
    name: str = "tb_tuned"
    target_bars_per_day: float = 12.0
    price_col: str = "close"
    aggregate_ohlcv: bool = True
    _threshold_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "TunedTickBarSampler":
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns:
                continue
            tpd = ticks_per_day(df)
            if tpd > 0:
                self._threshold_by_ticker[ticker] = max(1.0, tpd / self.target_bars_per_day)
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name, "target_bars_per_day": float(self.target_bars_per_day),
                "price_col": self.price_col, "aggregate_ohlcv": bool(self.aggregate_ohlcv)}

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        return {"threshold": self._threshold_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        _check_fitted(self.__class__.__name__, ticker, self._threshold_by_ticker)
        ends = _count_bar_ends(np.ones(len(df), dtype=np.float64),
                                self._threshold_by_ticker[ticker])
        return aggregate_ohlcv(df, ends) if self.aggregate_ohlcv else df.iloc[ends].copy()


# ---------------------------------------------------------------------------
# Volume Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedVolumeBarSampler(BarSampler):
    """
    Volume Bar sampler — a new bar closes every V shares.

    V = avg_volume_per_day / target_bars_per_day (fitted per ticker).
    """
    name: str = "vb_tuned"
    target_bars_per_day: float = 12.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _threshold_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "TunedVolumeBarSampler":
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            volume = df[self.volume_col].to_numpy(dtype=np.float64)
            daily = flow_per_day(df, volume)
            if daily > 0:
                self._threshold_by_ticker[ticker] = daily / self.target_bars_per_day
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name, "target_bars_per_day": float(self.target_bars_per_day),
                "price_col": self.price_col, "volume_col": self.volume_col,
                "aggregate_ohlcv": bool(self.aggregate_ohlcv)}

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        return {"threshold": self._threshold_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        _check_fitted(self.__class__.__name__, ticker, self._threshold_by_ticker)
        volume = df[self.volume_col].to_numpy(dtype=np.float64)
        ends = _count_bar_ends(volume, self._threshold_by_ticker[ticker])
        return aggregate_ohlcv(df, ends) if self.aggregate_ohlcv else df.iloc[ends].copy()


# ---------------------------------------------------------------------------
# Dollar Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedDollarBarSampler(BarSampler):
    """
    Dollar Bar sampler — a new bar closes every D dollars traded.

    D = avg_dollar_flow_per_day / target_bars_per_day (fitted per ticker).
    dollar_flow_t = close_t * volume_t
    """
    name: str = "db_tuned"
    target_bars_per_day: float = 12.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _threshold_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "TunedDollarBarSampler":
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            close = df[self.price_col].to_numpy(dtype=np.float64)
            volume = df[self.volume_col].to_numpy(dtype=np.float64)
            daily = flow_per_day(df, close * volume)
            if daily > 0:
                self._threshold_by_ticker[ticker] = daily / self.target_bars_per_day
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name, "target_bars_per_day": float(self.target_bars_per_day),
                "price_col": self.price_col, "volume_col": self.volume_col,
                "aggregate_ohlcv": bool(self.aggregate_ohlcv)}

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        return {"threshold": self._threshold_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        _check_fitted(self.__class__.__name__, ticker, self._threshold_by_ticker)
        close = df[self.price_col].to_numpy(dtype=np.float64)
        volume = df[self.volume_col].to_numpy(dtype=np.float64)
        ends = _count_bar_ends(close * volume, self._threshold_by_ticker[ticker])
        return aggregate_ohlcv(df, ends) if self.aggregate_ohlcv else df.iloc[ends].copy()
