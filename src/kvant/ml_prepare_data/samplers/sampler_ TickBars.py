"""
Count-Based Bar Samplers (Tick / Volume / Dollar Bars)
======================================================
Count-based bars (López de Prado, "Advances in Financial Machine Learning", ch. 2)
close when a running counter of a flow measure exceeds a fixed threshold.

Three variants are provided:

  TB — Tick Bars
    counter += 1 per row; close every N ticks.

  VB — Volume Bars
    counter += volume_t; close every V shares.

  DB — Dollar Bars
    counter += close_t * volume_t; close every D dollars.

Each sampler has a "Tuned" variant that estimates the threshold from
training data using ``target_bars_per_day``:

  threshold = total_flow_per_day / target_bars_per_day

Unlike imbalance bars the threshold is *fixed* after fitting (no EWMA
adaptation), which makes count-based bars simple and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index
from kvant.ml_prepare_data.samplers.sampling import BaseBarSampler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flow_per_day(df: pd.DataFrame, flow: np.ndarray) -> float:
    """Average total flow per calendar trading day."""
    idx = ensure_utc_sorted_index(df).index
    n_days = int(idx.normalize().nunique())
    if n_days == 0:
        return 0.0
    return float(flow.sum()) / n_days


def _count_bar_ends(counter: np.ndarray, threshold: float) -> np.ndarray:
    """
    Core count-bar algorithm shared by TB, VB, and DB.

    Parameters
    ----------
    counter : array
        Per-row flow increments: 1 (TB), volume (VB), or price*volume (DB).
    threshold : float
        Bar closes when the running total >= threshold.

    Returns
    -------
    np.ndarray of 0-based row indices at which bars close.
    """
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


def _aggregate_ohlcv_segments(df: pd.DataFrame, ends: np.ndarray) -> pd.DataFrame:
    """Aggregate raw rows into OHLCV bars at the given end-row indices."""
    if len(df) == 0:
        return df.copy()

    if len(ends) == 0:
        seg = df.iloc[:]
        bar: dict = {}
        if "open"   in seg: bar["open"]   = float(seg["open"].iloc[0])
        if "high"   in seg: bar["high"]   = float(seg["high"].max())
        if "low"    in seg: bar["low"]    = float(seg["low"].min())
        if "close"  in seg: bar["close"]  = float(seg["close"].iloc[-1])
        if "volume" in seg: bar["volume"] = float(seg["volume"].sum())
        return pd.DataFrame([bar], index=pd.DatetimeIndex([df.index[-1]]))

    ends = np.unique(np.clip(ends, 0, len(df) - 1))
    rows, idx = [], []
    start = 0
    for end in ends:
        if end < start:
            continue
        seg = df.iloc[start : end + 1]
        bar = {}
        if "open"   in seg: bar["open"]   = float(seg["open"].iloc[0])
        if "high"   in seg: bar["high"]   = float(seg["high"].max())
        if "low"    in seg: bar["low"]    = float(seg["low"].min())
        if "close"  in seg: bar["close"]  = float(seg["close"].iloc[-1])
        if "volume" in seg: bar["volume"] = float(seg["volume"].sum())
        rows.append(bar)
        idx.append(df.index[end])
        start = end + 1

    return ensure_utc_sorted_index(pd.DataFrame(rows, index=pd.DatetimeIndex(idx)))


# ---------------------------------------------------------------------------
# TB — Tick Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedTickBarSampler(BaseBarSampler):
    """
    Tick Bar sampler.

    A new bar closes every N ticks, where N is chosen so that the expected
    number of bars per trading day matches ``target_bars_per_day``.

    Parameters
    ----------
    target_bars_per_day : float
        Desired bar frequency; sets N = ticks_per_day / target_bars_per_day.
    price_col : str
        Column used only to guard against missing data during fit.
    aggregate_ohlcv : bool
        If True, each bar is aggregated to OHLCV. If False, only the
        closing row is returned.
    """
    name: str = "tb_tuned"
    target_bars_per_day: float = 12.0
    price_col: str = "close"
    aggregate_ohlcv: bool = True
    # Stored after fit()
    _threshold_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedTickBarSampler":
        for ticker, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns:
                continue
            idx = df.index
            n_days = int(idx.normalize().nunique())
            if n_days == 0:
                continue
            ticks_per_day = len(df) / n_days
            self._threshold_by_ticker[ticker] = max(1.0, ticks_per_day / self.target_bars_per_day)
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "price_col": self.price_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {"threshold": self._threshold_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        if ticker not in self._threshold_by_ticker:
            raise KeyError(
                f"TunedTickBarSampler has no tuned parameters for ticker={ticker}. "
                f"Call sampler.fit(ticker_dfs_train) first."
            )
        counter = np.ones(len(df), dtype=np.float64)
        ends = _count_bar_ends(counter, self._threshold_by_ticker[ticker])
        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()
        return _aggregate_ohlcv_segments(df, ends)


# ---------------------------------------------------------------------------
# VB — Volume Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedVolumeBarSampler(BaseBarSampler):
    """
    Volume Bar sampler.

    A new bar closes every V shares, where V is chosen so that the expected
    number of bars per trading day matches ``target_bars_per_day``.

    Parameters
    ----------
    target_bars_per_day : float
        Desired bar frequency; sets V = avg_volume_per_day / target_bars_per_day.
    price_col / volume_col : str
        Columns for close price and share volume.
    aggregate_ohlcv : bool
        If True, OHLCV aggregation. If False, closing row only.
    """
    name: str = "vb_tuned"
    target_bars_per_day: float = 12.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _threshold_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedVolumeBarSampler":
        for ticker, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            volume = df[self.volume_col].to_numpy(dtype=np.float64)
            daily_flow = _flow_per_day(df, volume)
            if daily_flow <= 0:
                continue
            self._threshold_by_ticker[ticker] = daily_flow / self.target_bars_per_day
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "price_col": self.price_col,
            "volume_col": self.volume_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {"threshold": self._threshold_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        if ticker not in self._threshold_by_ticker:
            raise KeyError(
                f"TunedVolumeBarSampler has no tuned parameters for ticker={ticker}. "
                f"Call sampler.fit(ticker_dfs_train) first."
            )
        volume = df[self.volume_col].to_numpy(dtype=np.float64)
        ends = _count_bar_ends(volume, self._threshold_by_ticker[ticker])
        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()
        return _aggregate_ohlcv_segments(df, ends)


# ---------------------------------------------------------------------------
# DB — Dollar Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedDollarBarSampler(BaseBarSampler):
    """
    Dollar Bar sampler.

    A new bar closes every D dollars traded, where D is chosen so that the
    expected number of bars per trading day matches ``target_bars_per_day``.

      dollar_flow_t = close_t * volume_t

    Parameters
    ----------
    target_bars_per_day : float
        Desired bar frequency; sets D = avg_dollar_flow_per_day / target_bars_per_day.
    price_col / volume_col : str
        Columns for close price and share volume.
    aggregate_ohlcv : bool
        If True, OHLCV aggregation. If False, closing row only.
    """
    name: str = "db_tuned"
    target_bars_per_day: float = 12.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _threshold_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedDollarBarSampler":
        for ticker, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            close = df[self.price_col].to_numpy(dtype=np.float64)
            volume = df[self.volume_col].to_numpy(dtype=np.float64)
            dollar_flow = close * volume
            daily_flow = _flow_per_day(df, dollar_flow)
            if daily_flow <= 0:
                continue
            self._threshold_by_ticker[ticker] = daily_flow / self.target_bars_per_day
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "price_col": self.price_col,
            "volume_col": self.volume_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {"threshold": self._threshold_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        if ticker not in self._threshold_by_ticker:
            raise KeyError(
                f"TunedDollarBarSampler has no tuned parameters for ticker={ticker}. "
                f"Call sampler.fit(ticker_dfs_train) first."
            )
        close = df[self.price_col].to_numpy(dtype=np.float64)
        volume = df[self.volume_col].to_numpy(dtype=np.float64)
        ends = _count_bar_ends(close * volume, self._threshold_by_ticker[ticker])
        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()
        return _aggregate_ohlcv_segments(df, ends)