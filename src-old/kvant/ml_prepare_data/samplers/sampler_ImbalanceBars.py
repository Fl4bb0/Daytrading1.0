"""
Imbalance Bar Samplers (TIB / VIB / DIB)
=========================================
Imbalance bars (López de Prado, "Advances in Financial Machine Learning", ch. 2)
close when the *signed imbalance* of a flow measure exceeds its expected value.
Unlike count-based bars (tick/volume/dollar bars), imbalance bars are
event-driven by *order-flow direction*, which makes them more sensitive to
informed trading activity.

Three variants are provided, differing only in what flow they accumulate:

  TIB — Tick Imbalance Bars
    flow_t = b_t                  (b_t = tick direction: +1 up, -1 down)

  VIB — Volume Imbalance Bars
    flow_t = b_t * volume_t       (signed share volume)

  DIB — Dollar Imbalance Bars
    flow_t = b_t * close_t * volume_t   (signed dollar volume)

Bar formation rule (all three):
  θ_T = Σ_{t=bar_start}^{T} flow_t
  Close bar when |θ_T| >= E[T] * |E[flow_t]|

  where E[T] and E[flow_t] are updated after each bar via EWMA, giving the
  sampler the ability to adapt to changing market regimes.

Tick directions are computed with the *tick rule*:
  b_t = sign(Δclose_t)  if Δclose_t ≠ 0
  b_t = b_{t-1}         otherwise  (carry forward)
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

def _tick_directions(close: np.ndarray) -> np.ndarray:
    """
    Apply the tick rule to a close-price array.

    Returns an array of +1.0 / -1.0 values (never 0).
    The first tick is arbitrarily set to +1.
    """
    n = len(close)
    b = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            b[i] = 1.0
        elif diff < 0.0:
            b[i] = -1.0
        else:
            b[i] = b[i - 1]  # tick rule: carry the last known direction
    return b


def _ticks_per_day(df: pd.DataFrame) -> float:
    """Average number of rows per calendar trading day."""
    if len(df) == 0:
        return 0.0
    idx = ensure_utc_sorted_index(df).index
    n_days = int(idx.normalize().nunique())
    return len(df) / n_days if n_days > 0 else 0.0


def _imbalance_bar_ends(
    signed_flow: np.ndarray,
    E_T_init: float,
    E_flow_init: float,
    ewma_span: float,
) -> np.ndarray:
    """
    Core imbalance bar algorithm, shared by TIB, VIB, and DIB.

    Parameters
    ----------
    signed_flow : array
        Per-row flow: b_t (TIB), b_t*v_t (VIB), or b_t*p_t*v_t (DIB).
    E_T_init : float
        Initial estimate of expected bar length (ticks).
    E_flow_init : float
        Initial estimate of expected signed flow per tick.
    ewma_span : float
        EWMA span used to update E_T and E_flow after each bar closes.
        Using E_T_init here gives roughly one bar of memory per update.

    Returns
    -------
    np.ndarray of 0-based row indices at which bars close.
    """
    n = len(signed_flow)
    if n == 0:
        return np.array([], dtype=np.int64)

    alpha = 2.0 / (max(ewma_span, 1.0) + 1.0)
    E_T = max(1.0, float(E_T_init))
    E_flow = float(E_flow_init)

    ends = []
    theta = 0.0
    bar_start = 0

    for i, sf in enumerate(signed_flow):
        theta += sf
        threshold = E_T * abs(E_flow)

        # Degenerate guard: if expected flow is ~0, fall back to bar-length only
        if threshold < 1e-12:
            threshold = E_T

        if abs(theta) >= threshold:
            ends.append(i)
            bar_len = i - bar_start + 1
            # Update expectations via EWMA
            E_T = (1.0 - alpha) * E_T + alpha * bar_len
            E_flow = (1.0 - alpha) * E_flow + alpha * (theta / bar_len)
            theta = 0.0
            bar_start = i + 1

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
# TIB — Tick Imbalance Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedTIBSampler(BaseBarSampler):
    """
    Tick Imbalance Bar sampler.

    Bars close when the cumulative signed tick count |Σ b_t| exceeds
    E[T] * |E[b_t]|. Both expectations are updated via EWMA after each bar.

    Parameters
    ----------
    target_bars_per_day : float
        Used to set the initial E[T] = ticks_per_day / target_bars_per_day.
    ewma_span_multiplier : float
        ewma_span = E_T_init * ewma_span_multiplier. Default 1.0 means one
        bar's worth of history drives each update.
    price_col : str
        Column used to compute tick directions.
    aggregate_ohlcv : bool
        If True, each bar is aggregated to OHLCV. If False, only the closing
        row is returned.
    """
    name: str = "tib_tuned"
    target_bars_per_day: float = 12.0
    ewma_span_multiplier: float = 1.0
    price_col: str = "close"
    aggregate_ohlcv: bool = True
    # Stored after fit()
    _E_T_by_ticker: Dict[str, float] = field(default_factory=dict)
    _E_flow_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedTIBSampler":
        for ticker, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns:
                continue
            tpd = _ticks_per_day(df)
            if tpd <= 0:
                continue
            close = df[self.price_col].to_numpy(dtype=np.float64)
            b = _tick_directions(close)
            self._E_T_by_ticker[ticker] = tpd / self.target_bars_per_day
            self._E_flow_by_ticker[ticker] = float(np.mean(b))
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "ewma_span_multiplier": float(self.ewma_span_multiplier),
            "price_col": self.price_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {
            "E_T_init": self._E_T_by_ticker.get(ticker),
            "E_flow_init": self._E_flow_by_ticker.get(ticker),
        }

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        if ticker not in self._E_T_by_ticker:
            raise KeyError(
                f"TunedTIBSampler has no tuned parameters for ticker={ticker}. "
                f"Call sampler.fit(ticker_dfs_train) first."
            )
        close = df[self.price_col].to_numpy(dtype=np.float64)
        b = _tick_directions(close)
        E_T = self._E_T_by_ticker[ticker]
        ends = _imbalance_bar_ends(
            b, E_T, self._E_flow_by_ticker[ticker],
            ewma_span=E_T * self.ewma_span_multiplier,
        )
        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()
        return _aggregate_ohlcv_segments(df, ends)


# ---------------------------------------------------------------------------
# VIB — Volume Imbalance Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedVIBSampler(BaseBarSampler):
    """
    Volume Imbalance Bar sampler.

    Bars close when the cumulative signed volume |Σ b_t * v_t| exceeds
    E[T] * |E[b_t * v_t]|.

    Parameters
    ----------
    target_bars_per_day : float
        Used to set the initial E[T].
    ewma_span_multiplier : float
        ewma_span = E_T_init * ewma_span_multiplier.
    price_col / volume_col : str
        Columns for close price and share volume.
    aggregate_ohlcv : bool
        If True, OHLCV aggregation. If False, closing row only.
    """
    name: str = "vib_tuned"
    target_bars_per_day: float = 12.0
    ewma_span_multiplier: float = 1.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _E_T_by_ticker: Dict[str, float] = field(default_factory=dict)
    _E_flow_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedVIBSampler":
        for ticker, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            tpd = _ticks_per_day(df)
            if tpd <= 0:
                continue
            close = df[self.price_col].to_numpy(dtype=np.float64)
            volume = df[self.volume_col].to_numpy(dtype=np.float64)
            b = _tick_directions(close)
            self._E_T_by_ticker[ticker] = tpd / self.target_bars_per_day
            self._E_flow_by_ticker[ticker] = float(np.mean(b * volume))
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "ewma_span_multiplier": float(self.ewma_span_multiplier),
            "price_col": self.price_col,
            "volume_col": self.volume_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {
            "E_T_init": self._E_T_by_ticker.get(ticker),
            "E_flow_init": self._E_flow_by_ticker.get(ticker),
        }

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        if ticker not in self._E_T_by_ticker:
            raise KeyError(
                f"TunedVIBSampler has no tuned parameters for ticker={ticker}. "
                f"Call sampler.fit(ticker_dfs_train) first."
            )
        close = df[self.price_col].to_numpy(dtype=np.float64)
        volume = df[self.volume_col].to_numpy(dtype=np.float64)
        b = _tick_directions(close)
        E_T = self._E_T_by_ticker[ticker]
        ends = _imbalance_bar_ends(
            b * volume, E_T, self._E_flow_by_ticker[ticker],
            ewma_span=E_T * self.ewma_span_multiplier,
        )
        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()
        return _aggregate_ohlcv_segments(df, ends)


# ---------------------------------------------------------------------------
# DIB — Dollar Imbalance Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedDIBSampler(BaseBarSampler):
    """
    Dollar Imbalance Bar sampler.

    Bars close when the cumulative signed dollar flow |Σ b_t * p_t * v_t|
    exceeds E[T] * |E[b_t * p_t * v_t]|.

    This is the most theoretically motivated variant: it normalises for both
    price level and volume, making it suitable for cross-ticker comparison.

    Parameters
    ----------
    target_bars_per_day : float
        Used to set the initial E[T].
    ewma_span_multiplier : float
        ewma_span = E_T_init * ewma_span_multiplier.
    price_col / volume_col : str
        Columns for close price and share volume.
    aggregate_ohlcv : bool
        If True, OHLCV aggregation. If False, closing row only.
    """
    name: str = "dib_tuned"
    target_bars_per_day: float = 12.0
    ewma_span_multiplier: float = 1.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _E_T_by_ticker: Dict[str, float] = field(default_factory=dict)
    _E_flow_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedDIBSampler":
        for ticker, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            tpd = _ticks_per_day(df)
            if tpd <= 0:
                continue
            close = df[self.price_col].to_numpy(dtype=np.float64)
            volume = df[self.volume_col].to_numpy(dtype=np.float64)
            b = _tick_directions(close)
            self._E_T_by_ticker[ticker] = tpd / self.target_bars_per_day
            self._E_flow_by_ticker[ticker] = float(np.mean(b * close * volume))
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "ewma_span_multiplier": float(self.ewma_span_multiplier),
            "price_col": self.price_col,
            "volume_col": self.volume_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {
            "E_T_init": self._E_T_by_ticker.get(ticker),
            "E_flow_init": self._E_flow_by_ticker.get(ticker),
        }

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        if ticker not in self._E_T_by_ticker:
            raise KeyError(
                f"TunedDIBSampler has no tuned parameters for ticker={ticker}. "
                f"Call sampler.fit(ticker_dfs_train) first."
            )
        close = df[self.price_col].to_numpy(dtype=np.float64)
        volume = df[self.volume_col].to_numpy(dtype=np.float64)
        b = _tick_directions(close)
        E_T = self._E_T_by_ticker[ticker]
        ends = _imbalance_bar_ends(
            b * close * volume, E_T, self._E_flow_by_ticker[ticker],
            ewma_span=E_T * self.ewma_span_multiplier,
        )
        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()
        return _aggregate_ohlcv_segments(df, ends)