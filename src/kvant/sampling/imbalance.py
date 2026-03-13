"""
sampling.imbalance — Imbalance bar samplers: TIB, VIB, DIB.

Reference: López de Prado, "Advances in Financial Machine Learning", ch. 2.

Imbalance bars close when the signed order-flow imbalance exceeds its
expected value. Unlike count-based bars, they adapt to changing market
regimes via EWMA updates to E[T] and E[flow] after every bar closes.

  TunedTIBSampler — Tick Imbalance Bars    (flow = b_t)
  TunedVIBSampler — Volume Imbalance Bars  (flow = b_t * volume_t)
  TunedDIBSampler — Dollar Imbalance Bars  (flow = b_t * close_t * volume_t)

where b_t is the tick direction (+1 / -1) from the tick rule.

Bar formation rule (all three):
  θ_T = Σ flow_t
  Close when |θ_T| ≥ E[T] * |E[flow_t]|

Both expectations update via EWMA after each bar, controlled by ewma_span_multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from kvant.sampling.base import BarSampler
from kvant.sampling._utils import aggregate_ohlcv, ticks_per_day, tick_directions
from kvant.utils.time_utils import ensure_utc_sorted_index


# ---------------------------------------------------------------------------
# Core algorithm (shared by TIB, VIB, DIB)
# ---------------------------------------------------------------------------

def _imbalance_bar_ends(
    signed_flow: np.ndarray,
    E_T_init: float,
    E_flow_init: float,
    ewma_span: float,
) -> np.ndarray:
    """
    Return 0-based row indices where imbalance bars close.

    Parameters
    ----------
    signed_flow  : per-row flow (b_t, b_t*v_t, or b_t*p_t*v_t).
    E_T_init     : initial expected bar length (ticks).
    E_flow_init  : initial expected signed flow per tick.
    ewma_span    : EWMA span for updating E_T and E_flow after each bar.
    """
    n = len(signed_flow)
    if n == 0:
        return np.array([], dtype=np.int64)

    alpha = 2.0 / (max(ewma_span, 1.0) + 1.0)
    E_T    = max(1.0, float(E_T_init))
    E_flow = float(E_flow_init)

    ends, theta, bar_start = [], 0.0, 0
    for i, sf in enumerate(signed_flow):
        theta += sf
        threshold = E_T * abs(E_flow)
        if threshold < 1e-12:
            threshold = E_T          # degenerate guard: fall back to bar-length

        if abs(theta) >= threshold:
            ends.append(i)
            bar_len = i - bar_start + 1
            E_T    = (1.0 - alpha) * E_T    + alpha * bar_len
            E_flow = (1.0 - alpha) * E_flow + alpha * (theta / bar_len)
            theta, bar_start = 0.0, i + 1

    return np.asarray(ends, dtype=np.int64)


def _check_fitted(name: str, ticker: str, store: dict) -> None:
    if ticker not in store:
        raise KeyError(f"{name}: no tuned parameters for ticker={ticker!r}. "
                       f"Call fit(ticker_dfs_train) first.")


# ---------------------------------------------------------------------------
# TIB — Tick Imbalance Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedTIBSampler(BarSampler):
    """
    Tick Imbalance Bar sampler.

    Bars close when |Σ b_t| ≥ E[T] * |E[b_t]|. Both expectations adapt
    via EWMA after every bar.

    Parameters
    ----------
    target_bars_per_day  : float — sets initial E[T] = ticks_per_day / target.
    ewma_span_multiplier : float — ewma_span = E_T_init * multiplier.
    """
    name: str = "tib_tuned"
    target_bars_per_day: float = 12.0
    ewma_span_multiplier: float = 1.0
    price_col: str = "close"
    aggregate_ohlcv: bool = True
    _E_T_by_ticker: Dict[str, float] = field(default_factory=dict)
    _E_flow_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "TunedTIBSampler":
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 10 or self.price_col not in df.columns:
                continue
            df = ensure_utc_sorted_index(df)
            tpd = ticks_per_day(df)
            if tpd <= 0:
                continue
            b = tick_directions(df[self.price_col].to_numpy(dtype=np.float64))
            self._E_T_by_ticker[ticker]    = tpd / self.target_bars_per_day
            self._E_flow_by_ticker[ticker] = float(np.mean(b))
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name, "target_bars_per_day": float(self.target_bars_per_day),
                "ewma_span_multiplier": float(self.ewma_span_multiplier),
                "price_col": self.price_col, "aggregate_ohlcv": bool(self.aggregate_ohlcv)}

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        return {"E_T_init": self._E_T_by_ticker.get(ticker),
                "E_flow_init": self._E_flow_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        _check_fitted(self.__class__.__name__, ticker, self._E_T_by_ticker)
        b = tick_directions(df[self.price_col].to_numpy(dtype=np.float64))
        E_T = self._E_T_by_ticker[ticker]
        ends = _imbalance_bar_ends(b, E_T, self._E_flow_by_ticker[ticker],
                                   ewma_span=E_T * self.ewma_span_multiplier)
        return aggregate_ohlcv(df, ends) if self.aggregate_ohlcv else df.iloc[ends].copy()


# ---------------------------------------------------------------------------
# VIB — Volume Imbalance Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedVIBSampler(BarSampler):
    """
    Volume Imbalance Bar sampler.

    Bars close when |Σ b_t * v_t| ≥ E[T] * |E[b_t * v_t]|.
    """
    name: str = "vib_tuned"
    target_bars_per_day: float = 12.0
    ewma_span_multiplier: float = 1.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _E_T_by_ticker: Dict[str, float] = field(default_factory=dict)
    _E_flow_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "TunedVIBSampler":
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            tpd = ticks_per_day(df)
            if tpd <= 0:
                continue
            b = tick_directions(df[self.price_col].to_numpy(dtype=np.float64))
            v = df[self.volume_col].to_numpy(dtype=np.float64)
            self._E_T_by_ticker[ticker]    = tpd / self.target_bars_per_day
            self._E_flow_by_ticker[ticker] = float(np.mean(b * v))
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name, "target_bars_per_day": float(self.target_bars_per_day),
                "ewma_span_multiplier": float(self.ewma_span_multiplier),
                "price_col": self.price_col, "volume_col": self.volume_col,
                "aggregate_ohlcv": bool(self.aggregate_ohlcv)}

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        return {"E_T_init": self._E_T_by_ticker.get(ticker),
                "E_flow_init": self._E_flow_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        _check_fitted(self.__class__.__name__, ticker, self._E_T_by_ticker)
        b = tick_directions(df[self.price_col].to_numpy(dtype=np.float64))
        v = df[self.volume_col].to_numpy(dtype=np.float64)
        E_T = self._E_T_by_ticker[ticker]
        ends = _imbalance_bar_ends(b * v, E_T, self._E_flow_by_ticker[ticker],
                                   ewma_span=E_T * self.ewma_span_multiplier)
        return aggregate_ohlcv(df, ends) if self.aggregate_ohlcv else df.iloc[ends].copy()


# ---------------------------------------------------------------------------
# DIB — Dollar Imbalance Bars
# ---------------------------------------------------------------------------

@dataclass
class TunedDIBSampler(BarSampler):
    """
    Dollar Imbalance Bar sampler.

    Bars close when |Σ b_t * p_t * v_t| ≥ E[T] * |E[b_t * p_t * v_t]|.
    The most theoretically grounded variant: normalises for price level and
    volume, making it suitable for cross-ticker comparison.
    """
    name: str = "dib_tuned"
    target_bars_per_day: float = 12.0
    ewma_span_multiplier: float = 1.0
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    _E_T_by_ticker: Dict[str, float] = field(default_factory=dict)
    _E_flow_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "TunedDIBSampler":
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue
            tpd = ticks_per_day(df)
            if tpd <= 0:
                continue
            close = df[self.price_col].to_numpy(dtype=np.float64)
            v     = df[self.volume_col].to_numpy(dtype=np.float64)
            b     = tick_directions(close)
            self._E_T_by_ticker[ticker]    = tpd / self.target_bars_per_day
            self._E_flow_by_ticker[ticker] = float(np.mean(b * close * v))
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name, "target_bars_per_day": float(self.target_bars_per_day),
                "ewma_span_multiplier": float(self.ewma_span_multiplier),
                "price_col": self.price_col, "volume_col": self.volume_col,
                "aggregate_ohlcv": bool(self.aggregate_ohlcv)}

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        return {"E_T_init": self._E_T_by_ticker.get(ticker),
                "E_flow_init": self._E_flow_by_ticker.get(ticker)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        _check_fitted(self.__class__.__name__, ticker, self._E_T_by_ticker)
        close = df[self.price_col].to_numpy(dtype=np.float64)
        v     = df[self.volume_col].to_numpy(dtype=np.float64)
        b     = tick_directions(close)
        E_T   = self._E_T_by_ticker[ticker]
        ends  = _imbalance_bar_ends(b * close * v, E_T, self._E_flow_by_ticker[ticker],
                                    ewma_span=E_T * self.ewma_span_multiplier)
        return aggregate_ohlcv(df, ends) if self.aggregate_ohlcv else df.iloc[ends].copy()
