"""
sampling.cusum — CUSUM event-driven bar sampler.

Reference: López de Prado, "Advances in Financial Machine Learning", ch. 2.

The symmetric CUSUM filter fires a bar-end event whenever the cumulative
positive or negative log-return exceeds a threshold h. h is tuned per-ticker
from training data so that the expected bars-per-day matches ``target_bars_per_day``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from kvant.sampling.base import BarSampler
from kvant.sampling._utils import aggregate_ohlcv
from kvant.utils.time_utils import ensure_utc_sorted_index


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _cusum_bar_ends(close: np.ndarray, h: float) -> np.ndarray:
    """Return 0-based row indices where CUSUM bars close."""
    if len(close) < 2:
        return np.array([], dtype=np.int64)
    r = close[1:] / close[:-1] - 1.0
    s_pos = s_neg = 0.0
    ends = []
    for i, ri in enumerate(r, start=1):
        s_pos = max(0.0, s_pos + float(ri))
        s_neg = min(0.0, s_neg + float(ri))
        if s_pos > h or s_neg < -h:
            ends.append(i)
            s_pos = s_neg = 0.0
    return np.asarray(ends, dtype=np.int64)


def _bars_per_day(df: pd.DataFrame, ends: np.ndarray) -> float:
    if len(df) == 0 or len(ends) == 0:
        return 0.0
    idx = ensure_utc_sorted_index(df).index
    n_days = int(idx.normalize().nunique())
    if n_days <= 0:
        return 0.0
    return float(pd.Series(idx[ends]).dt.normalize().value_counts().sum()) / n_days


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class TunedCUSUMBarSampler(BarSampler):
    """
    Per-ticker tuned CUSUM bar sampler.

    Searches ``h_grid`` for the threshold that produces a bars-per-day count
    closest to ``target_bars_per_day`` on the training set.

    Parameters
    ----------
    target_bars_per_day : float
        Desired bar density.
    h_grid : tuple of float
        Grid of candidate thresholds searched during fit.
    price_col : str
        Column used as the close price for the CUSUM filter.
    aggregate_ohlcv : bool
        If True, each bar is aggregated to OHLCV. If False, only the
        closing row is returned.
    """
    name: str = "cusum_tuned"
    target_bars_per_day: float = 12.0
    h_grid: Tuple[float, ...] = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02,
                                  0.025, 0.03, 0.04, 0.05)
    price_col: str = "close"
    aggregate_ohlcv: bool = True
    tuned_h_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "TunedCUSUMBarSampler":
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns:
                continue
            close = df[self.price_col].to_numpy(dtype=np.float64)
            best_h, best_err = None, None
            for h in self.h_grid:
                ends = _cusum_bar_ends(close, float(h))
                err = abs(_bars_per_day(df, ends) - self.target_bars_per_day)
                if best_err is None or err < best_err or (err == best_err and (best_h is None or h > best_h)):
                    best_err, best_h = err, float(h)
            if best_h is not None:
                self.tuned_h_by_ticker[ticker] = best_h
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "h_grid": list(self.h_grid),
            "price_col": self.price_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        h = self.tuned_h_by_ticker.get(ticker)
        return {"h": None if h is None else float(h)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()
        if ticker not in self.tuned_h_by_ticker:
            raise KeyError(
                f"TunedCUSUMBarSampler: no tuned h for ticker={ticker!r}. "
                f"Call fit(ticker_dfs_train) first."
            )
        close = df[self.price_col].to_numpy(dtype=np.float64)
        ends = _cusum_bar_ends(close, self.tuned_h_by_ticker[ticker])
        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()
        return aggregate_ohlcv(df, ends)
