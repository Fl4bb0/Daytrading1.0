from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from src.kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index
from kvant.ml_prepare_data.samplers.sampling import BaseBarSampler

def _cusum_event_ends(close: np.ndarray, h: float) -> np.ndarray:
    if len(close) < 2:
        return np.array([], dtype=np.int64)

    r = close[1:] / close[:-1] - 1.0
    s_pos = 0.0
    s_neg = 0.0
    ends = []

    for i, ri in enumerate(r, start=1):
        s_pos = max(0.0, s_pos + float(ri))
        s_neg = min(0.0, s_neg + float(ri))
        if (s_pos > h) or (s_neg < -h):
            ends.append(i)
            s_pos = 0.0
            s_neg = 0.0

    return np.asarray(ends, dtype=np.int64)


def _aggregate_ohlcv_segments(df: pd.DataFrame, ends: np.ndarray) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    if len(ends) == 0:
        # policy: single bar over entire range
        seg = df.iloc[:]
        bar = {}
        if "open" in seg: bar["open"] = float(seg["open"].iloc[0])
        if "high" in seg: bar["high"] = float(seg["high"].max())
        if "low" in seg: bar["low"] = float(seg["low"].min())
        if "close" in seg: bar["close"] = float(seg["close"].iloc[-1])
        if "volume" in seg: bar["volume"] = float(seg["volume"].sum())
        return pd.DataFrame([bar], index=pd.DatetimeIndex([df.index[-1]]))

    ends = np.unique(np.clip(ends, 0, len(df) - 1))

    rows = []
    idx = []
    start = 0
    for end in ends:
        if end < start:
            continue
        seg = df.iloc[start : end + 1]
        bar = {}
        if "open" in seg: bar["open"] = float(seg["open"].iloc[0])
        if "high" in seg: bar["high"] = float(seg["high"].max())
        if "low" in seg: bar["low"] = float(seg["low"].min())
        if "close" in seg: bar["close"] = float(seg["close"].iloc[-1])
        if "volume" in seg: bar["volume"] = float(seg["volume"].sum())
        rows.append(bar)
        idx.append(df.index[end])
        start = end + 1

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    return ensure_utc_sorted_index(out)


def _bars_per_day(df: pd.DataFrame, ends: np.ndarray) -> float:
    if len(df) == 0 or len(ends) == 0:
        return 0.0
    idx = ensure_utc_sorted_index(df).index
    days = idx.normalize()
    n_days = int(days.nunique())
    if n_days <= 0:
        return 0.0
    end_days = days[ends]
    return float(pd.Series(end_days).value_counts().sum() / n_days)


@dataclass
class TunedCUSUMBarSampler(BaseBarSampler):
    """
    Per-ticker tuned CUSUM sampler.
    Tuning objective: choose h per ticker so bars/day ~= target_bars_per_day.
    """
    name: str = "cusum_tuned"
    target_bars_per_day: float = 12.0  # reasonable "few-to-dozen bars/day" starting point
    h_grid: Tuple[float, ...] = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
    price_col: str = "close"
    aggregate_ohlcv: bool = True
    tuned_h_by_ticker: Dict[str, float] = field(default_factory=dict)

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedCUSUMBarSampler":
        tuned: Dict[str, float] = {}

        for t, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns:
                continue

            close = df[self.price_col].to_numpy(dtype=np.float64)

            best_h = None
            best_err = None

            for h in self.h_grid:
                ends = _cusum_event_ends(close, float(h))
                bpd = _bars_per_day(df, ends)
                err = abs(bpd - self.target_bars_per_day)

                # tie-break: prefer slightly sparser (higher h)
                if (best_err is None) or (err < best_err) or (err == best_err and (best_h is None or h > best_h)):
                    best_err = err
                    best_h = float(h)

            if best_h is not None:
                tuned[t] = best_h

        self.tuned_h_by_ticker = tuned
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "h_grid": [float(x) for x in self.h_grid],
            "price_col": self.price_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        h = self.tuned_h_by_ticker.get(ticker, None)
        return {"h": None if h is None else float(h)}

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()

        if ticker not in self.tuned_h_by_ticker:
            raise KeyError(
                f"TunedCUSUMBarSampler has no tuned parameters for ticker={ticker}. "
                f"Make sure sampler.fit(ticker_dfs_train) was called and that {ticker} exists in train."
            )

        h = float(self.tuned_h_by_ticker[ticker])
        close = df[self.price_col].to_numpy(dtype=np.float64)
        ends = _cusum_event_ends(close, h)

        if not self.aggregate_ohlcv:
            return df.iloc[ends].copy()

        return _aggregate_ohlcv_segments(df, ends)