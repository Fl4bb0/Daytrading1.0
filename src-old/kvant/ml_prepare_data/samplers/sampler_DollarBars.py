"""
Dollar Bar Sampler
==================
Dollar bars are an alternative to time bars. Instead of closing a new bar every
N minutes, a bar closes whenever a fixed *dollar volume* (price × volume) has
traded since the last bar closed.

Why dollar bars?
- Time bars oversample quiet periods and undersample busy ones, making the
  series non-stationary in volatility and volume.
- Dollar bars produce roughly the same amount of *information* per bar, which
  leads to better-behaved returns and more i.i.d.-like features for ML models.

How the threshold D is chosen:
  fit() tries a log-spaced grid of candidate thresholds on the training data and
  picks the D that produces bars/day closest to `target_bars_per_day`.
  A log-spaced grid is used because dollar volumes can span many orders of
  magnitude across tickers (e.g. micro-cap vs large-cap stocks).
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index
from kvant.ml_prepare_data.samplers.sampling import BaseBarSampler


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _dollar_bar_ends(dollar_volume: np.ndarray, threshold: float) -> np.ndarray:
    """
    Scan rows left-to-right, accumulating dollar volume.
    Every time the running total crosses `threshold`, record that row index as
    a bar-end and reset the accumulator to zero.

    Returns an array of row indices (0-based, inclusive) at which bars close.
    """
    if len(dollar_volume) == 0:
        return np.array([], dtype=np.int64)

    ends = []
    cumulative = 0.0
    for i, dv in enumerate(dollar_volume):
        cumulative += float(dv)
        if cumulative >= threshold:
            ends.append(i)
            cumulative = 0.0  # reset — start counting the next bar

    return np.asarray(ends, dtype=np.int64)


# ---------------------------------------------------------------------------
# OHLCV aggregation
# ---------------------------------------------------------------------------

def _aggregate_ohlcv_segments(df: pd.DataFrame, ends: np.ndarray) -> pd.DataFrame:
    """
    Given a raw tick/minute DataFrame and bar-end row indices, aggregate each
    segment into a single OHLCV row.

    The timestamp of each bar is taken from the *last* row in the segment
    (i.e. the moment the bar closed).

    If `ends` is empty (no threshold was ever crossed), the entire DataFrame
    is collapsed into one bar — this is a safe fallback.
    """
    if len(df) == 0:
        return df.copy()

    if len(ends) == 0:
        # Fallback: treat the whole series as one bar
        seg = df.iloc[:]
        bar = {}
        if "open" in seg:   bar["open"]   = float(seg["open"].iloc[0])
        if "high" in seg:   bar["high"]   = float(seg["high"].max())
        if "low" in seg:    bar["low"]    = float(seg["low"].min())
        if "close" in seg:  bar["close"]  = float(seg["close"].iloc[-1])
        if "volume" in seg: bar["volume"] = float(seg["volume"].sum())
        return pd.DataFrame([bar], index=pd.DatetimeIndex([df.index[-1]]))

    # Deduplicate and clip so indices are safe
    ends = np.unique(np.clip(ends, 0, len(df) - 1))

    rows, idx = [], []
    start = 0
    for end in ends:
        if end < start:
            continue
        seg = df.iloc[start : end + 1]
        bar = {}
        if "open" in seg:   bar["open"]   = float(seg["open"].iloc[0])
        if "high" in seg:   bar["high"]   = float(seg["high"].max())
        if "low" in seg:    bar["low"]    = float(seg["low"].min())
        if "close" in seg:  bar["close"]  = float(seg["close"].iloc[-1])
        if "volume" in seg: bar["volume"] = float(seg["volume"].sum())
        rows.append(bar)
        idx.append(df.index[end])   # bar timestamp = time the bar closed
        start = end + 1

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    return ensure_utc_sorted_index(out)


# ---------------------------------------------------------------------------
# Helper: measure bar density
# ---------------------------------------------------------------------------

def _bars_per_day(df: pd.DataFrame, ends: np.ndarray) -> float:
    """
    Given the raw DataFrame and the bar-end indices produced by a candidate
    threshold, return the average number of bars that closed per calendar day.
    Used by fit() to evaluate each candidate threshold.
    """
    if len(df) == 0 or len(ends) == 0:
        return 0.0
    idx = ensure_utc_sorted_index(df).index
    days = idx.normalize()                      # strip time, keep date
    n_days = int(days.nunique())
    if n_days <= 0:
        return 0.0
    end_days = days[ends]
    return float(pd.Series(end_days).value_counts().sum() / n_days)


# ---------------------------------------------------------------------------
# Sampler class
# ---------------------------------------------------------------------------

@dataclass
class TunedDollarBarSampler(BaseBarSampler):
    """
    Per-ticker tuned dollar-bar sampler.

    A new bar closes whenever the cumulative dollar volume (price x volume)
    since the last bar exceeds threshold D.  D is tuned per ticker on the
    training set so that bars/day ~ target_bars_per_day.

    Typical usage
    -------------
    sampler = TunedDollarBarSampler(target_bars_per_day=12)
    sampler.fit(train_dfs)          # tunes D for each ticker
    bars = sampler.transform(df, ticker="AAPL")

    Parameters
    ----------
    target_bars_per_day : float
        Desired number of dollar bars per trading day. Lower -> fewer, coarser
        bars. Higher -> more, finer bars.
    n_grid : int
        How many threshold candidates to evaluate during tuning. More candidates
        give a finer search but take longer. 30 is usually sufficient.
    price_col / volume_col : str
        Column names in the input DataFrame used to compute dollar volume.
    aggregate_ohlcv : bool
        If True (default), each bar is aggregated to open/high/low/close/volume.
        If False, only the closing row of each bar is returned (useful for
        attaching labels at the exact bar-close timestamp).
    """
    name: str = "dollar_bars_tuned"
    target_bars_per_day: float = 12.0
    n_grid: int = 30
    price_col: str = "close"
    volume_col: str = "volume"
    aggregate_ohlcv: bool = True
    tuned_threshold_by_ticker: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def _make_grid(self, dollar_volume: np.ndarray) -> Tuple[float, ...]:
        """
        Build a log-spaced grid of candidate thresholds anchored to the data.

        Lower bound: average dollar volume per row -> would produce one bar per
          row (maximum granularity).
        Upper bound: total dollar volume -> would produce exactly one bar for
          the entire series (minimum granularity).

        Log spacing is important because dollar volumes can differ by orders of
        magnitude between tickers, so equal linear steps would waste most of the
        search budget on irrelevant values.
        """
        total = float(dollar_volume.sum())
        if total <= 0:
            return ()
        lo = total / max(len(dollar_volume), 1)  # avg dollar vol per row
        hi = total                                # one bar for the whole series
        return tuple(np.logspace(np.log10(lo), np.log10(hi), self.n_grid))

    # ------------------------------------------------------------------ #
    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "TunedDollarBarSampler":
        """
        Tune the dollar threshold D for every ticker in the training set.

        For each ticker:
          1. Compute per-row dollar volume = close x volume.
          2. Try each candidate D from the log-spaced grid.
          3. Simulate bar formation with _dollar_bar_ends().
          4. Measure bars/day with _bars_per_day().
          5. Keep the D whose bars/day is closest to target_bars_per_day.
             Tie-break: prefer the larger D (sparser bars) for stability.

        Results are stored in self.tuned_threshold_by_ticker and used later
        by transform().
        """
        tuned: Dict[str, float] = {}

        for ticker, df in ticker_dfs_train.items():
            if df is None or len(df) < 10:
                continue
            df = ensure_utc_sorted_index(df)
            if self.price_col not in df.columns or self.volume_col not in df.columns:
                continue

            dollar_volume = (
                df[self.price_col].to_numpy(dtype=np.float64)
                * df[self.volume_col].to_numpy(dtype=np.float64)
            )
            grid = self._make_grid(dollar_volume)
            if not grid:
                continue

            best_threshold, best_err = None, None

            for D in grid:
                ends = _dollar_bar_ends(dollar_volume, D)
                bpd  = _bars_per_day(df, ends)
                err  = abs(bpd - self.target_bars_per_day)

                # Prefer lower error; on a tie prefer larger D (sparser bars)
                if (best_err is None) or (err < best_err) or (
                    err == best_err and best_threshold is not None and D > best_threshold
                ):
                    best_err       = err
                    best_threshold = float(D)

            if best_threshold is not None:
                tuned[ticker] = best_threshold

        self.tuned_threshold_by_ticker = tuned
        return self

    # ------------------------------------------------------------------ #
    def get_global_meta(self) -> dict:
        """Sampler-level config saved alongside experiment artifacts."""
        return {
            "name": self.name,
            "target_bars_per_day": float(self.target_bars_per_day),
            "n_grid": int(self.n_grid),
            "price_col": self.price_col,
            "volume_col": self.volume_col,
            "aggregate_ohlcv": bool(self.aggregate_ohlcv),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        """Returns the tuned dollar threshold for this ticker."""
        D = self.tuned_threshold_by_ticker.get(ticker, None)
        return {"dollar_threshold": None if D is None else float(D)}

    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        """
        Apply dollar-bar sampling to a single ticker's DataFrame.

        Requires fit() to have been called first so that the threshold D for
        this ticker is known.  Raises KeyError if the ticker was not seen
        during training.
        """
        df = ensure_utc_sorted_index(df)
        if len(df) == 0:
            return df.copy()

        if ticker not in self.tuned_threshold_by_ticker:
            raise KeyError(
                f"TunedDollarBarSampler has no tuned parameters for ticker={ticker}. "
                f"Call sampler.fit(ticker_dfs_train) first and ensure {ticker} is in train."
            )

        D = float(self.tuned_threshold_by_ticker[ticker])
        dollar_volume = (
            df[self.price_col].to_numpy(dtype=np.float64)
            * df[self.volume_col].to_numpy(dtype=np.float64)
        )
        # Find the row index where each bar closes
        ends = _dollar_bar_ends(dollar_volume, D)

        if not self.aggregate_ohlcv:
            # Return only the closing row of each bar (no aggregation)
            return df.iloc[ends].copy()

        # Aggregate each segment into a single OHLCV bar
        return _aggregate_ohlcv_segments(df, ends)