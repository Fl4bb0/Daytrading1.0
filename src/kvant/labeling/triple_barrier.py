"""
labeling.triple_barrier — Triple-barrier labeling method.

Reference: López de Prado, "Advances in Financial Machine Learning", ch. 3.

For each bar at time t, a position is opened at t's open price. Three barriers
are placed:
  - Upper barrier:   entry * (1 + height)   → label 2 (BUY / take-profit)
  - Lower barrier:   entry * (1 - height)   → label 0 (SHORT / stop-loss)
  - Vertical barrier: t + width_minutes      → label 1 (HOLD / time exit)

The earliest barrier hit determines the label. Barrier hits are evaluated on
future bars only (starting at t+1), so the entry bar itself never determines
the label. If the lower and upper barriers are both hit in the same future bar,
stop-loss wins (conservative).

Bars where entry/exit fall outside the NYSE trading window or have invalid
prices receive label -1 (abstain).

Label convention
----------------
  0 = stop-loss hit   (SHORT)
  1 = vertical exit   (HOLD / time)
  2 = take-profit hit (BUY)
 -1 = abstain / invalid

TripleBarLabel          — frozen dataclass with label + pnl metadata
triple_barrier_label()  — single-bar labeling function
TripleBarrierLabeler    — Labeler subclass: fit() is a no-op, transform() labels a full DataFrame
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tqdm

from kvant.labeling.base import Labeler
from kvant.kmarket_info.is_nyse_open import is_nyse_available, nyse_trade_window_is_valid
from kvant.utils.time_utils import ensure_utc_sorted_index

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TripleBarLabel:
    bar_open_time: pd.Timestamp   # UTC entry timestamp
    bar_close_time: pd.Timestamp  # UTC exit timestamp
    label: int                    # 0 = stop-loss, 1 = time exit, 2 = take-profit
    pnl_fraction: float           # (exit_price - entry_price) / entry_price
    pnl_absolute: float           # exit_price - entry_price  ($ per share)
    height_used: float            # effective fractional barrier half-width used


# ---------------------------------------------------------------------------
# Core single-bar algorithm
# ---------------------------------------------------------------------------

def _to_utc_ts(x: Union[pd.Timestamp, str]) -> pd.Timestamp:
    """Convert any timestamp to tz-naive UTC, matching ensure_utc_sorted_index output."""
    ts = pd.Timestamp(x)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.tz_localize(None)


def _as_utc_aware(ts: pd.Timestamp) -> pd.Timestamp:
    """Re-attach UTC timezone to a tz-naive UTC timestamp for calendar checks."""
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def triple_barrier_label(
    data: pd.DataFrame,
    time_start: Union[pd.Timestamp, str],
    width: int,
    height: float,
) -> Optional[TripleBarLabel]:
    """
    Compute the triple-barrier label for a single entry at *time_start*.

    Parameters
    ----------
    data       : UTC-indexed OHLCV DataFrame (columns: open, high, low, close).
    time_start : Entry timestamp (UTC or tz-naive treated as UTC).
    width      : Vertical barrier in minutes (max holding period).
    height     : Fractional barrier half-width (e.g. 0.01 = ±1 %).

    Returns
    -------
    TripleBarLabel, or None if the bar is invalid / outside trading hours.
    Barrier-hit detection starts from the next bar after entry (strictly future bars).
    """
    if data is None or data.empty:
        return None

    required = {"open", "high", "low", "close"}
    if not required.issubset(data.columns):
        raise ValueError(f"data must have columns {sorted(required)}")

    ts0 = _to_utc_ts(time_start)
    pos = data.index.searchsorted(ts0, side="left")
    if pos >= len(data.index):
        return None

    entry_ts = data.index[pos]

    # Vertical barrier: last bar whose timestamp <= entry + width minutes
    end_target = entry_ts + pd.Timedelta(minutes=int(width))
    end_pos = data.index.searchsorted(end_target, side="right") - 1
    if end_pos <= pos:
        return None

    exit_ts_vertical = data.index[end_pos]

    # Both entry and the outer vertical exit must be tradeable
    if not nyse_trade_window_is_valid(_as_utc_aware(entry_ts), _as_utc_aware(exit_ts_vertical)):
        return None

    entry_price = float(data.iloc[pos]["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    upper = entry_price * (1.0 + float(height))
    lower = entry_price * (1.0 - float(height))
    # Causal labeling: only bars after entry can trigger a barrier hit.
    path  = data.iloc[pos + 1 : end_pos + 1]

    hit_up = path.index[path["high"] >= upper]
    hit_dn = path.index[path["low"]  <= lower]

    if len(hit_up) == 0 and len(hit_dn) == 0:
        # Vertical (time) exit
        label      = 1
        exit_ts    = exit_ts_vertical
        exit_price = float(path.loc[exit_ts, "close"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            return None
    else:
        first_up = hit_up[0] if len(hit_up) else None
        first_dn = hit_dn[0] if len(hit_dn) else None

        # If both hit in the same bar, conservative: stop-loss wins
        if first_dn is not None and (first_up is None or first_dn <= first_up):
            label, exit_ts, exit_price = 0, first_dn, lower
        else:
            label, exit_ts, exit_price = 2, first_up, upper  # type: ignore[assignment]

    # Realised exit must also be within the trading window
    if not nyse_trade_window_is_valid(_as_utc_aware(entry_ts), _as_utc_aware(exit_ts)):
        return None

    pnl_abs  = float(exit_price - entry_price)
    pnl_frac = pnl_abs / entry_price

    return TripleBarLabel(
        bar_open_time=entry_ts,
        bar_close_time=exit_ts,
        label=int(label),
        pnl_fraction=pnl_frac,
        pnl_absolute=pnl_abs,
        height_used=float(height),
    )


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

@dataclass
class TripleBarrierLabeler(Labeler):
    """
    Labels every bar in a DataFrame with the triple-barrier method.

    Parameters
    ----------
    name               : str   — identifier used in experiment configs.
    width_minutes      : int   — vertical barrier in minutes.
    height             : float — fractional barrier half-width (e.g. 0.01 = ±1 %).
    drop_time_exit     : bool  — if True, vertical-exit bars (label==1) are set to -1
                                 (abstain). Useful when only directional signals are wanted.
    show_progress      : bool  — show a tqdm progress bar during transform().

    Label convention
    ----------------
      0 = stop-loss  (SHORT)
      1 = time exit  (HOLD)   — or -1 if drop_time_exit=True
      2 = take-profit (BUY)
     -1 = abstain / invalid
    """
    name: str = "triple_barrier"
    width_minutes: int = 30
    height: float = 0.01
    drop_time_exit: bool = False
    show_progress: bool = True
    brokerage_fee: float = 0.0
    volatility_scale_mode: str = "none"   # "none" | "ticker_std"
    vol_scale_min: float = 0.5
    vol_scale_max: float = 2.0
    use_numba_fastpath: bool = True
    _fit_reference_vol: Optional[float] = field(default=None, init=False, repr=False)

    @staticmethod
    def _series_volatility(df: pd.DataFrame) -> Optional[float]:
        if "close" not in df.columns or len(df) < 3:
            return None
        r = df["close"].astype(float).pct_change().dropna()
        if len(r) == 0:
            return None
        v = float(r.std())
        if not np.isfinite(v) or v <= 0.0:
            return None
        return v

    def _effective_height(self, df: pd.DataFrame) -> float:
        base = float(self.height)
        round_trip_fee = 2.0 * float(self.brokerage_fee)
        if self.volatility_scale_mode == "none":
            return base + round_trip_fee
        if self.volatility_scale_mode != "ticker_std":
            raise ValueError(
                f"Unknown volatility_scale_mode={self.volatility_scale_mode!r}. "
                f"Expected 'none' or 'ticker_std'."
            )

        ref = self._fit_reference_vol
        cur = self._series_volatility(df)
        if ref is None or cur is None or ref <= 0.0:
            return base + round_trip_fee

        raw_scale = cur / ref
        scale = float(np.clip(raw_scale, self.vol_scale_min, self.vol_scale_max))
        return base * scale + round_trip_fee

    def fit(self, df: pd.DataFrame) -> "TripleBarrierLabeler":
        df = ensure_utc_sorted_index(df)
        self._fit_reference_vol = self._series_volatility(df)
        return self

    def fit_from_ticker_dfs(
        self, ticker_dfs: Dict[str, pd.DataFrame]
    ) -> "TripleBarrierLabeler":
        """
        Fit the reference volatility from a dict of per-ticker DataFrames.

        Computes each ticker's return std in isolation, then takes the median
        across tickers as the reference. Use this instead of ``fit`` whenever
        the corpus mixes multiple tickers: a concatenated+sorted DataFrame
        would leak cross-ticker price jumps into ``pct_change`` and corrupt
        the reference.

        Falls back to ``None`` (no scaling) if no usable per-ticker std can
        be computed.
        """
        if self.volatility_scale_mode == "none":
            self._fit_reference_vol = None
            return self

        vols: List[float] = []
        for df in ticker_dfs.values():
            if df is None or len(df) == 0:
                continue
            v = self._series_volatility(ensure_utc_sorted_index(df))
            if v is not None:
                vols.append(v)

        self._fit_reference_vol = float(np.median(vols)) if vols else None
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[Optional[dict]]]:
        """
        Label every bar in *df*.

        Returns
        -------
        labels   : int8 array, shape (n,). -1 = abstain.
        metadata : list of dicts (or None), length n. Each dict contains:
                   bar_open_time, bar_close_time, label, pnl_fraction, pnl_absolute.
        """
        df = ensure_utc_sorted_index(df)
        n  = len(df)
        height_used = self._effective_height(df)
        labels: np.ndarray         = np.full(n, -1, dtype=np.int8)
        metadata: List[Optional[dict]] = [None] * n

        iterable = enumerate(df.index)
        if self.show_progress:
            iterable = enumerate(
                tqdm.tqdm(df.index, desc=f"Labeling [{self.name}]", leave=False)
            )

        if self.use_numba_fastpath and self.width_minutes > 0 and n > 0:
            labels, metadata = _transform_fast(
                df=df,
                n=n,
                width_minutes=self.width_minutes,
                height_used=height_used,
                drop_time_exit=self.drop_time_exit,
                volatility_scale_mode=self.volatility_scale_mode,
                fit_reference_vol=self._fit_reference_vol,
                fallback_iterable=iterable,
            )
            return labels, metadata

        for i, t in iterable:
            res = triple_barrier_label(df, time_start=t, width=self.width_minutes, height=height_used)
            if res is None:
                continue

            metadata[i] = dataclasses.asdict(res)
            metadata[i]["volatility_scale_mode"] = self.volatility_scale_mode
            metadata[i]["fit_reference_vol"] = self._fit_reference_vol

            lab = int(res.label)
            if self.drop_time_exit and lab == 1:
                continue   # leave labels[i] == -1
            labels[i] = lab

        return labels, metadata


def _transform_fast(
    *,
    df: pd.DataFrame,
    n: int,
    width_minutes: int,
    height_used: float,
    drop_time_exit: bool,
    volatility_scale_mode: str,
    fit_reference_vol: Optional[float],
    fallback_iterable,
) -> Tuple[np.ndarray, List[Optional[dict]]]:
    labels: np.ndarray = np.full(n, -1, dtype=np.int8)
    metadata: List[Optional[dict]] = [None] * n

    if njit is None:
        # Keep exact behavior if numba is unavailable.
        for i, t in fallback_iterable:
            res = triple_barrier_label(df, time_start=t, width=width_minutes, height=height_used)
            if res is None:
                continue
            metadata[i] = dataclasses.asdict(res)
            metadata[i]["volatility_scale_mode"] = volatility_scale_mode
            metadata[i]["fit_reference_vol"] = fit_reference_vol
            lab = int(res.label)
            if not (drop_time_exit and lab == 1):
                labels[i] = lab
        return labels, metadata

    open_arr = df["open"].to_numpy(dtype=np.float64, copy=False)
    high_arr = df["high"].to_numpy(dtype=np.float64, copy=False)
    low_arr = df["low"].to_numpy(dtype=np.float64, copy=False)
    close_arr = df["close"].to_numpy(dtype=np.float64, copy=False)
    ts_ns = df.index.view("i8")
    width_ns = int(pd.Timedelta(minutes=int(width_minutes)).value)

    out_label, out_exit_pos, out_exit_price = _scan_barriers_numba(
        open_arr, high_arr, low_arr, close_arr, ts_ns, width_ns, float(height_used)
    )

    valid_trade_cache: dict[int, bool] = {}

    def _is_valid_trade_ts(ts: pd.Timestamp) -> bool:
        k = int(ts.value)
        cached = valid_trade_cache.get(k)
        if cached is not None:
            return cached
        v = bool(is_nyse_available(_as_utc_aware(ts)))
        valid_trade_cache[k] = v
        return v

    for i in range(n):
        lab = int(out_label[i])
        if lab < 0:
            continue
        exit_pos = int(out_exit_pos[i])
        if exit_pos < 0 or exit_pos >= n:
            continue

        entry_ts = df.index[i]
        exit_ts = df.index[exit_pos]
        if not (_is_valid_trade_ts(entry_ts) and _is_valid_trade_ts(exit_ts)):
            continue

        entry_price = float(open_arr[i])
        exit_price = float(out_exit_price[i])
        pnl_abs = float(exit_price - entry_price)
        pnl_frac = float(pnl_abs / entry_price)

        metadata[i] = {
            "bar_open_time": entry_ts,
            "bar_close_time": exit_ts,
            "label": lab,
            "pnl_fraction": pnl_frac,
            "pnl_absolute": pnl_abs,
            "height_used": float(height_used),
            "volatility_scale_mode": volatility_scale_mode,
            "fit_reference_vol": fit_reference_vol,
        }

        if drop_time_exit and lab == 1:
            continue
        labels[i] = lab

    return labels, metadata


if njit is not None:
    @njit(cache=True)
    def _scan_barriers_numba(open_arr, high_arr, low_arr, close_arr, ts_ns, width_ns, height):
        n = len(open_arr)
        out_label = np.full(n, -1, dtype=np.int8)
        out_exit_pos = np.full(n, -1, dtype=np.int32)
        out_exit_price = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            entry_price = open_arr[i]
            if not np.isfinite(entry_price) or entry_price <= 0.0:
                continue

            end_target = ts_ns[i] + width_ns
            end_pos = i
            while end_pos + 1 < n and ts_ns[end_pos + 1] <= end_target:
                end_pos += 1
            if end_pos <= i:
                continue

            upper = entry_price * (1.0 + height)
            lower = entry_price * (1.0 - height)
            first_up = -1
            first_dn = -1
            j = i + 1
            while j <= end_pos:
                if first_up < 0 and high_arr[j] >= upper:
                    first_up = j
                if first_dn < 0 and low_arr[j] <= lower:
                    first_dn = j
                if first_up >= 0 and first_dn >= 0:
                    break
                j += 1

            if first_up < 0 and first_dn < 0:
                exit_pos = end_pos
                exit_price = close_arr[end_pos]
                if not np.isfinite(exit_price) or exit_price <= 0.0:
                    continue
                out_label[i] = 1
                out_exit_pos[i] = exit_pos
                out_exit_price[i] = exit_price
            else:
                if first_dn >= 0 and (first_up < 0 or first_dn <= first_up):
                    out_label[i] = 0
                    out_exit_pos[i] = first_dn
                    out_exit_price[i] = lower
                else:
                    out_label[i] = 2
                    out_exit_pos[i] = first_up
                    out_exit_price[i] = upper

        return out_label, out_exit_pos, out_exit_price
else:
    def _scan_barriers_numba(open_arr, high_arr, low_arr, close_arr, ts_ns, width_ns, height):
        raise RuntimeError("numba fastpath unavailable")
