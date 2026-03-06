# New, current, implementation.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, Dict

import numpy as np
import pandas as pd

from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index


# ---------------------------------------------------------------------
# Protocol (same public API you already use)
# ---------------------------------------------------------------------
class FeatureEngineer(Protocol):
    name: str
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer": ...
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]: ...
    def get_meta(self) -> dict: ...


# ---------------------------------------------------------------------
# Base class: compute a DataFrame of features; handles common utilities
# ---------------------------------------------------------------------
@dataclass
class BaseDFEngineer:
    """
    Subclasses implement _transform_df -> pd.DataFrame of numeric features.
    This keeps feature name bookkeeping and scaling straightforward.
    """
    name: str = "base_df_eng"
    fillna_value: Optional[float] = 0.0  # None => keep NaN

    def fit(self, df: pd.DataFrame) -> "BaseDFEngineer":
        return self

    def get_meta(self) -> dict:
        return {"name": self.name, "fillna_value": self.fillna_value}

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        f = self._transform_df(df)
        if self.fillna_value is not None:
            f = f.fillna(self.fillna_value)
        X = f.to_numpy(dtype=np.float32)
        return X, list(f.columns)


# ---------------------------------------------------------------------
# 1) Simple OHLCV
# ---------------------------------------------------------------------
@dataclass
class OHLCVFeatures(BaseDFEngineer):
    name: str = "ohlcv"
    cols: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    log1p_volume: bool = True

    def get_meta(self) -> dict:
        return {
            **super().get_meta(),
            "cols": list(self.cols),
            "log1p_volume": self.log1p_volume,
        }

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        feat = df.loc[:, list(self.cols)].copy()

        if self.log1p_volume and "volume" in feat.columns:
            feat["volume"] = np.log1p(feat["volume"].astype(float))

        return feat.astype(float)


# ---------------------------------------------------------------------
# 2) TA10 indicator set aligned to the paper’s feature section
#     (computed after sampling; scaling is handled separately by wrapper)
# ---------------------------------------------------------------------
@dataclass
class IntradayTA10Features(BaseDFEngineer):
    """
    Implements the 10 feature groups described in the paper’s feature engineering section
    (computed after sampling) .
    """
    name: str = "intraday_ta10"

    cols: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    volume_output: str = "log1p"  # "raw" or "log1p" for the *volume feature column*
    include_time_features: bool = True

    # Optional: period scaling if you want to reinterpret "bar" length
    typical_bar_minutes: Optional[int] = None  # None => no scaling (periods are in bars)
    data_bar_minutes: int = 1

    def get_meta(self) -> dict:
        return {
            **super().get_meta(),
            "cols": list(self.cols),
            "volume_output": self.volume_output,
            "include_time_features": self.include_time_features,
            "typical_bar_minutes": self.typical_bar_minutes,
            "data_bar_minutes": self.data_bar_minutes,
        }

    def _scale(self, n_bars_in_paper: int) -> int:
        if self.typical_bar_minutes is None:
            return max(1, int(n_bars_in_paper))
        scaled = int(round(n_bars_in_paper * self.typical_bar_minutes / self.data_bar_minutes))
        return max(1, scaled)

    @staticmethod
    def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
        denom = denom.replace(0.0, np.nan)
        return numer / denom

    @staticmethod
    def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        x = df.loc[:, list(self.cols)].copy()

        # Core series
        o = x["open"].astype(float)
        h = x["high"].astype(float)
        l = x["low"].astype(float)
        c = x["close"].astype(float)
        v_raw = x["volume"].astype(float)

        feat = pd.DataFrame(index=x.index)

        # Base OHLC
        feat["open"] = o
        feat["high"] = h
        feat["low"] = l
        feat["close"] = c

        # Volume feature column (raw or log1p), but use RAW volume in CMF/MFI formulas
        if self.volume_output == "log1p":
            feat["volume"] = np.log1p(v_raw)
        elif self.volume_output == "raw":
            feat["volume"] = v_raw
        else:
            raise ValueError("volume_output must be 'raw' or 'log1p'")

        # 7) Historical returns between consecutive periods (log return)
        feat["logret_1"] = np.log(c).diff()

        # 1) EMA and EWM std of close: 5,10,15,20,50
        for p in (5, 10, 15, 20, 50):
            n = self._scale(p)
            feat[f"ema_close_{p}b"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
            feat[f"ewmstd_close_{p}b"] = c.ewm(span=n, adjust=False, min_periods=n).std(bias=False)

        # 2) MACD (12,26) + signal(9) + hist
        n_fast = self._scale(12)
        n_slow = self._scale(26)
        n_signal = self._scale(9)
        ema_fast = c.ewm(span=n_fast, adjust=False, min_periods=n_fast).mean()
        ema_slow = c.ewm(span=n_slow, adjust=False, min_periods=n_slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=n_signal, adjust=False, min_periods=n_signal).mean()
        feat["macd"] = macd
        feat["macd_signal"] = macd_signal
        feat["macd_hist"] = macd - macd_signal

        # 3) RSI: 6,10,14
        for p in (6, 10, 14):
            n = self._scale(p)
            feat[f"rsi_{p}b"] = self._rsi_wilder(c, n)

        # 4) Stochastic Oscillator %K, %D (lookback=14, d=3)
        n_stoch = self._scale(14)
        n_d = self._scale(3)
        low_n = l.rolling(window=n_stoch, min_periods=n_stoch).min()
        high_n = h.rolling(window=n_stoch, min_periods=n_stoch).max()
        stoch_k = 100.0 * self._safe_div((c - low_n), (high_n - low_n))
        stoch_d = stoch_k.rolling(window=n_d, min_periods=n_d).mean()
        feat["stoch_k"] = stoch_k
        feat["stoch_d"] = stoch_d

        # 5) Williams %R lookback=14
        feat["willr_14"] = -100.0 * self._safe_div((high_n - c), (high_n - low_n))

        # 6) Bollinger bands: period=5, n_std=2
        n_bb = self._scale(5)
        bb_mid = c.rolling(window=n_bb, min_periods=n_bb).mean()
        bb_std = c.rolling(window=n_bb, min_periods=n_bb).std(ddof=0)
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        feat["bb_mid"] = bb_mid
        feat["bb_upper"] = bb_upper
        feat["bb_lower"] = bb_lower
        feat["bb_width"] = self._safe_div((bb_upper - bb_lower), bb_mid)
        feat["bb_pctb"] = self._safe_div((c - bb_lower), (bb_upper - bb_lower))

        # 8) CMF period=21
        n_cmf = self._scale(21)
        mf_mult = self._safe_div(((c - l) - (h - c)), (h - l))
        mf_vol = mf_mult * v_raw
        feat["cmf_21"] = self._safe_div(
            mf_vol.rolling(window=n_cmf, min_periods=n_cmf).sum(),
            v_raw.rolling(window=n_cmf, min_periods=n_cmf).sum(),
        )

        # 9) MFI period=14
        n_mfi = self._scale(14)
        tp = (h + l + c) / 3.0
        raw_mf = tp * v_raw
        tp_delta = tp.diff()
        pos_mf = raw_mf.where(tp_delta > 0.0, 0.0)
        neg_mf = raw_mf.where(tp_delta < 0.0, 0.0).abs()
        pos_sum = pos_mf.rolling(window=n_mfi, min_periods=n_mfi).sum()
        neg_sum = neg_mf.rolling(window=n_mfi, min_periods=n_mfi).sum()
        mfr = self._safe_div(pos_sum, neg_sum)
        feat["mfi_14"] = 100.0 - (100.0 / (1.0 + mfr))

        # 10) Sine/cosine of hour and weekday
        if self.include_time_features:
            idx = feat.index
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            hour = idx.hour.to_numpy()
            weekday = idx.weekday.to_numpy()
            feat["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
            feat["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
            feat["sin_wday"] = np.sin(2.0 * np.pi * weekday / 7.0)
            feat["cos_wday"] = np.cos(2.0 * np.pi * weekday / 7.0)

        return feat


# ---------------------------------------------------------------------
# Paper-compatible scaling: fit mean/std on TRAIN, apply to all splits
# ---------------------------------------------------------------------
@dataclass
class StandardizedFeatures:
    """
    Wraps another engineer and standardizes outputs:
      X_scaled = (X - mean) / std

    This matches the paper’s scaling step .
    """
    base: BaseDFEngineer
    name: str = "standardized"
    eps: float = 1e-12

    mean_: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    std_: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    feature_names_: Optional[list[str]] = field(default=None, init=False, repr=False)

    def fit(self, df: pd.DataFrame) -> "StandardizedFeatures":
        X, names = self.base.transform(df)
        self.feature_names_ = names
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(sd < self.eps, 1.0, sd)
        self.mean_ = mu.astype(np.float32)
        self.std_ = sd.astype(np.float32)
        return self

    def get_meta(self) -> dict:
        return {
            "name": self.name,
            "base": self.base.get_meta(),
            "eps": float(self.eps),
            "n_features": None if self.feature_names_ is None else int(len(self.feature_names_)),
        }

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardizedFeatures.transform called before fit()")
        X, names = self.base.transform(df)
        if names != self.feature_names_:
            raise RuntimeError("Feature names changed between fit and transform.")
        X = (X - self.mean_) / self.std_
        return X.astype(np.float32), list(names)