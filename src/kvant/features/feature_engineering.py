"""
Feature engineering module for financial data.

This module provides tools to extract features from OHLCV data, including:
- Basic OHLCV features
- Technical analysis indicators
- Standardized features for machine learning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from kvant.utils.time_utils import ensure_utc_sorted_index


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
    (computed after sampling).
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

    def _transform_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        # Extracts and returns OHLC (Open, High, Low, Close) columns as float.
        ohlc = df.loc[:, ["open", "high", "low", "close"]].astype(float)
        return ohlc

    def _transform_volume(self, df: pd.DataFrame) -> pd.Series:
        # Transforms the volume column based on the specified output format (log1p or raw).
        v_raw = df["volume"].astype(float)
        if self.volume_output == "log1p":
            return np.log1p(v_raw)
        elif self.volume_output == "raw":
            return v_raw
        else:
            raise ValueError("volume_output must be 'raw' or 'log1p'")

    def _transform_ema(self, close: pd.Series) -> pd.DataFrame:
        # Computes Exponential Moving Averages (EMA) and EWM standard deviations for given periods.
        ema_features = {}
        for p in (5, 10, 15, 20, 50):
            n = self._scale(p)
            ema_features[f"ema_close_{p}b"] = close.ewm(span=n, adjust=False, min_periods=n).mean()
            ema_features[f"ewmstd_close_{p}b"] = close.ewm(span=n, adjust=False, min_periods=n).std(bias=False)
        return pd.DataFrame(ema_features)

    def _transform_macd(self, close: pd.Series) -> pd.DataFrame:
        # Calculates MACD (Moving Average Convergence Divergence) and related features.
        n_fast = self._scale(12)
        n_slow = self._scale(26)
        n_signal = self._scale(9)
        ema_fast = close.ewm(span=n_fast, adjust=False, min_periods=n_fast).mean()
        ema_slow = close.ewm(span=n_slow, adjust=False, min_periods=n_slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=n_signal, adjust=False, min_periods=n_signal).mean()
        return pd.DataFrame({
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd - macd_signal,
        })

    def _transform_rsi(self, close: pd.Series) -> pd.DataFrame:
        # Computes Relative Strength Index (RSI) for specified periods.
        rsi_features = {}
        for p in (6, 10, 14):
            n = self._scale(p)
            rsi_features[f"rsi_{p}b"] = self._rsi_wilder(close, n)
        return pd.DataFrame(rsi_features)

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Main transformation pipeline combining OHLC, volume, EMA, MACD, and RSI features.
        df = ensure_utc_sorted_index(df)
        ohlc = self._transform_ohlc(df)
        volume = self._transform_volume(df)
        ema = self._transform_ema(ohlc["close"])
        macd = self._transform_macd(ohlc["close"])
        rsi = self._transform_rsi(ohlc["close"])

        features = pd.concat([ohlc, volume.rename("volume"), ema, macd, rsi], axis=1)
        return features


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
