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
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(100.0)  # avg_loss == 0 → all gains → RSI = 100

    def _transform_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        # Log-return-based OHLC features — stationary across time, no price-level drift.
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        safe_open = open_.replace(0.0, np.nan)
        return pd.DataFrame({
            "close_ret": np.log(close / close.shift(1)),
            "hl_range":  np.log(high / low.replace(0.0, np.nan)),
            "co_ret":    np.log(close / safe_open),
            "ho_ret":    np.log(high  / safe_open),
            "lo_ret":    np.log(low   / safe_open),
        }, index=df.index)

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
        # Price deviation from EMA (close/ema - 1) and normalised EWM std — both stationary.
        ema_features = {}
        for p in (5, 10, 15, 20, 50):
            n = self._scale(p)
            ema = close.ewm(span=n, adjust=False, min_periods=n).mean()
            ema_std = close.ewm(span=n, adjust=False, min_periods=n).std(bias=False)
            safe_ema = ema.replace(0.0, np.nan)
            ema_features[f"ema_dev_{p}b"] = close / safe_ema - 1.0
            ema_features[f"ewmstd_{p}b"] = ema_std / safe_ema
        return pd.DataFrame(ema_features)

    def _transform_macd(self, close: pd.Series) -> pd.DataFrame:
        # MACD values normalised by close price so they are scale-independent.
        n_fast = self._scale(12)
        n_slow = self._scale(26)
        n_signal = self._scale(9)
        ema_fast = close.ewm(span=n_fast, adjust=False, min_periods=n_fast).mean()
        ema_slow = close.ewm(span=n_slow, adjust=False, min_periods=n_slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=n_signal, adjust=False, min_periods=n_signal).mean()
        safe_close = close.replace(0.0, np.nan)
        return pd.DataFrame({
            "macd":        macd / safe_close,
            "macd_signal": macd_signal / safe_close,
            "macd_hist":   (macd - macd_signal) / safe_close,
        })

    def _transform_rsi(self, close: pd.Series) -> pd.DataFrame:
        rsi_features = {}
        for p in (6, 10, 14):
            n = self._scale(p)
            rsi_features[f"rsi_{p}b"] = self._rsi_wilder(close, n)
        return pd.DataFrame(rsi_features)

    def _transform_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        # Wilder's ATR normalised by close — volatility regime feature distinct
        # from realized_vol since it uses gap/range, not just close-to-close.
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        n = self._scale(14)
        atr = true_range.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        safe_close = close.replace(0.0, np.nan)
        return pd.DataFrame({"atr_14b": atr / safe_close}, index=df.index)

    def _transform_bollinger(self, close: pd.Series) -> pd.DataFrame:
        # %B positions price within its recent distribution; bandwidth flags
        # volatility expansion/contraction independently of ATR.
        n = self._scale(20)
        sma = close.rolling(n, min_periods=n).mean()
        std = close.rolling(n, min_periods=n).std(ddof=0)
        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        band_range = (upper - lower).replace(0.0, np.nan)
        bb_pct = (close - lower) / band_range
        bb_bandwidth = band_range / sma.replace(0.0, np.nan)
        return pd.DataFrame(
            {"bb_pct_20b": bb_pct, "bb_bandwidth_20b": bb_bandwidth},
            index=close.index,
        )

    def _transform_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        # Cumulative VWAP reset each calendar day, plus close-to-VWAP deviation.
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float).clip(lower=0.0)

        typical_price = (high + low + close) / 3.0
        day_key = df.index.floor("D")

        cum_tp_vol = (typical_price * vol).groupby(day_key).cumsum()
        cum_vol    = vol.groupby(day_key).cumsum()

        vwap     = cum_tp_vol / cum_vol.replace(0.0, np.nan)
        vwap_dev = (close - vwap) / vwap.replace(0.0, np.nan)
        return pd.DataFrame({"vwap": vwap, "vwap_dev": vwap_dev}, index=df.index)

    def _transform_vol_regime(self, close: pd.Series) -> pd.DataFrame:
        # Rolling realized-vol ratio: short/long window flags vol expansion/contraction.
        n_short = self._scale(5)
        n_long  = self._scale(20)
        log_ret = np.log(close / close.shift(1))

        vol_short = log_ret.rolling(n_short, min_periods=2).std()
        vol_long  = log_ret.rolling(n_long,  min_periods=5).std()
        vol_ratio = vol_short / vol_long.replace(0.0, np.nan)

        return pd.DataFrame(
            {"realized_vol_20b": vol_long, "vol_regime_ratio": vol_ratio},
            index=close.index,
        )

    def _transform_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sin/cos encoding of intraday position within the NYSE session (14:30–21:00 UTC).
        minutes_since_open = np.clip(df.index.hour * 60 + df.index.minute - 14 * 60 - 30, 0, 390)
        frac = minutes_since_open / 390.0
        return pd.DataFrame(
            {
                "time_of_day_sin": np.sin(2 * np.pi * frac),
                "time_of_day_cos": np.cos(2 * np.pi * frac),
            },
            index=df.index,
        )

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        close  = df["close"].astype(float)
        ohlc   = self._transform_ohlc(df)
        volume = self._transform_volume(df)
        ema    = self._transform_ema(close)
        macd   = self._transform_macd(close)
        rsi    = self._transform_rsi(close)
        atr    = self._transform_atr(df)
        boll   = self._transform_bollinger(close)
        vwap   = self._transform_vwap(df)
        regime = self._transform_vol_regime(close)

        parts = [ohlc, volume.rename("volume"), ema, macd, rsi, atr, boll, vwap, regime]
        if self.include_time_features:
            parts.append(self._transform_time_features(df))

        return pd.concat(parts, axis=1)


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

    def fit_from_ticker_dfs(
        self,
        ticker_dfs: dict[str, pd.DataFrame],
    ) -> "StandardizedFeatures":
        """
        Fit global standardization stats from per-ticker feature matrices.

        Rolling features such as EMA, MACD, and RSI must be computed on each
        ticker independently. After that, the resulting feature rows can be
        pooled to learn one global mean/std for the model input scale.
        """
        X_parts: list[np.ndarray] = []
        names_ref: Optional[list[str]] = None

        for df in ticker_dfs.values():
            if df is None or len(df) == 0:
                continue

            X, names = self.base.transform(df)
            if names_ref is None:
                names_ref = names
            elif names != names_ref:
                raise RuntimeError("Feature names changed between tickers.")

            X_parts.append(X)

        if not X_parts or names_ref is None:
            raise RuntimeError("No feature rows available for ticker-aware standardization.")

        X_all = np.concatenate(X_parts, axis=0)
        mu = np.nanmean(X_all, axis=0)
        sd = np.nanstd(X_all, axis=0)
        sd = np.where(sd < self.eps, 1.0, sd)

        self.feature_names_ = names_ref
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
