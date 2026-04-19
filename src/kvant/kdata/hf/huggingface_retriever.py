"""
kdata.huggingface_retriever — HuggingFace 1m OHLCV data ingestion.

Fetches monthly shards of 1-minute OHLCV data from HuggingFace datasets,
normalizes to project schema, and returns as DataFrames.

HuggingFaceRetriever(dataset_id, cache_dir)
  .get_month_shard(symbol, year_month)  → pd.DataFrame
  .get_history(symbol, start, end)      → pd.DataFrame
  .get_ticker_data(symbols, start, end) → Dict[str, pd.DataFrame]
  .get_current_price(symbol, **kwargs)  → Dict
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from kvant.kdata.retriever import DataRetriever

logger = logging.getLogger(__name__)

_COLUMNS = ["open", "high", "low", "close", "volume"]


class HuggingFaceRetriever(DataRetriever):
    """
    Retrieves 1m OHLCV data from HuggingFace datasets.

    Assumes HuggingFace dataset has monthly splits named '{SYMBOL}-{YYYY-MM}'
    (e.g., 'AAPL-2025-03', 'MSFT-2025-03').

    Parameters
    ----------
    dataset_id : str
        HuggingFace dataset identifier (e.g., 'username/stocks-1m').
        Can be "" to disable HF source.
    cache_dir : str, optional
        Cache directory for HF downloads. Defaults to ~/.cache/huggingface.
    """

    def __init__(
        self,
        dataset_id: str = "",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.dataset_id = dataset_id
        resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "huggingface"
        self.cache_dir = str(resolved_cache_dir)
        self._default_train_df_cache: Optional[pd.DataFrame] = None
        self._month_frame_cache: Dict[str, pd.DataFrame] = {}
        if not dataset_id:
            logger.warning("HuggingFaceRetriever initialized with empty dataset_id; will raise on use")

    def _check_enabled(self) -> None:
        """Raise if retriever not properly configured."""
        if not self.dataset_id:
            raise ValueError(
                "HuggingFaceRetriever.dataset_id is empty. "
                "Set 'hf_config.dataset_id' in pipeline.toml to enable HF source."
            )

    def get_month_shard(
            self,
            symbol: str,
            year_month: str,  # "2025-03"
    ) -> pd.DataFrame:
        """
        Fetch one month of 1m data for symbol from HuggingFace.

        Supports two dataset layouts:
          0) month parquet shards in repo files (e.g. data/ohlcv_2025-03.parquet)
          1) split-specific config per symbol-month (e.g. name="AAPL-2025-03")
          2) single default config containing all rows, then filtered by symbol+month
        """
        self._check_enabled()

        # Validate month format early
        try:
            month_start, month_end = self._month_bounds(year_month)
        except Exception:
            logger.warning(f"Invalid year_month={year_month!r}, expected YYYY-MM")
            return pd.DataFrame(columns=_COLUMNS)

        try:
            return self._get_month_shard_from_month_file(symbol, year_month, month_start, month_end)
        except Exception as e:
            logger.info(
                f"Month parquet shard for {year_month!r} unavailable, falling back to config-based loading ({e})"
            )

        return self._get_month_shard_via_dataset_configs(symbol, year_month, month_start, month_end)

    def get_month_shards(
        self,
        symbols: List[str],
        year_month: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch one month of data for multiple symbols.

        Uses a month parquet shard when available so import_month can scan a
        single monthly file once instead of repeating remote dataset resolution
        per symbol.
        """
        self._check_enabled()

        month_start, month_end = self._month_bounds(year_month)
        original_by_upper = {str(sym).upper(): str(sym) for sym in symbols}
        requested = list(original_by_upper.keys())
        result = {str(sym): pd.DataFrame(columns=_COLUMNS) for sym in symbols}

        try:
            month_df = self._load_month_frame(year_month)
            if month_df.empty:
                return result

            symbol_col = self._find_symbol_column(month_df)
            if symbol_col is None:
                raise ValueError(f"Could not find symbol/ticker column in month shard {year_month}")

            symbol_series = month_df[symbol_col].astype(str).str.upper()
            df_filtered = month_df[symbol_series.isin(requested)].copy()
            if df_filtered.empty:
                return result

            for sym, group in df_filtered.groupby(symbol_series.loc[df_filtered.index]):
                normalized = self._normalize_schema(group)
                normalized = normalized[(normalized.index >= month_start) & (normalized.index < month_end)]
                result[original_by_upper[sym]] = normalized.sort_index()

            return result
        except Exception as e:
            logger.info(
                f"Batch month loading for {year_month!r} unavailable, falling back to per-symbol fetches ({e})"
            )
            return {
                str(sym): self._get_month_shard_via_dataset_configs(str(sym), year_month, month_start, month_end)
                for sym in symbols
            }

    def _get_month_shard_from_month_file(
        self,
        symbol: str,
        year_month: str,
        month_start: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> pd.DataFrame:
        month_df = self._load_month_frame(year_month)
        if month_df.empty:
            return pd.DataFrame(columns=_COLUMNS)

        symbol_col = self._find_symbol_column(month_df)
        if symbol_col is None:
            raise ValueError(f"Could not find symbol/ticker column in month shard {year_month}")

        df_filtered = month_df[month_df[symbol_col].astype(str).str.upper() == symbol.upper()]
        if df_filtered.empty:
            return pd.DataFrame(columns=_COLUMNS)

        df_filtered = self._normalize_schema(df_filtered)
        df_filtered = df_filtered[(df_filtered.index >= month_start) & (df_filtered.index < month_end)]
        return df_filtered.sort_index()

    def _load_month_frame(self, year_month: str) -> pd.DataFrame:
        if year_month not in self._month_frame_cache:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "Please install 'huggingface-hub' package: pip install huggingface-hub"
                ) from exc

            filename = f"data/ohlcv_{year_month}.parquet"
            logger.info(f"Loading month shard {filename} from {self.dataset_id}...")
            local_path = hf_hub_download(
                repo_id=self.dataset_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=self.cache_dir,
            )

            preferred_columns = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
            try:
                df = pd.read_parquet(local_path, columns=preferred_columns)
            except Exception:
                df = pd.read_parquet(local_path)

            self._month_frame_cache[year_month] = df

        return self._month_frame_cache[year_month]

    def _get_month_shard_via_dataset_configs(
        self,
        symbol: str,
        year_month: str,
        month_start: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> pd.DataFrame:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Please install 'datasets' package: pip install datasets") from exc

        # Fast path: try split-specific config first (your old behavior)
        split_name = f"{symbol}-{year_month}"
        logger.info(f"Loading {split_name} from {self.dataset_id}...")

        try:
            ds = load_dataset(
                self.dataset_id,
                name=split_name,
                cache_dir=self.cache_dir,
                trust_remote_code=False,
            )
            if "train" not in ds:
                logger.warning(f"No 'train' split in {split_name}; returning empty")
                return pd.DataFrame(columns=_COLUMNS)
            df = ds["train"].to_pandas()
            if df.empty:
                return pd.DataFrame(columns=_COLUMNS)
            df = self._normalize_schema(df)
            # Still enforce month filter for safety
            df = df[(df.index >= month_start) & (df.index < month_end)]
            return df.sort_index()
        except Exception as e:
            # Fall back to default config layout
            logger.info(
                f"Config {split_name!r} unavailable, falling back to default config "
                f"and filtering symbol/month ({e})"
            )

        # Fallback path: load default config and filter
        try:
            if self._default_train_df_cache is None:
                logger.info("Loading default HF dataset config into memory cache...")
                ds = load_dataset(
                    self.dataset_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=False,
                )
                if "train" not in ds:
                    logger.warning("No 'train' split in default dataset config; returning empty")
                    return pd.DataFrame(columns=_COLUMNS)
                self._default_train_df_cache = ds["train"].to_pandas()

            df = self._default_train_df_cache
            if df is None or df.empty:
                return pd.DataFrame(columns=_COLUMNS)

            # Filter by symbol if a symbol-like column exists
            symbol_col = self._find_symbol_column(df)
            if symbol_col is None:
                logger.warning(
                    "Could not find symbol/ticker column in default dataset; "
                    "cannot filter by symbol. Returning empty."
                )
                return pd.DataFrame(columns=_COLUMNS)

            df_filtered = df[df[symbol_col].astype(str).str.upper() == symbol.upper()]
            if df_filtered.empty:
                return pd.DataFrame(columns=_COLUMNS)

            # Normalize and month-filter
            df_filtered = self._normalize_schema(df_filtered)
            df_filtered = df_filtered[(df_filtered.index >= month_start) & (df_filtered.index < month_end)]
            return df_filtered.sort_index()

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}/{year_month} from HF: {e}")
            return pd.DataFrame(columns=_COLUMNS)

    @staticmethod
    def _month_bounds(year_month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        month_start = pd.Timestamp(f"{year_month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        return month_start, month_end

    def get_history(
        self,
        symbol: str,
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1m",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch HF data for date range by month shards.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        start : str or datetime, optional
            Start date. If None, defaults to 2020-01-01.
        end : str or datetime, optional
            End date. If None, defaults to today.
        period : str, optional
            Ignored (for DataRetriever compatibility).
        interval : str, optional
            Ignored (HF only provides 1m).

        Returns
        -------
        pd.DataFrame
            UTC-indexed DataFrame with all bars in [start, end) range.
        """
        self._check_enabled()

        start_dt = self._to_dt(start) if start else pd.Timestamp("2020-01-01", tz="UTC")
        end_dt = self._to_dt(end) if end else pd.Timestamp.now(tz="UTC")

        dfs = []
        months = self._months_in_range(start_dt, end_dt)

        for year_month in months:
            try:
                df = self.get_month_shard(symbol, year_month)
                if not df.empty:
                    # Filter to requested range
                    df_filtered = df[(df.index >= start_dt) & (df.index < end_dt)]
                    if not df_filtered.empty:
                        dfs.append(df_filtered)
            except Exception as e:
                logger.debug(f"Skipping {symbol}/{year_month}: {e}")
                continue

        return pd.concat(dfs).sort_index() if dfs else pd.DataFrame(columns=_COLUMNS)

    def get_ticker_data(
        self,
        symbols: List[str],
        *,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch for multiple symbols."""
        return {sym: self.get_history(sym, start=start, end=end, **kwargs) for sym in symbols}

    def get_current_price(self, symbol: str, **kwargs) -> dict:
        """
        Return most recent price (HF only has historical data, not real-time).

        Returns last bar from most recent available month.
        """
        self._check_enabled()

        # Try to find most recent month with data
        current = pd.Timestamp.now(tz="UTC")
        for offset in range(12):
            month_ts = current - pd.DateOffset(months=offset)
            year_month = month_ts.strftime("%Y-%m")

            df = self.get_month_shard(symbol, year_month)
            if not df.empty:
                last = df.iloc[-1]
                return {
                    "symbol": symbol,
                    "current_price": float(last.get("close", float("nan"))),
                    "timestamp": df.index[-1].isoformat(),
                    "source": "HuggingFace",
                }

        return {
            "symbol": symbol,
            "current_price": None,
            "source": "HuggingFace",
        }

    @staticmethod
    def _to_dt(ts: Union[str, datetime]) -> pd.Timestamp:
        """Convert to UTC-aware pandas Timestamp."""
        if isinstance(ts, datetime):
            return pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None else pd.Timestamp(ts).tz_convert("UTC")
        return pd.to_datetime(ts, utc=True)

    @staticmethod
    def _months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
        """Generate list of 'YYYY-MM' strings between start and end (exclusive)."""
        if start.tz is None:
            start = start.tz_localize("UTC")
        if end.tz is None:
            end = end.tz_localize("UTC")

        months = []
        current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        while current < end:
            months.append(current.strftime("%Y-%m"))
            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return months

    @staticmethod
    def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame schema to project standard.

        Handles common column name variations (Open/open, High/high, etc.)
        and ensures UTC datetime index with correct dtype.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame from HF dataset

        Returns
        -------
        pd.DataFrame
            Normalized: timestamp (UTC index), columns: open, high, low, close, volume
        """
        # Rename columns (case-insensitive match)
        mapping = {}
        for col in df.columns:
            lower_col = col.lower()
            if "timestamp" in lower_col or "time" in lower_col:
                mapping[col] = "timestamp"
            elif "open" in lower_col:
                mapping[col] = "open"
            elif "high" in lower_col:
                mapping[col] = "high"
            elif "low" in lower_col:
                mapping[col] = "low"
            elif "close" in lower_col:
                mapping[col] = "close"
            elif "volume" in lower_col:
                mapping[col] = "volume"

        df = df.rename(columns=mapping)

        # Set timestamp as index if not already
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Convert index to UTC datetime
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"

        # Ensure OHLCV columns and correct dtypes
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

        # Select only OHLCV columns in order
        available_cols = [c for c in _COLUMNS if c in df.columns]
        return df[available_cols]

    @staticmethod
    def _find_symbol_column(df: pd.DataFrame) -> Optional[str]:
        """Find best-effort symbol/ticker column name."""
        candidates = ["symbol", "ticker", "tic", "asset", "code", "instrument"]
        lower_map = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in lower_map:
                return lower_map[c]

        # Fallback fuzzy search
        for col in df.columns:
            lc = col.lower()
            if "symbol" in lc or "ticker" in lc:
                return col
        return None

