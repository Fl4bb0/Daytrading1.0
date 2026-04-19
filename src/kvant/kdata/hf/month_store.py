"""
kdata.month_store — Month-partitioned OHLCV storage with append-only semantics.

Storage structure: data/1m/{YYYY-MM}/{ticker}.csv

Append-only guarantees:
  - Existing rows never overwritten
  - Incoming duplicates skipped
  - First write always wins
  - All rows kept sorted by timestamp

MonthPartitionedStore(root_dir)
  .get_month_dir(year_month)              → Path
  .get_month_path(symbol, year_month)     → Path
  .ensure_month_dir(year_month)           → Path
  .load_month(symbol, year_month)         → pd.DataFrame
  .load_range(symbols, start_month, end_month) → Dict[str, pd.DataFrame]
  .append_month(symbol, year_month, df)   → Dict with stats
  .list_months()                          → List[str]
  .list_symbols_in_month(year_month)      → List[str]
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = ["open", "high", "low", "close", "volume"]
_INDEX_NAME = "timestamp"


def _is_valid_month(s: str) -> bool:
    """Check if string is valid YYYY-MM format."""
    if not isinstance(s, str) or len(s) != 7:
        return False
    try:
        datetime.strptime(s, "%Y-%m")
        return True
    except ValueError:
        return False


def _month_range(start_month: str, end_month: str) -> List[str]:
    """Generate list of 'YYYY-MM' strings from start_month to end_month (inclusive)."""
    if not _is_valid_month(start_month) or not _is_valid_month(end_month):
        raise ValueError(f"Invalid month format. Use YYYY-MM. Got: {start_month}, {end_month}")

    months = []
    start = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")

    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        # Next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return months


class MonthPartitionedStore:
    """
    Persistent local store for per-ticker 1-minute OHLCV CSVs, partitioned by month.

    Parameters
    ----------
    root_dir : str | Path
        Base directory for month partitions. Created automatically if absent.
        Example: "data/1m" → will contain "data/1m/2025-01/", "data/1m/2025-02/", etc.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"MonthPartitionedStore initialized at {self.root_dir}")

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------

    def get_month_dir(self, year_month: str) -> Path:
        """Get directory path for month (e.g., 'data/1m/2025-03')."""
        if not _is_valid_month(year_month):
            raise ValueError(f"Invalid month format. Use YYYY-MM. Got: {year_month}")
        return self.root_dir / year_month

    def get_month_path(self, symbol: str, year_month: str) -> Path:
        """Get full path to month CSV for symbol."""
        return self.get_month_dir(year_month) / f"{symbol}.csv"

    def ensure_month_dir(self, year_month: str) -> Path:
        """Create month directory if absent. Returns directory path."""
        d = self.get_month_dir(year_month)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_month(self, symbol: str, year_month: str) -> pd.DataFrame:
        """
        Load all 1m bars for (symbol, month).

        Returns empty DataFrame if file doesn't exist.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g., 'AAPL')
        year_month : str
            Month in 'YYYY-MM' format (e.g., '2025-03')

        Returns
        -------
        pd.DataFrame
            UTC-indexed DataFrame with columns: open, high, low, close, volume
            Sorted by timestamp (ascending).
        """
        path = self.get_month_path(symbol, year_month)

        if not path.exists():
            return pd.DataFrame(columns=_COLUMNS)

        df = pd.read_csv(path, index_col=_INDEX_NAME)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.as_unit("us")  # Normalize to microseconds
        df = df[_COLUMNS].astype({"volume": int})  # Ensure correct dtypes
        return df.sort_index()

    def load_range(
        self,
        symbols: List[str],
        start_month: str,
        end_month: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load and concatenate all months in range for each symbol.

        Parameters
        ----------
        symbols : List[str]
            Ticker symbols to load
        start_month : str
            Start month in 'YYYY-MM' format (inclusive)
        end_month : str
            End month in 'YYYY-MM' format (inclusive)

        Returns
        -------
        Dict[str, pd.DataFrame]
            {symbol: concatenated_dataframe} for each symbol.
            Each DataFrame spans the entire month range.
        """
        result = {}
        months = _month_range(start_month, end_month)

        for sym in symbols:
            dfs = []
            for ym in months:
                df = self.load_month(sym, ym)
                if not df.empty:
                    dfs.append(df)

            result[sym] = pd.concat(dfs).sort_index() if dfs else pd.DataFrame(columns=_COLUMNS)

        return result

    # ------------------------------------------------------------------
    # Appending (append-only, duplicate-skip)
    # ------------------------------------------------------------------

    def append_month(
        self,
        symbol: str,
        year_month: str,
        df: pd.DataFrame,
        skip_existing: bool = True,
    ) -> Dict:
        """
        Append bars to month CSV for symbol with append-only, duplicate-skip semantics.

        Guarantees:
          - Existing rows never overwritten
          - Incoming rows with timestamps in existing set are skipped
          - Result is deduplicated (first occurrence kept) and sorted

        Parameters
        ----------
        symbol : str
            Ticker symbol
        year_month : str
            Month in 'YYYY-MM' format
        df : pd.DataFrame
            DataFrame with UTC-indexed rows (timestamp index, OHLCV columns)
        skip_existing : bool
            If True (default), skip incoming rows whose timestamps exist in current month.
            If False, keep both (not recommended; will be deduplicated afterward).

        Returns
        -------
        Dict
            {
              "appended": int,  # Number of new rows written
              "skipped": int,   # Number of duplicates skipped
              "total": int,     # Total rows in month after operation
            }
        """
        path = self.get_month_path(symbol, year_month)
        self.ensure_month_dir(year_month)

        # Normalize incoming DataFrame
        df_in = df.copy()
        if df_in.index.name != _INDEX_NAME:
            if isinstance(df_in.index, pd.DatetimeIndex):
                df_in.index.name = _INDEX_NAME
            else:
                raise ValueError(
                    f"Expected DatetimeIndex for {symbol}/{year_month}, got {type(df_in.index)}"
                )
        df_in = df_in[_COLUMNS].astype({"volume": int})

        # Load existing
        existing = self.load_month(symbol, year_month)

        # Deduplication logic
        if skip_existing and not existing.empty:
            existing_ts = set(existing.index)
            df_new = df_in[~df_in.index.isin(existing_ts)]
            skipped = len(df_in) - len(df_new)
        else:
            df_new = df_in
            skipped = 0

        # Combine
        combined = pd.concat([existing, df_new]) if not existing.empty else df_new

        # Deduplicate globally (keep first), then sort
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        # Write CSV
        combined.to_csv(path)

        appended = len(df_new)
        total = len(combined)

        result = {
            "appended": appended,
            "skipped": skipped,
            "total": total,
        }

        logger.debug(
            f"append_month({symbol}, {year_month}): appended={appended}, skipped={skipped}, total={total}"
        )

        return result

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_months(self) -> List[str]:
        """
        List all month directories present in root.

        Returns
        -------
        List[str]
            Sorted list of month strings in YYYY-MM format (e.g., ['2025-01', '2025-02', ...])
        """
        months = []
        for d in self.root_dir.iterdir():
            if d.is_dir() and _is_valid_month(d.name):
                months.append(d.name)
        return sorted(months)

    def list_symbols_in_month(self, year_month: str) -> List[str]:
        """
        List all ticker symbols that have data in a month.

        Returns
        -------
        List[str]
            Sorted list of symbol strings (e.g., ['AAPL', 'MSFT', ...])
        """
        month_dir = self.get_month_dir(year_month)
        if not month_dir.exists():
            return []
        return sorted([f.stem for f in month_dir.glob("*.csv")])

    # ------------------------------------------------------------------
    # Legacy compatibility: migrate from flat CSV layout
    # ------------------------------------------------------------------

    def migrate_from_flat(
        self,
        old_store_dir: str | Path,
        year_month: str,
        symbols: Optional[List[str]] = None,
    ) -> Dict:
        """
        Migrate flat CSV files from old layout to partitioned layout for a specific month.

        This is a helper for one-time migration from data/1m/{ticker}.csv to
        data/1m/{YYYY-MM}/{ticker}.csv layout.

        Parameters
        ----------
        old_store_dir : str | Path
            Directory containing old flat CSV files (e.g., "data/1m_old")
        year_month : str
            Month to migrate data for (e.g., "2025-03")
        symbols : List[str], optional
            Symbols to migrate. If None, migrates all .csv files in old_store_dir.

        Returns
        -------
        Dict
            {symbol: result_dict} for each migrated symbol
        """
        old_dir = Path(old_store_dir)
        if not old_dir.exists():
            raise ValueError(f"Old store directory not found: {old_dir}")

        if symbols is None:
            symbols = [f.stem for f in old_dir.glob("*.csv")]

        results = {}
        for sym in symbols:
            old_path = old_dir / f"{sym}.csv"
            if not old_path.exists():
                results[sym] = {"status": "not_found"}
                continue

            try:
                df = pd.read_csv(old_path, index_col=_INDEX_NAME)
                df.index = pd.to_datetime(df.index, utc=True)

                # Filter to month
                start = pd.Timestamp(f"{year_month}-01", tz="UTC")
                end = pd.Timestamp(year_month, tz="UTC") + pd.DateOffset(months=1)
                df_month = df[(df.index >= start) & (df.index < end)]

                if not df_month.empty:
                    report = self.append_month(sym, year_month, df_month, skip_existing=False)
                    results[sym] = {"status": "migrated", **report}
                else:
                    results[sym] = {"status": "no_data_for_month"}
            except Exception as e:
                results[sym] = {"status": "error", "error": str(e)}

        return results
