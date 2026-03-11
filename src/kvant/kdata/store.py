"""
kdata.store — Local CSV store for accumulated 1-minute OHLCV data.

Yahoo Finance only provides 7 days of 1-minute history. This module
fetches Mon–Fri of the target week day-by-day, appends new bars to
per-ticker CSVs, and maintains a status TOML that tracks:

  - the last day successfully retrieved per ticker
  - any trading days that are missing (not yet fetched or outside
    Yahoo's 7-day window at time of fetch), marked "unavailable"

Layout on disk
--------------
  <store_dir>/
    AAPL.csv
    MSFT.csv
    status.toml          ← per-ticker retrieval status

status.toml format
------------------
  [AAPL]
  last_retrieved = "2026-03-13"
  missing = ["2026-03-09", "2026-03-10"]   # permanently unavailable

  [AAPL.days]
  "2026-03-10" = "ok"
  "2026-03-11" = "ok"
  "2026-03-12" = "unavailable"   # was outside Yahoo window when fetched
  "2026-03-13" = "ok"

Rules
-----
- Run on Sat/Sun        → fetch full Mon–Fri of the week just ended.
- Run on Fri after close → same: fetch full Mon–Fri of current week.
- Run on Mon–Thu (or Fri before close) → fetch Mon up to last *complete*
  trading day (today excluded unless market has closed).
- Any trading day that Yahoo returns no bars for is written as
  "unavailable" in status.toml (it was likely outside the 7-day window).
- On subsequent runs, days already marked "ok" are skipped; days marked
  "unavailable" are retried if they are still within Yahoo's window.

OHLCVStore(store_dir)
  .weekly_update(symbols, retriever)  → WeeklyUpdateReport
  .load(symbol)                       → pd.DataFrame  (UTC-indexed)
  .load_all(symbols)                  → Dict[str, pd.DataFrame]
  .save(symbol, df)                   → None
  .read_status()                      → dict   (raw TOML)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, cast

import pandas as pd

from kvant.kdata.retriever import DataRetriever
from kvant.kmarket_info.is_nyse_open import (
    is_nyse_trading_day,
    nyse_market_close_today,
)

logger = logging.getLogger(__name__)

_COLUMNS    = ["open", "high", "low", "close", "volume"]
_INDEX_NAME = "timestamp"
_STATUS_FILE = "status.toml"


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    day: date
    new_rows: int
    status: str   # "ok" | "unavailable" | "skipped" | "error:<msg>"


@dataclass
class TickerUpdateResult:
    symbol: str
    day_results: List[DayResult] = field(default_factory=list)
    total_rows: int = 0

    @property
    def new_rows(self) -> int:
        return sum(d.new_rows for d in self.day_results)

    def __str__(self) -> str:
        lines = [f"  {self.symbol}  (+{self.new_rows} new rows, {self.total_rows} total)"]
        for d in self.day_results:
            lines.append(f"    {d.day}  {d.status}  +{d.new_rows} rows")
        return "\n".join(lines)


@dataclass
class WeeklyUpdateReport:
    results: List[TickerUpdateResult] = field(default_factory=list)
    target_days: List[date] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "OHLCVStore weekly update report:",
            f"  Target days : {[str(d) for d in self.target_days]}",
        ]
        for r in self.results:
            lines.append(str(r))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OHLCVStore
# ---------------------------------------------------------------------------

class OHLCVStore:
    """
    Persistent local store for per-ticker 1-minute OHLCV CSVs.

    Parameters
    ----------
    store_dir : str | Path
        Directory where CSVs and status.toml are kept.
        Created automatically if absent.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def weekly_update(
        self,
        symbols: List[str],
        retriever: DataRetriever,
        *,
        now_utc: Optional[pd.Timestamp] = None,
        interval: str = "1m",
    ) -> WeeklyUpdateReport:
        """
        Fetch Mon–Fri bars for the appropriate week and append to CSVs.

        *now_utc* defaults to the current UTC time; pass an explicit value
        for testing or back-filling.

        Which days are targeted
        -----------------------
        - Sat / Sun, or Fri after NYSE close → full Mon–Fri of that week.
        - Mon–Thu, or Fri before/at NYSE close → Mon up to yesterday
          (last complete trading day), plus today if market has closed.
        Days already stored as "ok" are skipped. Days marked "unavailable"
        are retried if they are within Yahoo's 7-day rolling window.
        """
        if now_utc is None:
            now_utc = pd.Timestamp.now(tz="UTC")

        target_days = _target_trading_days(now_utc)
        report = WeeklyUpdateReport(target_days=target_days)
        status = self._load_status()

        for sym in symbols:
            result = self._update_ticker(
                sym, retriever, target_days, now_utc, interval, status
            )
            report.results.append(result)

        self._save_status(status)
        return report

    def load(self, symbol: str) -> pd.DataFrame:
        """Load the full stored history for *symbol* as a UTC-indexed DataFrame."""
        path = self._csv_path(symbol)
        if not path.exists():
            return _empty_df()
        df = pd.read_csv(path, index_col=_INDEX_NAME)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = cast(pd.DatetimeIndex, df.index)
        df.index.name = _INDEX_NAME
        return df.sort_index()

    def load_all(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Return ``{ticker: DataFrame}`` for every symbol in *symbols*."""
        return {sym: self.load(sym) for sym in symbols}

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Overwrite the CSV for *symbol*. Deduplicates and sorts before writing."""
        _normalise(df).to_csv(self._csv_path(symbol))

    def read_status(self) -> dict:
        """Return the raw parsed status TOML as a dict."""
        return self._load_status()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _csv_path(self, symbol: str) -> Path:
        return self.store_dir / f"{symbol}.csv"

    def _status_path(self) -> Path:
        return self.store_dir / _STATUS_FILE

    def _load_status(self) -> dict:
        path = self._status_path()
        if not path.exists():
            return {}
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _save_status(self, status: dict) -> None:
        lines: List[str] = []
        for sym, data in sorted(status.items()):
            lines.append(f"[{sym}]")
            last = data.get("last_retrieved", "")
            lines.append(f'last_retrieved = "{last}"')
            missing = data.get("missing", [])
            missing_str = ", ".join(f'"{d}"' for d in missing)
            lines.append(f"missing = [{missing_str}]")
            lines.append("")
            days = data.get("days", {})
            if days:
                lines.append(f"[{sym}.days]")
                for d, s in sorted(days.items()):
                    lines.append(f'"{d}" = "{s}"')
                lines.append("")
        self._status_path().write_text("\n".join(lines), encoding="utf-8")

    def _update_ticker(
        self,
        symbol: str,
        retriever: DataRetriever,
        target_days: List[date],
        now_utc: pd.Timestamp,
        interval: str,
        status: dict,
    ) -> TickerUpdateResult:
        result = TickerUpdateResult(symbol=symbol)
        ticker_status = status.setdefault(symbol, {"last_retrieved": "", "missing": [], "days": {}})
        days_status: dict = ticker_status.setdefault("days", {})
        existing = self.load(symbol)
        yahoo_cutoff = (now_utc - pd.Timedelta(days=7)).date()

        for day in target_days:
            day_str = str(day)
            current = days_status.get(day_str)

            # Skip days already successfully fetched
            if current == "ok":
                logger.debug("%s %s already ok, skipping", symbol, day_str)
                day_results_entry = DayResult(day=day, new_rows=0, status="skipped")
                result.day_results.append(day_results_entry)
                continue

            # If previously unavailable, only retry if within Yahoo's window
            if current == "unavailable" and day < yahoo_cutoff:
                day_results_entry = DayResult(day=day, new_rows=0, status="unavailable")
                result.day_results.append(day_results_entry)
                continue

            # Fetch the full day: start = day 09:30 ET, end = next day 00:00 ET
            day_start = pd.Timestamp(day, tz="America/New_York").replace(hour=9, minute=30)
            day_end   = pd.Timestamp(day + timedelta(days=1), tz="America/New_York").replace(hour=0, minute=0)

            try:
                fresh = retriever.get_history(
                    symbol,
                    start=day_start.tz_convert("UTC"),
                    end=day_end.tz_convert("UTC"),
                    interval=interval,
                )
            except Exception as exc:
                msg = str(exc)
                logger.warning("%s %s fetch error: %s", symbol, day_str, msg)
                days_status[day_str] = f"error"
                result.day_results.append(DayResult(day=day, new_rows=0, status=f"error:{msg}"))
                continue

            if fresh.empty:
                logger.info("%s %s: no data returned (unavailable)", symbol, day_str)
                days_status[day_str] = "unavailable"
                if day_str not in ticker_status["missing"]:
                    ticker_status["missing"].append(day_str)
                result.day_results.append(DayResult(day=day, new_rows=0, status="unavailable"))
                continue

            fresh = _normalise(fresh)
            # Keep only bars that belong to this calendar day (NY time)
            fresh = fresh[fresh.index.tz_convert("America/New_York").date == day]

            if existing.empty:
                new_rows_df = fresh
            else:
                new_rows_df = fresh.loc[~fresh.index.isin(existing.index)]

            if not new_rows_df.empty:
                existing = pd.concat([existing, new_rows_df]).sort_index()

            new_count = len(new_rows_df)
            days_status[day_str] = "ok"
            # Remove from missing list if it was there
            ticker_status["missing"] = [m for m in ticker_status["missing"] if m != day_str]
            ticker_status["last_retrieved"] = day_str
            result.day_results.append(DayResult(day=day, new_rows=new_count, status="ok"))
            logger.info("%s %s: +%d rows", symbol, day_str, new_count)

        self.save(symbol, existing)
        result.total_rows = len(self.load(symbol))
        return result


# ---------------------------------------------------------------------------
# Day-selection logic
# ---------------------------------------------------------------------------

def _target_trading_days(now_utc: pd.Timestamp) -> List[date]:
    """
    Return the list of NYSE trading days to target for this run.

    Sat / Sun  → full Mon–Fri of the week just ended.
    Fri after close, or Sat/Sun → full Mon–Fri of that week.
    Mon–Thu, or Fri before/at close → Mon of current week up to and
      including today only if NYSE has already closed; otherwise yesterday.
    Also includes any earlier days in the same Mon–Fri window that are
    trading days (handles holidays in the middle of the week).
    """
    ny_now   = now_utc.tz_convert("America/New_York")
    weekday  = ny_now.weekday()   # 0=Mon … 6=Sun
    today    = ny_now.date()

    # Determine which Monday to anchor on
    if weekday == 5:   # Saturday → week just ended
        monday = today - timedelta(days=5)
    elif weekday == 6: # Sunday → week just ended
        monday = today - timedelta(days=6)
    else:
        monday = today - timedelta(days=weekday)  # this week's Monday

    # Determine the last day to include
    if weekday in (5, 6):
        # Weekend → include full Mon–Fri
        last_day = monday + timedelta(days=4)
    else:
        # Weekday: include today only if market has already closed
        close_ts = nyse_market_close_today(now_utc)
        if close_ts is not None and now_utc >= close_ts:
            last_day = today
        else:
            # Market still open or not a trading day: stop at yesterday
            last_day = today - timedelta(days=1)

    # Collect all NYSE trading days in [monday, last_day]
    days: List[date] = []
    d = monday
    while d <= last_day:
        if is_nyse_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Private DataFrame helpers
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    dti = cast(pd.DatetimeIndex, df.index)
    if dti.tz is None:
        df.index = dti.tz_localize("UTC")
    else:
        df.index = dti.tz_convert("UTC")
    df.index.name = _INDEX_NAME
    df.columns = [c.lower() for c in df.columns]
    present = [c for c in _COLUMNS if c in df.columns]
    df = df[present]
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _empty_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=_COLUMNS)
    df.index = pd.DatetimeIndex([], tz="UTC", name=_INDEX_NAME)
    return df
