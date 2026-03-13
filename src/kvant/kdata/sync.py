"""
kdata.sync — Daily quota-aware 1-minute synchronization utilities.

This module keeps all tracked tickers aligned on a shared [global_start, global_end]
window while using a HybridRetriever (Yahoo for recent/live, Alpha Vantage for
history).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, cast

import pandas as pd

from kvant.kdata.retriever import DataRetriever
from kvant.kdata.store import OHLCVStore
from kvant.kmarket_info.is_nyse_open import is_nyse_trading_day, nyse_market_close_today


@dataclass
class BudgetState:
    limit_per_day: int = 25
    used_today: int = 0
    last_reset_date: str = ""


@dataclass
class TickerState:
    first_day: str
    last_day: str
    rows: int


@dataclass
class SyncState:
    interval: str = "1m"
    global_start: Optional[str] = None
    global_end: Optional[str] = None
    budget: BudgetState = field(default_factory=BudgetState)
    tickers: Dict[str, TickerState] = field(default_factory=dict)


@dataclass
class DailySyncReport:
    symbol: str
    mode: str
    requests_used: int
    notes: List[str] = field(default_factory=list)


class DailyTickerSync:
    """Synchronize one ticker per run while keeping a globally aligned window."""

    def __init__(
        self,
        store_dir: str | Path,
        retriever: DataRetriever,
        *,
        state_file: str = "sync_state.json",
        budget_limit: int = 25,
        interval: str = "1m",
    ) -> None:
        self.store = OHLCVStore(store_dir)
        self.retriever = retriever
        self.interval = interval
        self.state_path = Path(store_dir) / state_file
        self.state = self._load_state()
        self.state.budget.limit_per_day = budget_limit

    def run(self, symbol: str, *, now_utc: Optional[pd.Timestamp] = None, roll_forward: bool = True) -> DailySyncReport:
        now = now_utc or pd.Timestamp.now(tz="UTC")
        self._reset_budget(now.date())

        if symbol in self.state.tickers:
            report = self._update_existing(symbol, now)
        elif not self.state.tickers:
            report = self._bootstrap_first_ticker(symbol, now)
        else:
            report = self._onboard_ticker(symbol, now, roll_forward=roll_forward)

        self._recompute_intersection_window()
        self._save_state()
        return report

    def remaining_requests(self) -> int:
        return max(0, self.state.budget.limit_per_day - self.state.budget.used_today)

    def _update_existing(self, symbol: str, now: pd.Timestamp) -> DailySyncReport:
        target_day = _last_complete_trading_day(now)
        used = 0
        notes: List[str] = [f"daily update target day: {target_day}"]

        existing = self.store.load(symbol)
        two_year_start = _next_trading_day(target_day - timedelta(days=730))

        # Backfill first so each daily run tends toward a full 2-year window.
        if existing.empty or _first_day(existing) > two_year_start:
            backfill_end = _last_day(existing) if not existing.empty else target_day
            backfill_df, backfill_used = self._fetch_range(
                symbol,
                two_year_start,
                backfill_end,
                max_requests=self.remaining_requests(),
                request_day=now.date(),
            )
            used += backfill_used
            if not backfill_df.empty:
                existing = _merge_keep_last(existing, backfill_df)
                notes.append(f"backfilled rows: {len(backfill_df)}")
            elif backfill_used > 0:
                notes.append("backfill requests returned no rows")

        if self.remaining_requests() <= 0:
            notes.append("no request budget remaining")
            self.store.save(symbol, existing)
            self._refresh_ticker_state(symbol, existing)
            return DailySyncReport(symbol=symbol, mode="daily", requests_used=used, notes=notes)

        fresh = self._fetch_single_day(symbol, target_day)
        used += 1
        self._consume_request(now.date())

        merged = _merge_keep_last(existing, fresh)

        if self.state.global_start is not None:
            start = _as_date(self.state.global_start)
            merged = _filter_to_day_window(merged, start_day=start)

        self.store.save(symbol, merged)
        self._refresh_ticker_state(symbol, merged)

        notes.append(f"rows after update: {len(merged)}")
        return DailySyncReport(symbol=symbol, mode="daily", requests_used=used, notes=notes)

    def _bootstrap_first_ticker(self, symbol: str, now: pd.Timestamp) -> DailySyncReport:
        end_day = _last_complete_trading_day(now)
        start_day = _next_trading_day(end_day - timedelta(days=730))
        used = 0

        df, reqs = self._fetch_range(
            symbol,
            start_day,
            end_day,
            max_requests=self.remaining_requests(),
            request_day=now.date(),
        )
        used += reqs
        if reqs == 0:
            return DailySyncReport(
                symbol=symbol,
                mode="bootstrap",
                requests_used=0,
                notes=["insufficient budget for bootstrap requests"],
            )

        df = _filter_to_day_window(df, start_day=start_day, end_day=end_day)
        if df.empty:
            return DailySyncReport(symbol=symbol, mode="bootstrap", requests_used=used, notes=["no data returned"])

        self.store.save(symbol, df)
        self._refresh_ticker_state(symbol, df)
        self.state.global_start = str(_first_day(df))
        self.state.global_end = str(_last_day(df))

        return DailySyncReport(
            symbol=symbol,
            mode="bootstrap",
            requests_used=used,
            notes=[f"window {self.state.global_start} -> {self.state.global_end}"],
        )

    def _onboard_ticker(self, symbol: str, now: pd.Timestamp, *, roll_forward: bool) -> DailySyncReport:
        if self.state.global_start is None or self.state.global_end is None:
            raise ValueError("global window is missing; cannot onboard ticker")

        start_day = _as_date(self.state.global_start)
        end_day = _as_date(self.state.global_end)

        df, used = self._fetch_range(
            symbol,
            start_day,
            end_day,
            max_requests=self.remaining_requests(),
            request_day=now.date(),
        )
        df = _filter_to_day_window(df, start_day=start_day, end_day=end_day)
        notes: List[str] = [f"onboard window {start_day} -> {end_day}"]

        if df.empty:
            return DailySyncReport(symbol=symbol, mode="onboard", requests_used=used, notes=notes + ["no data returned"])

        self.store.save(symbol, df)
        self._refresh_ticker_state(symbol, df)

        if not roll_forward:
            return DailySyncReport(symbol=symbol, mode="onboard", requests_used=used, notes=notes)

        all_symbols = sorted(self.state.tickers.keys())
        required = len(all_symbols)
        if self.remaining_requests() < required:
            notes.append("not enough request budget to roll forward shared window")
            return DailySyncReport(symbol=symbol, mode="onboard", requests_used=used, notes=notes)

        next_start = _next_trading_day(start_day)
        next_end = _next_trading_day(end_day)

        for sym in all_symbols:
            day_df = self._fetch_single_day(sym, next_end)
            self._consume_request(now.date())
            used += 1

            current = self.store.load(sym)
            current = _filter_to_day_window(current, start_day=next_start)
            current = _merge_keep_last(current, day_df)
            current = _filter_to_day_window(current, start_day=next_start, end_day=next_end)
            self.store.save(sym, current)
            self._refresh_ticker_state(sym, current)

        self.state.global_start = str(next_start)
        self.state.global_end = str(next_end)
        notes.append(f"rolled window to {next_start} -> {next_end}")
        return DailySyncReport(symbol=symbol, mode="onboard", requests_used=used, notes=notes)

    def _fetch_single_day(self, symbol: str, day: date) -> pd.DataFrame:
        start = pd.Timestamp(day, tz="America/New_York").replace(hour=9, minute=30).tz_convert("UTC")
        end = pd.Timestamp(day + timedelta(days=1), tz="America/New_York").replace(hour=0, minute=0).tz_convert("UTC")
        df = self.retriever.get_history(
            symbol,
            start=cast(datetime, start.to_pydatetime()),
            end=cast(datetime, end.to_pydatetime()),
            interval=self.interval,
        )
        if df.empty:
            return df
        return _filter_to_day_window(_normalise(df), start_day=day, end_day=day)

    def _fetch_range(
        self,
        symbol: str,
        start_day: date,
        end_day: date,
        *,
        max_requests: int,
        request_day: date,
    ) -> tuple[pd.DataFrame, int]:
        completed_end_day = _completed_month_end_before(request_day)
        fetch_end_day = min(end_day, completed_end_day)
        if start_day > fetch_end_day:
            return pd.DataFrame(), 0

        months = _months_between(start_day, fetch_end_day)
        used = 0
        chunks: List[pd.DataFrame] = []
        for month in months:
            if used >= max_requests:
                break
            part = self.retriever.get_history(symbol, interval=self.interval, month=month)
            self._consume_request(request_day)
            used += 1
            if not part.empty:
                chunks.append(_normalise(part))

        if chunks:
            merged = pd.concat(chunks).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            return merged, used

        # Fallback: one explicit range query helps when provider key labels differ
        # or month-slice responses are empty for the symbol.
        if used < max_requests and months:
            start_dt = pd.Timestamp(start_day, tz="America/New_York").replace(hour=9, minute=30).tz_convert("UTC")
            end_dt = pd.Timestamp(fetch_end_day + timedelta(days=1), tz="America/New_York").replace(hour=0, minute=0).tz_convert("UTC")
            whole = self.retriever.get_history(
                symbol,
                start=cast(datetime, start_dt.to_pydatetime()),
                end=cast(datetime, end_dt.to_pydatetime()),
                interval=self.interval,
            )
            self._consume_request(request_day)
            used += 1
            if not whole.empty:
                return _normalise(whole), used

        return pd.DataFrame(), used

    def _consume_request(self, today: date) -> None:
        self._reset_budget(today)
        self.state.budget.used_today += 1

    def _reset_budget(self, today: date) -> None:
        today_str = str(today)
        if self.state.budget.last_reset_date != today_str:
            self.state.budget.last_reset_date = today_str
            self.state.budget.used_today = 0

    def _refresh_ticker_state(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.state.tickers[symbol] = TickerState(
            first_day=str(_first_day(df)),
            last_day=str(_last_day(df)),
            rows=len(df),
        )

    def _recompute_intersection_window(self) -> None:
        if not self.state.tickers:
            self.state.global_start = None
            self.state.global_end = None
            return
        starts = [pd.Timestamp(v.first_day).date() for v in self.state.tickers.values()]
        ends = [pd.Timestamp(v.last_day).date() for v in self.state.tickers.values()]
        self.state.global_start = str(max(starts))
        self.state.global_end = str(min(ends))

    def _load_state(self) -> SyncState:
        if not self.state_path.exists():
            return SyncState(interval=self.interval)

        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        budget_raw = data.get("budget", {})
        tickers_raw = data.get("tickers", {})
        return SyncState(
            interval=data.get("interval", self.interval),
            global_start=data.get("global_start"),
            global_end=data.get("global_end"),
            budget=BudgetState(
                limit_per_day=int(budget_raw.get("limit_per_day", 25)),
                used_today=int(budget_raw.get("used_today", 0)),
                last_reset_date=str(budget_raw.get("last_reset_date", "")),
            ),
            tickers={
                sym: TickerState(
                    first_day=str(meta.get("first_day", "")),
                    last_day=str(meta.get("last_day", "")),
                    rows=int(meta.get("rows", 0)),
                )
                for sym, meta in tickers_raw.items()
            },
        )

    def _save_state(self) -> None:
        payload = {
            "interval": self.state.interval,
            "global_start": self.state.global_start,
            "global_end": self.state.global_end,
            "budget": {
                "limit_per_day": self.state.budget.limit_per_day,
                "used_today": self.state.budget.used_today,
                "last_reset_date": self.state.budget.last_reset_date,
            },
            "tickers": {
                sym: {
                    "first_day": meta.first_day,
                    "last_day": meta.last_day,
                    "rows": meta.rows,
                }
                for sym, meta in sorted(self.state.tickers.items())
            },
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "timestamp"
    out.columns = [c.lower() for c in out.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[keep]
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _merge_keep_last(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return _normalise(right) if not right.empty else left
    if right.empty:
        return _normalise(left)
    merged = pd.concat([_normalise(left), _normalise(right)]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _first_day(df: pd.DataFrame) -> date:
    return df.index[0].tz_convert("America/New_York").date()


def _last_day(df: pd.DataFrame) -> date:
    return df.index[-1].tz_convert("America/New_York").date()


def _filter_to_day_window(df: pd.DataFrame, *, start_day: date, end_day: Optional[date] = None) -> pd.DataFrame:
    if df.empty:
        return df
    ny_dates = df.index.tz_convert("America/New_York").date
    mask = ny_dates >= start_day
    if end_day is not None:
        mask &= ny_dates <= end_day
    return df.loc[mask]


def _months_between(start_day: date, end_day: date) -> List[str]:
    start = pd.Period(start_day, freq="M")
    end = pd.Period(end_day, freq="M")
    months: List[str] = []
    cur = start
    while cur <= end:
        months.append(str(cur))
        cur += 1
    return months


def _completed_month_end_before(day: date) -> date:
    first_of_month = day.replace(day=1)
    return first_of_month - timedelta(days=1)


def _as_date(value: str) -> date:
    return cast(date, pd.Timestamp(value).date())


def _last_complete_trading_day(now_utc: pd.Timestamp) -> date:
    now_utc = now_utc.tz_convert("UTC") if now_utc.tz is not None else now_utc.tz_localize("UTC")
    today_ny = now_utc.tz_convert("America/New_York").date()
    close_ts = nyse_market_close_today(now_utc)
    if close_ts is not None and now_utc >= close_ts and is_nyse_trading_day(today_ny):
        return today_ny
    return _previous_trading_day(today_ny - timedelta(days=1))


def _previous_trading_day(day: date) -> date:
    d = day
    while not is_nyse_trading_day(d):
        d -= timedelta(days=1)
    return d


def _next_trading_day(day: date) -> date:
    d = day + timedelta(days=1)
    while not is_nyse_trading_day(d):
        d += timedelta(days=1)
    return d

