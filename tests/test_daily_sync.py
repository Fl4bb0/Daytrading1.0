from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pandas as pd

from kvant.kdata.sync import DailyTickerSync
from kvant.kmarket_info.is_nyse_open import is_nyse_trading_day
from kvant.kdata.retriever import DataRetriever


class FakeRetriever:
    def __init__(self) -> None:
        self.requested_months: list[str] = []

    def get_history(self, symbol, *, start=None, end=None, period=None, interval="1m", **kwargs):
        month = kwargs.get("month")
        if month is not None:
            self.requested_months.append(str(month))
            start_day = pd.Period(month, freq="M").start_time.date()
            end_day = pd.Period(month, freq="M").end_time.date()
            return _make_intraday_df(start_day, end_day)

        start_ts = cast(pd.Timestamp, pd.Timestamp(start))
        end_ts = cast(pd.Timestamp, pd.Timestamp(end) if end is not None else start_ts + pd.Timedelta(days=1))
        start_day = _to_date(start_ts.tz_convert("America/New_York"))
        end_minus = cast(pd.Timestamp, cast(object, end_ts - pd.Timedelta(seconds=1)))
        end_day = _to_date(end_minus.tz_convert("America/New_York"))
        return _make_intraday_df(start_day, end_day)


def _to_date(ts: pd.Timestamp) -> date:
    return ts.date()


def _make_intraday_df(start_day: date, end_day: date) -> pd.DataFrame:
    rows = []
    idx = []
    day = start_day
    while day <= end_day:
        if is_nyse_trading_day(day):
            base = pd.Timestamp(day, tz="America/New_York").replace(hour=9, minute=30)
            for minute in range(6):
                ts = base + pd.Timedelta(minutes=minute)
                idx.append(ts.tz_convert("UTC"))
                rows.append({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
        day += timedelta(days=1)
    if not idx:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="timestamp"))


class DailySyncTests(unittest.TestCase):
    def test_bootstrap_and_onboard_roll_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir)
            retriever = FakeRetriever()
            sync = DailyTickerSync(store, cast(DataRetriever, cast(object, retriever)), budget_limit=100)

            now = cast(pd.Timestamp, pd.Timestamp("2026-03-13T22:00:00Z"))
            report_aapl = sync.run("AAPL", now_utc=now)
            self.assertEqual(report_aapl.mode, "bootstrap")
            self.assertGreater(report_aapl.requests_used, 0)
            self.assertIn("2026-02", retriever.requested_months)
            self.assertNotIn("2026-03", retriever.requested_months)

            old_start = pd.Timestamp(sync.state.global_start).date()
            old_end = pd.Timestamp(sync.state.global_end).date()

            report_msft = sync.run(
                "MSFT",
                now_utc=cast(pd.Timestamp, pd.Timestamp("2026-03-14T12:00:00Z")),
                roll_forward=True,
            )
            self.assertEqual(report_msft.mode, "onboard")
            self.assertIn("rolled window", " ".join(report_msft.notes))

            new_start = pd.Timestamp(sync.state.global_start).date()
            new_end = pd.Timestamp(sync.state.global_end).date()
            self.assertGreater(new_start, old_start)
            self.assertGreater(new_end, old_end)

            aapl = sync.store.load("AAPL")
            msft = sync.store.load("MSFT")
            self.assertFalse(aapl.empty)
            self.assertFalse(msft.empty)

            aapl_first = aapl.index[0].tz_convert("America/New_York").date()
            msft_first = msft.index[0].tz_convert("America/New_York").date()
            self.assertEqual(aapl_first, new_start)
            self.assertEqual(msft_first, new_start)


if __name__ == "__main__":
    unittest.main()

