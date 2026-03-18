from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from kvant.kdata.store import _target_trading_days


class TargetTradingDaysTests(unittest.TestCase):
    def test_midweek_before_close_includes_previous_friday(self) -> None:
        now = pd.Timestamp("2026-03-18T18:00:00Z")  # Wed, before NYSE close
        got = _target_trading_days(now)
        expected = [
            date(2026, 3, 11),
            date(2026, 3, 12),
            date(2026, 3, 13),
            date(2026, 3, 16),
            date(2026, 3, 17),
        ]
        self.assertEqual(got, expected)

    def test_midweek_after_close_includes_today(self) -> None:
        now = pd.Timestamp("2026-03-18T22:30:00Z")  # Wed, after NYSE close
        got = _target_trading_days(now)
        expected = [
            date(2026, 3, 12),
            date(2026, 3, 13),
            date(2026, 3, 16),
            date(2026, 3, 17),
            date(2026, 3, 18),
        ]
        self.assertEqual(got, expected)

    def test_holiday_within_window_is_excluded(self) -> None:
        # Tue after close; rolling window includes MLK Day (2026-01-19), which should be excluded.
        now = pd.Timestamp("2026-01-20T22:00:00Z")
        got = _target_trading_days(now)
        self.assertNotIn(date(2026, 1, 19), got)
        self.assertIn(date(2026, 1, 20), got)


if __name__ == "__main__":
    unittest.main()

