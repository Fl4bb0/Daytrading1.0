import unittest
import sys
import types

import pandas as pd

if "pandas_market_calendars" not in sys.modules:
    fake_module = types.ModuleType("pandas_market_calendars")

    class _FakeCalendar:
        def schedule(self, start_date, end_date):
            idx = pd.DatetimeIndex([pd.Timestamp("2025-01-02")])
            return pd.DataFrame(
                {
                    "market_open": [pd.Timestamp("2025-01-02 14:30:00", tz="UTC")],
                    "market_close": [pd.Timestamp("2025-01-02 21:00:00", tz="UTC")],
                },
                index=idx,
            )

    fake_module.get_calendar = lambda name: _FakeCalendar()
    sys.modules["pandas_market_calendars"] = fake_module

from kvant.labeling.triple_barrier import TripleBarrierLabeler, triple_barrier_label


class TripleBarrierCausalityTests(unittest.TestCase):
    def test_same_bar_spike_does_not_set_label(self):
        # Entry bar crosses both barriers, but future bars do not.
        idx = pd.date_range("2025-01-02 14:45:00", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [200.0, 100.0, 100.0],
                "low": [1.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
            },
            index=idx,
        )

        res = triple_barrier_label(df, time_start=idx[0], width=2, height=0.01)

        self.assertIsNotNone(res)
        self.assertEqual(res.label, 1)
        self.assertEqual(res.bar_open_time, idx[0])
        self.assertEqual(res.bar_close_time, idx[2])

    def test_brokerage_fee_widens_buy_barrier_by_round_trip_cost(self):
        idx = pd.date_range("2025-01-02 14:45:00", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 101.15, 100.10],
                "low": [100.0, 99.95, 100.00],
                "close": [100.0, 100.10, 100.05],
            },
            index=idx,
        )

        labeler = TripleBarrierLabeler(
            width_minutes=2,
            height=0.01,
            brokerage_fee=0.001,
            show_progress=False,
        )

        labels, metadata = labeler.transform(df)

        self.assertEqual(int(labels[0]), 1)
        self.assertIsNotNone(metadata[0])
        self.assertAlmostEqual(float(metadata[0]["height_used"]), 0.012)

    def test_brokerage_fee_widens_short_barrier_by_round_trip_cost(self):
        idx = pd.date_range("2025-01-02 14:45:00", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 100.05, 100.00],
                "low": [100.0, 98.95, 99.90],
                "close": [100.0, 99.95, 99.98],
            },
            index=idx,
        )

        labeler = TripleBarrierLabeler(
            width_minutes=2,
            height=0.01,
            brokerage_fee=0.001,
            show_progress=False,
        )

        labels, metadata = labeler.transform(df)

        self.assertEqual(int(labels[0]), 1)
        self.assertIsNotNone(metadata[0])
        self.assertAlmostEqual(float(metadata[0]["height_used"]), 0.012)


if __name__ == "__main__":
    unittest.main()

