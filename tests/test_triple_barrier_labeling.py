import unittest

import pandas as pd

from kvant.labeling.triple_barrier import triple_barrier_label


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


if __name__ == "__main__":
    unittest.main()

