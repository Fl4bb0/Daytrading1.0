from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from kvant.benchmarking.runner import RandomTradingModel, _build_strategy_summary, _buy_and_hold_series
from kvant.models import MODEL_REGISTRY, ShallowCNNModel


class RandomTradingModelTests(unittest.TestCase):
    def test_random_trading_model_is_deterministic_for_same_seed(self) -> None:
        X = np.zeros((100, 4, 20), dtype=np.float32)
        a = RandomTradingModel(trade_probability=0.25, seed=7)
        b = RandomTradingModel(trade_probability=0.25, seed=7)

        np.testing.assert_array_equal(a.predict(X), b.predict(X))
        np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))

    def test_random_trading_model_probability_shape_and_pred_alignment(self) -> None:
        X = np.zeros((50, 3, 10), dtype=np.float32)
        model = RandomTradingModel(trade_probability=0.4, seed=3)

        pred = model.predict(X)
        proba = model.predict_proba(X)

        self.assertEqual(proba.shape, (50, 3))
        np.testing.assert_array_equal(proba.argmax(axis=1), pred)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(50), rtol=1e-6)


class ShallowCNNModelTests(unittest.TestCase):
    def test_shallow_cnn_is_registered(self) -> None:
        self.assertIs(MODEL_REGISTRY["shallow_cnn"], ShallowCNNModel)


class StrategySummaryTests(unittest.TestCase):
    def test_build_strategy_summary_aggregates_random_runs(self) -> None:
        seed_summary = pd.DataFrame(
            [
                {
                    "run": "council_meta",
                    "strategy": "council_meta",
                    "final_portfolio_net_pct": 2.0,
                    "executed_trades": 10,
                },
                {
                    "run": "random_seed_000",
                    "strategy": "random",
                    "final_portfolio_net_pct": -1.0,
                    "executed_trades": 8,
                },
                {
                    "run": "random_seed_001",
                    "strategy": "random",
                    "final_portfolio_net_pct": 1.0,
                    "executed_trades": 12,
                },
            ]
        )

        summary = _build_strategy_summary(seed_summary)
        random_row = summary[summary["strategy"] == "random"].iloc[0]

        self.assertEqual(int(random_row["n_runs"]), 2)
        self.assertAlmostEqual(float(random_row["final_portfolio_net_pct"]), 0.0)
        self.assertAlmostEqual(float(random_row["executed_trades"]), 10.0)


class BuyAndHoldSeriesTests(unittest.TestCase):
    def test_buy_and_hold_reads_month_partitioned_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir)
            month = store / "2025-02"
            month.mkdir()
            pd.DataFrame(
                {
                    "timestamp": [
                        "2025-02-03 14:30:00+00:00",
                        "2025-02-03 14:31:00+00:00",
                    ],
                    "open": [100.0, 110.0],
                    "high": [100.0, 110.0],
                    "low": [100.0, 110.0],
                    "close": [100.0, 110.0],
                    "volume": [1, 1],
                }
            ).to_csv(month / "AAPL.csv", index=False)

            series = _buy_and_hold_series(
                store_dir=store,
                tickers=["AAPL"],
                t_min=pd.Timestamp("2025-02-03 14:30:00", tz="UTC"),
                t_max=pd.Timestamp("2025-02-03 14:31:00", tz="UTC"),
            )

        self.assertIsNotNone(series)
        assert series is not None
        self.assertAlmostEqual(float(series.iloc[-1]), 10.0)


if __name__ == "__main__":
    unittest.main()
