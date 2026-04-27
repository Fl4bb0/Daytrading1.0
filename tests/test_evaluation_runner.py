from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kvant.evaluation.runner import (
    _apply_meta_score_thresholds,
    _apply_short_execution_policy,
    _save_equity_curve,
)


class ExecutionPriorityTests(unittest.TestCase):
    def test_model_confidence_prioritizes_same_timestamp_trade(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:30:00",
                    ]
                ),
                "ticker": ["LOW", "HIGH"],
                "y_pred": [2, 2],
                "pnl_fraction": [0.01, 0.02],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 14:50:00",
                        "2025-01-02 14:50:00",
                    ]
                ),
                "prob_BUY": [0.55, 0.95],
                "prob_SHORT": [0.10, 0.02],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "equity_curve.csv"
            _save_equity_curve(
                pred_df,
                out_path,
                fee=0.0,
                n_pools=1,
                execution_priority="model_confidence",
            )
            eq_df = pd.read_csv(out_path)

        self.assertEqual(eq_df.loc[0, "ticker"], "HIGH")
        self.assertFalse(bool(eq_df.loc[0, "skipped"]))
        self.assertTrue(bool(eq_df.loc[1, "skipped"]))
        self.assertAlmostEqual(float(eq_df.loc[1, "cumulative_portfolio_pnl_pct"]), 2.0)

    def test_first_seen_preserves_existing_order(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:30:00",
                    ]
                ),
                "ticker": ["LOW", "HIGH"],
                "y_pred": [2, 2],
                "pnl_fraction": [0.01, 0.02],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 14:50:00",
                        "2025-01-02 14:50:00",
                    ]
                ),
                "prob_BUY": [0.55, 0.95],
                "prob_SHORT": [0.10, 0.02],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "equity_curve.csv"
            _save_equity_curve(
                pred_df,
                out_path,
                fee=0.0,
                n_pools=1,
                execution_priority="first_seen",
            )
            eq_df = pd.read_csv(out_path)

        self.assertEqual(eq_df.loc[0, "ticker"], "LOW")
        self.assertFalse(bool(eq_df.loc[0, "skipped"]))
        self.assertTrue(bool(eq_df.loc[1, "skipped"]))
        self.assertAlmostEqual(float(eq_df.loc[1, "cumulative_portfolio_pnl_pct"]), 1.0)

    def test_meta_score_prioritizes_same_timestamp_trade(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:30:00",
                    ]
                ),
                "ticker": ["LOW", "HIGH"],
                "y_pred": [2, 2],
                "pnl_fraction": [0.01, 0.02],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 14:50:00",
                        "2025-01-02 14:50:00",
                    ]
                ),
                "meta_score": [0.10, 0.25],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "equity_curve.csv"
            _save_equity_curve(
                pred_df,
                out_path,
                fee=0.0,
                n_pools=1,
                execution_priority="meta_score",
            )
            eq_df = pd.read_csv(out_path)

        self.assertEqual(eq_df.loc[0, "ticker"], "HIGH")
        self.assertFalse(bool(eq_df.loc[0, "skipped"]))
        self.assertTrue(bool(eq_df.loc[1, "skipped"]))
        self.assertAlmostEqual(float(eq_df.loc[1, "cumulative_portfolio_pnl_pct"]), 2.0)

    def test_top_k_per_timestamp_limits_candidates(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:30:00",
                    ]
                ),
                "ticker": ["LOW", "MID", "HIGH"],
                "y_pred": [2, 2, 2],
                "pnl_fraction": [0.01, 0.02, 0.03],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 14:50:00",
                        "2025-01-02 14:50:00",
                        "2025-01-02 14:50:00",
                    ]
                ),
                "prob_BUY": [0.55, 0.75, 0.95],
                "prob_SHORT": [0.01, 0.01, 0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "equity_curve.csv"
            _save_equity_curve(
                pred_df,
                out_path,
                fee=0.0,
                n_pools=3,
                execution_priority="model_confidence",
                top_k_per_timestamp=2,
            )
            eq_df = pd.read_csv(out_path)

        self.assertEqual(len(eq_df), 2)
        self.assertEqual(list(eq_df["ticker"]), ["HIGH", "MID"])

    def test_equity_curve_realizes_pnl_at_exit_time(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:40:00",
                    ]
                ),
                "ticker": ["SLOW", "FAST"],
                "y_pred": [2, 2],
                "pnl_fraction": [0.01, 0.02],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 15:00:00",
                        "2025-01-02 14:45:00",
                    ]
                ),
                "prob_BUY": [0.60, 0.70],
                "prob_SHORT": [0.01, 0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "equity_curve.csv"
            _save_equity_curve(
                pred_df,
                out_path,
                fee=0.0,
                n_pools=2,
                execution_priority="first_seen",
            )
            eq_df = pd.read_csv(out_path)

        self.assertEqual(list(eq_df["ticker"]), ["FAST", "SLOW"])
        self.assertEqual(
            list(pd.to_datetime(eq_df["timestamp"]).dt.strftime("%H:%M:%S")),
            ["14:45:00", "15:00:00"],
        )
        self.assertEqual(
            list(pd.to_datetime(eq_df["entry_timestamp"]).dt.strftime("%H:%M:%S")),
            ["14:40:00", "14:30:00"],
        )
        self.assertEqual(
            list(pd.to_datetime(eq_df["exit_timestamp"]).dt.strftime("%H:%M:%S")),
            ["14:45:00", "15:00:00"],
        )
        self.assertAlmostEqual(float(eq_df.loc[0, "cumulative_portfolio_pnl_pct"]), 1.0)
        self.assertAlmostEqual(float(eq_df.loc[1, "cumulative_portfolio_pnl_pct"]), 1.5)

    def test_ticker_cooldown_minutes_blocks_quick_reentry(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:40:00",
                        "2025-01-02 15:35:00",
                    ]
                ),
                "ticker": ["AAA", "AAA", "AAA"],
                "y_pred": [2, 2, 2],
                "pnl_fraction": [0.01, 0.02, 0.03],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 14:35:00",
                        "2025-01-02 14:45:00",
                        "2025-01-02 15:40:00",
                    ]
                ),
                "prob_BUY": [0.60, 0.70, 0.80],
                "prob_SHORT": [0.01, 0.01, 0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "equity_curve.csv"
            _save_equity_curve(
                pred_df,
                out_path,
                fee=0.0,
                n_pools=3,
                execution_priority="model_confidence",
                ticker_cooldown_minutes=60,
            )
            eq_df = pd.read_csv(out_path)

        self.assertEqual(len(eq_df), 2)
        self.assertEqual(list(eq_df["ticker"]), ["AAA", "AAA"])
        self.assertEqual(
            list(pd.to_datetime(eq_df["timestamp"]).dt.strftime("%H:%M:%S")),
            ["14:35:00", "15:40:00"],
        )
        self.assertEqual(
            list(pd.to_datetime(eq_df["entry_timestamp"]).dt.strftime("%H:%M:%S")),
            ["14:30:00", "15:35:00"],
        )

    def test_meta_score_thresholds_demote_short_and_buy_independently(self) -> None:
        y_pred = pd.Series([0, 0, 2, 2], dtype="int64").to_numpy()
        meta_score = pd.Series([0.03, 0.01, 0.015, 0.005], dtype="float64").to_numpy()

        out = _apply_meta_score_thresholds(
            y_pred,
            meta_score,
            min_score_short=0.02,
            min_score_buy=0.01,
        )

        self.assertEqual(list(out), [0, 1, 2, 1])

    def test_short_execution_policy_demotes_short_to_hold(self) -> None:
        y_pred = pd.Series([0, 1, 2, 0], dtype="int64").to_numpy()

        blocked = _apply_short_execution_policy(y_pred, allow_short=False)
        allowed = _apply_short_execution_policy(y_pred, allow_short=True)

        self.assertEqual(list(blocked), [1, 1, 2, 1])
        self.assertEqual(list(allowed), [0, 1, 2, 0])


if __name__ == "__main__":
    unittest.main()
