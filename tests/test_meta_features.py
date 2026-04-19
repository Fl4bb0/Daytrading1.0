from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kvant.meta import RidgeMetaModel, add_meta_features


class MetaFeatureTests(unittest.TestCase):
    def test_causal_prior_waits_for_bar_close_time(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:40:00",
                        "2025-01-02 14:46:00",
                    ]
                ),
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "y_pred": [2, 2, 2],
                "pnl_fraction": [0.02, -0.01, 0.03],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 14:45:00",
                        "2025-01-02 14:50:00",
                        "2025-01-02 15:00:00",
                    ]
                ),
                "prob_BUY": [0.70, 0.60, 0.80],
                "prob_SHORT": [0.01, 0.01, 0.01],
            }
        )

        meta_df = add_meta_features(pred_df, fee=0.0, shrinkage_k=0.0)

        self.assertEqual(float(meta_df.loc[0, "ticker_side_prior_score"]), 0.0)
        self.assertEqual(float(meta_df.loc[0, "ticker_side_prior_n"]), 0.0)
        self.assertEqual(float(meta_df.loc[1, "ticker_side_prior_score"]), 0.0)
        self.assertEqual(float(meta_df.loc[1, "ticker_side_prior_n"]), 0.0)
        self.assertAlmostEqual(float(meta_df.loc[2, "ticker_side_prior_score"]), 0.02)
        self.assertEqual(float(meta_df.loc[2, "ticker_side_prior_n"]), 1.0)

    def test_shrinkage_blends_local_and_global_side_history(self) -> None:
        history_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-02 14:30:00",
                        "2025-01-02 14:31:00",
                    ]
                ),
                "ticker": ["MSFT", "AAPL"],
                "y_pred": [2, 2],
                "pnl_fraction": [0.02, 0.06],
                "bar_close_time": pd.to_datetime(
                    [
                        "2025-01-02 14:35:00",
                        "2025-01-02 14:36:00",
                    ]
                ),
                "prob_BUY": [0.60, 0.80],
                "prob_SHORT": [0.01, 0.01],
            }
        )
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-01-02 14:40:00"]),
                "ticker": ["AAPL"],
                "y_pred": [2],
                "pnl_fraction": [0.03],
                "bar_close_time": pd.to_datetime(["2025-01-02 14:50:00"]),
                "prob_BUY": [0.75],
                "prob_SHORT": [0.01],
            }
        )

        meta_df = add_meta_features(
            pred_df,
            history_df=history_df,
            fee=0.0,
            shrinkage_k=2.0,
        )

        self.assertEqual(float(meta_df.loc[0, "ticker_side_prior_n"]), 1.0)
        self.assertAlmostEqual(float(meta_df.loc[0, "ticker_side_prior_score"]), 0.0466666667)


class RidgeMetaModelTests(unittest.TestCase):
    def test_save_and_load_preserve_predictions(self) -> None:
        frame = pd.DataFrame(
            {
                "side_confidence": [0.55, 0.70, 0.85, 0.65],
                "side": [1.0, -1.0, 1.0, -1.0],
                "ticker_side_prior_score": [0.01, -0.02, 0.03, -0.01],
                "ticker_side_prior_n": [0.0, 2.0, 5.0, 3.0],
                "target_signed_net_pnl": [0.005, -0.004, 0.012, -0.006],
            }
        )

        model = RidgeMetaModel(alpha=1.0)
        model.fit(frame)
        pred_before = model.predict(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "meta"
            model.save(out_dir)
            loaded = RidgeMetaModel.load(out_dir)

        pred_after = loaded.predict(frame)
        np.testing.assert_allclose(pred_before, pred_after)


if __name__ == "__main__":
    unittest.main()
