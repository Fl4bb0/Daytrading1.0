from __future__ import annotations

import unittest

import pandas as pd

from kvant.utils.walk_forward import build_walk_forward_folds, split_ticker_dfs_for_fold


def _make_df(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(len(idx)),
            "high": range(len(idx)),
            "low": range(len(idx)),
            "close": range(len(idx)),
            "volume": [1] * len(idx),
        },
        index=idx,
    )


class WalkForwardTests(unittest.TestCase):
    def test_build_expanding_folds(self) -> None:
        ticker_dfs = {
            "AAPL": _make_df("2025-01-01", "2025-07-31"),
            "MSFT": _make_df("2025-01-01", "2025-07-31"),
        }
        folds = build_walk_forward_folds(
            ticker_dfs,
            {
                "enabled": True,
                "mode": "expanding",
                "start_month": "2025-01",
                "end_month": "2025-07",
                "train_span_months": 3,
                "val_span_months": 1,
                "test_span_months": 1,
                "step_span_months": 1,
            },
        )

        self.assertEqual([fold.fold_id for fold in folds], ["fold_000", "fold_001", "fold_002"])
        self.assertEqual(str(folds[0].train_start.date()), "2025-01-01")
        self.assertEqual(str(folds[0].val_start.date()), "2025-04-01")
        self.assertEqual(str(folds[0].test_start.date()), "2025-05-01")
        self.assertEqual(str(folds[1].train_end_exclusive.date()), "2025-05-01")

    def test_split_ticker_dfs_for_fold_filters_sparse_tickers(self) -> None:
        ticker_dfs = {
            "AAPL": _make_df("2025-01-01", "2025-05-31"),
            "SPARSE": _make_df("2025-01-01", "2025-03-10"),
        }
        fold = build_walk_forward_folds(
            ticker_dfs,
            {
                "enabled": True,
                "mode": "rolling",
                "start_month": "2025-01",
                "end_month": "2025-05",
                "train_span_months": 2,
                "val_span_months": 1,
                "test_span_months": 1,
                "step_span_months": 1,
            },
        )[0]

        train_dfs, val_dfs, test_dfs, rows = split_ticker_dfs_for_fold(
            ticker_dfs,
            fold,
            min_train_rows_per_ticker=5,
            min_val_rows_per_ticker=5,
            min_test_rows_per_ticker=5,
        )

        self.assertEqual(sorted(train_dfs), ["AAPL"])
        self.assertEqual(sorted(val_dfs), ["AAPL"])
        self.assertEqual(sorted(test_dfs), ["AAPL"])
        sparse_row = next(row for row in rows if row["ticker"] == "SPARSE")
        self.assertFalse(sparse_row["eligible"])


if __name__ == "__main__":
    unittest.main()
