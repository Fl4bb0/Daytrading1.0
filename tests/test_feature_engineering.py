import unittest

import numpy as np
import pandas as pd

from kvant.features.feature_engineering import IntradayTA10Features, StandardizedFeatures


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02 14:30:00", periods=len(close), freq="1min")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=idx,
    )


class StandardizedFeaturesTickerFitTests(unittest.TestCase):
    def test_fit_from_ticker_dfs_pools_per_ticker_features(self):
        aapl = _ohlcv(np.linspace(100.0, 110.0, 80))
        msft = _ohlcv(np.linspace(400.0, 430.0, 80))
        ticker_dfs = {"AAPL": aapl, "MSFT": msft}

        fe = StandardizedFeatures(base=IntradayTA10Features())
        fe.fit_from_ticker_dfs(ticker_dfs)

        X_parts = []
        names_ref = None
        for df in ticker_dfs.values():
            X, names = fe.base.transform(df)
            names_ref = names if names_ref is None else names_ref
            X_parts.append(X)

        X_expected = np.concatenate(X_parts, axis=0)
        expected_mean = np.nanmean(X_expected, axis=0).astype(np.float32)
        expected_std = np.nanstd(X_expected, axis=0)
        expected_std = np.where(expected_std < fe.eps, 1.0, expected_std).astype(np.float32)

        np.testing.assert_allclose(fe.mean_, expected_mean)
        np.testing.assert_allclose(fe.std_, expected_std)
        self.assertEqual(fe.feature_names_, names_ref)

    def test_fit_from_ticker_dfs_avoids_mixed_macd_statistics(self):
        aapl = _ohlcv(np.linspace(100.0, 110.0, 80))
        msft = _ohlcv(np.linspace(400.0, 430.0, 80))

        ticker_safe = StandardizedFeatures(base=IntradayTA10Features())
        ticker_safe.fit_from_ticker_dfs({"AAPL": aapl, "MSFT": msft})

        mixed = StandardizedFeatures(base=IntradayTA10Features())
        mixed.fit(pd.concat([aapl, msft], axis=0).sort_index())

        macd_idx = ticker_safe.feature_names_.index("macd")
        self.assertGreater(abs(float(mixed.std_[macd_idx] - ticker_safe.std_[macd_idx])), 1.0)


if __name__ == "__main__":
    unittest.main()
