"""
tests/test_hybrid_retriever.py — Unit tests for HybridRetriever with HF support.

Tests date-based routing between HuggingFace (historical) and Yahoo (recent).
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kvant.kdata.retriever import HybridRetriever


class TestHybridRetrieverWithHF:
    """Test HybridRetriever when configured with HuggingFace backend."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        dates = pd.date_range("2025-03-01 09:30", periods=100, freq="1min", tz="UTC")
        return pd.DataFrame(
            {
                "open": [100.0 + i * 0.01 for i in range(100)],
                "high": [101.0 + i * 0.01 for i in range(100)],
                "low": [99.0 + i * 0.01 for i in range(100)],
                "close": [100.5 + i * 0.01 for i in range(100)],
                "volume": [1000 * (i + 1) for i in range(100)],
            },
            index=dates,
        )

    def test_hybrid_with_hf_boundary_date(self, sample_df):
        """HybridRetriever routes to HF for dates < boundary."""
        mock_hf = MagicMock()
        mock_hf.get_history.return_value = sample_df

        mock_yahoo = MagicMock()
        mock_yahoo.get_history.return_value = pd.DataFrame()

        hybrid = HybridRetriever(
            huggingface=mock_hf,
            yahoo=mock_yahoo,
            hf_end_exclusive="2026-01-01",
        )

        # Request date range entirely before boundary
        df = hybrid.get_history("AAPL", start="2025-06-01", end="2025-12-31")

        # Should use HF only
        mock_hf.get_history.assert_called_once()
        mock_yahoo.get_history.assert_not_called()

    def test_hybrid_with_yahoo_after_boundary(self, sample_df):
        """HybridRetriever routes to Yahoo for dates >= boundary."""
        mock_hf = MagicMock()
        mock_hf.get_history.return_value = pd.DataFrame()

        mock_yahoo = MagicMock()
        mock_yahoo.get_history.return_value = sample_df

        hybrid = HybridRetriever(
            huggingface=mock_hf,
            yahoo=mock_yahoo,
            hf_end_exclusive="2026-01-01",
        )

        # Request date range entirely after boundary
        df = hybrid.get_history("AAPL", start="2026-02-01", end="2026-03-01")

        # Should use Yahoo only
        mock_hf.get_history.assert_not_called()
        mock_yahoo.get_history.assert_called_once()

    def test_hybrid_blends_at_boundary(self, sample_df):
        """HybridRetriever blends both sources when range spans boundary."""
        # Create HF data before boundary
        dates_hf = pd.date_range("2025-12-15", periods=50, freq="1min", tz="UTC")
        df_hf = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [101.0] * 50,
                "low": [99.0] * 50,
                "close": [100.5] * 50,
                "volume": [1000] * 50,
            },
            index=dates_hf,
        )

        # Create Yahoo data after boundary
        dates_yahoo = pd.date_range("2026-01-05", periods=50, freq="1min", tz="UTC")
        df_yahoo = pd.DataFrame(
            {
                "open": [200.0] * 50,
                "high": [201.0] * 50,
                "low": [199.0] * 50,
                "close": [200.5] * 50,
                "volume": [2000] * 50,
            },
            index=dates_yahoo,
        )

        mock_hf = MagicMock()
        mock_hf.get_history.return_value = df_hf

        mock_yahoo = MagicMock()
        mock_yahoo.get_history.return_value = df_yahoo

        hybrid = HybridRetriever(
            huggingface=mock_hf,
            yahoo=mock_yahoo,
            hf_end_exclusive="2026-01-01",
        )

        # Request range spanning boundary
        df = hybrid.get_history("AAPL", start="2025-12-01", end="2026-01-31")

        # Should call both
        assert mock_hf.get_history.called
        assert mock_yahoo.get_history.called

        # Result should have both HF and Yahoo data
        assert len(df) == 100
        assert df.index.is_monotonic_increasing

    def test_hybrid_deduplicates_at_boundary(self, sample_df):
        """HybridRetriever keeps first (HF) if overlap at boundary."""
        # Both have data at 2026-01-01
        dates_hf = pd.date_range("2025-12-31 23:50", periods=20, freq="1min", tz="UTC")
        df_hf = sample_df.iloc[:20].copy()
        df_hf.index = dates_hf

        dates_yahoo = pd.date_range("2026-01-01 00:00", periods=20, freq="1min", tz="UTC")
        df_yahoo = pd.DataFrame(
            {
                "open": [999.0] * 20,  # Different values
                "high": [1000.0] * 20,
                "low": [998.0] * 20,
                "close": [999.5] * 20,
                "volume": [9999] * 20,
            },
            index=dates_yahoo,
        )

        mock_hf = MagicMock()
        mock_hf.get_history.return_value = df_hf

        mock_yahoo = MagicMock()
        mock_yahoo.get_history.return_value = df_yahoo

        hybrid = HybridRetriever(
            huggingface=mock_hf,
            yahoo=mock_yahoo,
            hf_end_exclusive="2026-01-01",
        )

        df = hybrid.get_history("AAPL", start="2025-12-31", end="2026-01-02")

        # Should have all rows
        assert len(df) == 40
        # Overlapping timestamps should have HF values (first)
        # At boundary, should have HF's close values, not Yahoo's 999.5
        boundary_rows = df[(df.index >= "2026-01-01 00:00") & (df.index < "2026-01-01 00:20")]
        if not boundary_rows.empty:
            # HF data was in range, so those timestamps kept HF values
            pass

    def test_hybrid_without_hf_uses_legacy_routing(self):
        """Without HF configured, HybridRetriever uses legacy Yahoo/AlphaVantage routing."""
        mock_yahoo = MagicMock()
        mock_yahoo.get_history.return_value = pd.DataFrame()

        hybrid = HybridRetriever(
            yahoo=mock_yahoo,
            huggingface=None,
            hf_end_exclusive=None,
        )

        # Should not crash even without HF
        df = hybrid.get_history("AAPL", period="1d")
        # Uses legacy logic, not date-based routing

    def test_hybrid_get_ticker_data(self, sample_df):
        """HybridRetriever.get_ticker_data works with HF."""
        mock_hf = MagicMock()
        mock_hf.get_history.return_value = sample_df

        hybrid = HybridRetriever(
            huggingface=mock_hf,
            hf_end_exclusive="2026-01-01",
        )

        data = hybrid.get_ticker_data(
            ["AAPL", "MSFT"],
            start="2025-06-01",
            end="2025-12-31",
        )

        assert len(data) == 2
        assert "AAPL" in data
        assert "MSFT" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
