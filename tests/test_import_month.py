"""
tests/test_import_month.py — Unit tests for import_month CLI and function.

Tests month import functionality, idempotency, and error handling.
"""
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kvant.kdata.hf.huggingface_retriever import HuggingFaceRetriever
from kvant.kdata.hf.import_month import import_month
from kvant.kdata.hf.month_store import MonthPartitionedStore


class TestImportMonth:
    """Test import_month function."""

    @pytest.fixture
    def temp_store(self):
        """Create temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MonthPartitionedStore(tmpdir)

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config."""
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[paths]
store = "data/1m"

[data]
symbols = ["AAPL", "MSFT", "TSLA"]

[hf_config]
dataset_id = "test/stocks-1m"
cache_dir = "~/.cache/huggingface"
""")
        return str(config_file)

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

    def test_import_month_invalid_format(self, temp_store):
        """Import with invalid month format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid month format"):
            import_month("2025-3")  # Single digit month

    def test_import_month_basic(self, temp_store, sample_df, mock_config):
        """Import month successfully."""
        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.get_month_shard.return_value = sample_df

        result = import_month(
            "2025-03",
            symbols=["AAPL"],
            retriever=mock_retriever,
            store=temp_store,
            config_path=mock_config,
        )

        assert result["status"] == "complete"
        assert result["month"] == "2025-03"
        assert "AAPL" in result["results"]
        assert result["results"]["AAPL"]["status"] == "ok"
        assert result["results"]["AAPL"]["appended"] == 100

    def test_import_month_skips_existing(self, temp_store, sample_df, mock_config):
        """Import skips already-existing months."""
        # Pre-populate month
        temp_store.append_month("AAPL", "2025-03", sample_df)

        mock_retriever = MagicMock()

        result = import_month(
            "2025-03",
            symbols=["AAPL"],
            retriever=mock_retriever,
            store=temp_store,
            config_path=mock_config,
        )

        # Should be skipped
        assert result["status"] == "skipped"
        assert result["reason"] == "already_exists"
        # Retriever should not be called
        mock_retriever.get_month_shard.assert_not_called()

    def test_import_month_no_data(self, temp_store, mock_config):
        """Import handles missing data gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.get_month_shard.return_value = pd.DataFrame()  # Empty

        result = import_month(
            "2025-03",
            symbols=["AAPL"],
            retriever=mock_retriever,
            store=temp_store,
            config_path=mock_config,
        )

        assert result["status"] == "complete"
        assert result["results"]["AAPL"]["status"] == "no_data"

    def test_import_month_error_handling(self, temp_store, mock_config):
        """Import handles retriever errors gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.get_month_shard.side_effect = Exception("Network error")

        result = import_month(
            "2025-03",
            symbols=["AAPL"],
            retriever=mock_retriever,
            store=temp_store,
            config_path=mock_config,
        )

        assert result["status"] == "complete"
        assert result["results"]["AAPL"]["status"] == "error"
        assert "Network error" in result["results"]["AAPL"]["error"]

    def test_import_month_multiple_symbols(self, temp_store, sample_df, mock_config):
        """Import multiple symbols in one call."""
        mock_retriever = MagicMock()
        mock_retriever.get_month_shard.return_value = sample_df

        result = import_month(
            "2025-03",
            symbols=["AAPL", "MSFT"],
            retriever=mock_retriever,
            store=temp_store,
            config_path=mock_config,
        )

        assert result["status"] == "complete"
        assert "AAPL" in result["results"]
        assert "MSFT" in result["results"]
        assert result["results"]["AAPL"]["status"] == "ok"
        assert result["results"]["MSFT"]["status"] == "ok"
        assert mock_retriever.get_month_shard.call_count == 2

    def test_import_month_timestamp_format(self, temp_store, sample_df, mock_config):
        """Result includes ISO timestamp."""
        mock_retriever = MagicMock()
        mock_retriever.get_month_shard.return_value = sample_df

        result = import_month(
            "2025-03",
            symbols=["AAPL"],
            retriever=mock_retriever,
            store=temp_store,
            config_path=mock_config,
        )

        assert "timestamp" in result
        # Should be ISO format
        ts = pd.to_datetime(result["timestamp"], utc=True)
        assert ts is not None


class TestImportMonthIntegration:
    """Integration tests for import_month."""

    def test_import_month_end_to_end(self, tmp_path):
        """End-to-end import with real files."""
        # Create mock config
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[paths]
store = "data/1m"

[data]
symbols = ["AAPL"]

[hf_config]
dataset_id = "test/stocks"
""")

        # Create temp store
        store_dir = tmp_path / "data" / "1m"
        store_dir.mkdir(parents=True)
        store = MonthPartitionedStore(store_dir)

        # Create sample data
        dates = pd.date_range("2025-03-01", periods=50, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [101.0] * 50,
                "low": [99.0] * 50,
                "close": [100.5] * 50,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.get_month_shard.return_value = df

        # Import
        result = import_month(
            "2025-03",
            symbols=["AAPL"],
            retriever=mock_retriever,
            store=store,
            config_path=str(config_file),
        )

        assert result["status"] == "complete"

        # Verify file was created
        month_file = store_dir / "2025-03" / "AAPL.csv"
        assert month_file.exists()

        # Verify data
        loaded = store.load_month("AAPL", "2025-03")
        assert len(loaded) == 50


class TestHuggingFaceRetrieverBatchLoading:
    """Tests month-parquet loading and batch import behavior."""

    @pytest.fixture
    def month_source_df(self):
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-03-01T14:30:00Z",
                        "2025-03-01T14:31:00Z",
                        "2025-03-01T14:32:00Z",
                    ],
                    utc=True,
                ),
                "open": [100.0, 101.0, 200.0],
                "high": [101.0, 102.0, 201.0],
                "low": [99.0, 100.0, 199.0],
                "close": [100.5, 101.5, 200.5],
                "volume": [1000, 1100, 2000],
                "ticker": ["AAPL", "AAPL", "MSFT"],
            }
        )

    @pytest.fixture
    def temp_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MonthPartitionedStore(tmpdir)

    @pytest.fixture
    def mock_config(self, tmp_path):
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[paths]
store = "data/1m"

[data]
symbols = ["AAPL", "MSFT"]

[hf_config]
dataset_id = "test/stocks-1m"
cache_dir = "~/.cache/huggingface"
""")
        return str(config_file)

    @pytest.fixture
    def sample_df(self):
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

    def test_get_month_shards_reads_month_file_once(self, month_source_df):
        retriever = HuggingFaceRetriever(
            dataset_id="test/stocks-1m",
            cache_dir="~/.cache/huggingface",
        )

        with (
            patch("huggingface_hub.hf_hub_download", return_value="C:/tmp/ohlcv_2025-03.parquet") as mock_download,
            patch("pandas.read_parquet", return_value=month_source_df.copy()) as mock_read_parquet,
        ):
            result = retriever.get_month_shards(["AAPL", "MSFT", "NVDA"], "2025-03")

        mock_download.assert_called_once()
        mock_read_parquet.assert_called_once()
        assert len(result["AAPL"]) == 2
        assert len(result["MSFT"]) == 1
        assert result["NVDA"].empty
        assert list(result["AAPL"].columns) == ["open", "high", "low", "close", "volume"]

    def test_import_month_uses_batch_loading_for_hf_retriever(self, temp_store, mock_config, sample_df):
        retriever = HuggingFaceRetriever(dataset_id="test/stocks-1m")
        retriever.get_month_shards = MagicMock(
            return_value={"AAPL": sample_df, "MSFT": pd.DataFrame(columns=sample_df.columns)}
        )
        retriever.get_month_shard = MagicMock(side_effect=AssertionError("per-symbol fetch should not be used"))

        result = import_month(
            "2025-03",
            symbols=["AAPL", "MSFT"],
            retriever=retriever,
            store=temp_store,
            config_path=mock_config,
        )

        retriever.get_month_shards.assert_called_once_with(["AAPL", "MSFT"], "2025-03")
        retriever.get_month_shard.assert_not_called()
        assert result["results"]["AAPL"]["status"] == "ok"
        assert result["results"]["MSFT"]["status"] == "no_data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
