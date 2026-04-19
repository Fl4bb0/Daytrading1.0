"""
tests/test_month_store.py — Unit tests for MonthPartitionedStore.

Tests append-only semantics, duplicate skipping, month loading, and range queries.
"""
import tempfile

import pandas as pd
import pytest

from kvant.kdata.hf.month_store import MonthPartitionedStore, _is_valid_month, _month_range


class TestMonthValidation:
    """Test month format validation."""

    def test_is_valid_month_valid(self):
        """Valid month formats."""
        assert _is_valid_month("2025-03")
        assert _is_valid_month("2020-01")
        assert _is_valid_month("2099-12")

    def test_is_valid_month_invalid(self):
        """Invalid month formats."""
        assert not _is_valid_month("2025-3")      # Single digit
        assert not _is_valid_month("2025/03")     # Wrong separator
        assert not _is_valid_month("202503")      # No separator
        assert not _is_valid_month("2025-13")     # Invalid month
        assert not _is_valid_month("2025-00")     # Invalid month
        assert not _is_valid_month("")
        assert not _is_valid_month(None)


class TestMonthRange:
    """Test month range generation."""

    def test_month_range_single(self):
        """Range of one month."""
        result = _month_range("2025-03", "2025-03")
        assert result == ["2025-03"]

    def test_month_range_multi(self):
        """Range of multiple months."""
        result = _month_range("2025-01", "2025-03")
        assert result == ["2025-01", "2025-02", "2025-03"]

    def test_month_range_cross_year(self):
        """Range crossing year boundary."""
        result = _month_range("2024-11", "2025-02")
        assert result == ["2024-11", "2024-12", "2025-01", "2025-02"]

    def test_month_range_invalid(self):
        """Invalid month formats raise ValueError."""
        with pytest.raises(ValueError):
            _month_range("2025-3", "2025-05")  # Single digit
        with pytest.raises(ValueError):
            _month_range("2025-05", "2025-3")  # Single digit


class TestMonthPartitionedStore:
    """Test MonthPartitionedStore class."""

    @pytest.fixture
    def store(self):
        """Create temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MonthPartitionedStore(tmpdir)

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

    def test_store_creation(self, store):
        """Store creates root directory."""
        assert store.root_dir.exists()
        assert store.root_dir.is_dir()

    def test_get_month_dir(self, store):
        """Month directory path construction."""
        path = store.get_month_dir("2025-03")
        assert "2025-03" in str(path)

    def test_get_month_path(self, store):
        """Month file path construction."""
        path = store.get_month_path("AAPL", "2025-03")
        assert "AAPL.csv" in str(path)
        assert "2025-03" in str(path)

    def test_ensure_month_dir(self, store):
        """Month directory created."""
        path = store.ensure_month_dir("2025-03")
        assert path.exists()
        assert path.is_dir()

    def test_load_month_empty(self, store):
        """Load non-existent month returns empty DataFrame."""
        df = store.load_month("AAPL", "2025-03")
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_append_and_load(self, store, sample_df):
        """Append bars, then load them back."""
        report = store.append_month("AAPL", "2025-03", sample_df)

        assert report["appended"] == 100
        assert report["skipped"] == 0
        assert report["total"] == 100

        loaded = store.load_month("AAPL", "2025-03")
        assert len(loaded) == 100
        assert loaded.index.is_monotonic_increasing
        assert (loaded["open"] == sample_df["open"]).all()

    def test_append_only_duplicate_skip(self, store, sample_df):
        """Append-only: skip existing timestamps."""
        # First append
        report1 = store.append_month("AAPL", "2025-03", sample_df)
        assert report1["appended"] == 100

        # Second append with same data
        report2 = store.append_month("AAPL", "2025-03", sample_df, skip_existing=True)
        assert report2["appended"] == 0
        assert report2["skipped"] == 100
        assert report2["total"] == 100  # No increase

    def test_append_partial_overlap(self, store, sample_df):
        """Append with partial overlap."""
        # First append: 100 bars
        report1 = store.append_month("AAPL", "2025-03", sample_df)
        assert report1["appended"] == 100

        # Create new data: last 50 bars of existing + 50 new bars
        overlap_dates = sample_df.index[-50:]
        new_dates = pd.date_range(sample_df.index[-1] + pd.Timedelta("1min"), periods=50, freq="1min", tz="UTC")
        combined_dates = overlap_dates.union(new_dates)

        new_df = pd.DataFrame(
            {
                "open": [200.0 + i * 0.01 for i in range(100)],
                "high": [201.0 + i * 0.01 for i in range(100)],
                "low": [199.0 + i * 0.01 for i in range(100)],
                "close": [200.5 + i * 0.01 for i in range(100)],
                "volume": [2000 * (i + 1) for i in range(100)],
            },
            index=combined_dates,
        )

        # Second append with skip_existing=True
        report2 = store.append_month("AAPL", "2025-03", new_df, skip_existing=True)
        assert report2["appended"] == 50  # Only new bars appended
        assert report2["skipped"] == 50   # Overlapping bars skipped
        assert report2["total"] == 150    # Total now 100 + 50

    def test_append_no_skip(self, store, sample_df):
        """Append with skip_existing=False (keep first)."""
        # First append
        store.append_month("AAPL", "2025-03", sample_df, skip_existing=False)

        # Second append: modified data
        modified_df = sample_df.copy()
        modified_df["close"] = modified_df["close"] * 2  # Double all close prices

        # Append without skip (will deduplicate afterward, keeping first)
        report = store.append_month("AAPL", "2025-03", modified_df, skip_existing=False)

        # Should have dedup'd to keep first occurrence
        loaded = store.load_month("AAPL", "2025-03")
        assert len(loaded) == 100
        # Close prices should be original (first write wins)
        assert (loaded["close"] == sample_df["close"]).all()

    def test_append_preserves_order(self, store, sample_df):
        """Result is always sorted by timestamp."""
        # Append first 50 bars
        first_half = sample_df.iloc[:50]
        store.append_month("AAPL", "2025-03", first_half)

        # Append last 50 bars (non-overlapping, but out of order relative to file)
        second_half = sample_df.iloc[50:]
        store.append_month("AAPL", "2025-03", second_half)

        loaded = store.load_month("AAPL", "2025-03")
        assert len(loaded) == 100
        assert loaded.index.is_monotonic_increasing

    def test_list_months(self, store, sample_df):
        """List all months present."""
        store.append_month("AAPL", "2025-01", sample_df)
        store.append_month("AAPL", "2025-02", sample_df)
        store.append_month("MSFT", "2025-03", sample_df)

        months = store.list_months()
        assert sorted(months) == ["2025-01", "2025-02", "2025-03"]

    def test_list_symbols_in_month(self, store, sample_df):
        """List symbols present in a month."""
        store.append_month("AAPL", "2025-03", sample_df)
        store.append_month("MSFT", "2025-03", sample_df)
        store.append_month("TSLA", "2025-04", sample_df)

        symbols = store.list_symbols_in_month("2025-03")
        assert sorted(symbols) == ["AAPL", "MSFT"]

        symbols = store.list_symbols_in_month("2025-04")
        assert symbols == ["TSLA"]

    def test_load_range(self, store, sample_df):
        """Load multiple months for symbols."""
        symbols = ["AAPL", "MSFT"]
        for sym in symbols:
            for month in ["2025-01", "2025-02", "2025-03"]:
                store.append_month(sym, month, sample_df)

        # Load range
        data = store.load_range(symbols, "2025-01", "2025-03")

        assert len(data) == 2
        assert all(sym in data for sym in symbols)
        assert len(data["AAPL"]) == 300  # 3 months * 100 bars
        assert len(data["MSFT"]) == 300
        assert data["AAPL"].index.is_monotonic_increasing
        assert data["MSFT"].index.is_monotonic_increasing

    def test_load_range_partial(self, store, sample_df):
        """Load range with some months missing."""
        store.append_month("AAPL", "2025-01", sample_df)
        # 2025-02 missing
        store.append_month("AAPL", "2025-03", sample_df)

        data = store.load_range(["AAPL"], "2025-01", "2025-03")

        assert len(data["AAPL"]) == 200  # Only 2025-01 and 2025-03 loaded
        assert data["AAPL"].index.is_monotonic_increasing

    def test_load_range_symbol_not_found(self, store, sample_df):
        """Load range with symbols that don't exist."""
        store.append_month("AAPL", "2025-01", sample_df)

        data = store.load_range(["AAPL", "MSFT"], "2025-01", "2025-01")

        assert len(data) == 2
        assert len(data["AAPL"]) == 100
        assert data["MSFT"].empty

    def test_multiple_symbols(self, store, sample_df):
        """Store multiple symbols independently."""
        store.append_month("AAPL", "2025-03", sample_df)
        store.append_month("MSFT", "2025-03", sample_df)

        aapl_df = store.load_month("AAPL", "2025-03")
        msft_df = store.load_month("MSFT", "2025-03")

        assert len(aapl_df) == 100
        assert len(msft_df) == 100
        # Different symbols should be independent
        assert aapl_df["open"].iloc[0] != msft_df["open"].iloc[0]

    def test_csv_format(self, store, sample_df):
        """CSV is written in correct format."""
        store.append_month("AAPL", "2025-03", sample_df)

        path = store.get_month_path("AAPL", "2025-03")
        assert path.exists()

        # Read CSV manually
        with open(path) as f:
            lines = f.readlines()

        # First line should be header
        assert "timestamp" in lines[0].lower()
        # Should have data rows
        assert len(lines) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
