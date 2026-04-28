from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from kvant.utils.pipeline_config import list_from_config
from kvant.utils.time_utils import ensure_utc_sorted_index

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STORE = _PROJECT_ROOT / "data" / "1m"
_DEFAULT_INTERVAL = "1m"


def should_use_partition_layout(cfg: dict) -> bool:
    """Determine whether month-partitioned data should be used."""
    hf_config = cfg.get("hf_config", {})
    dataset_id = hf_config.get("dataset_id", "")
    return bool(dataset_id)


def load_pipeline_ticker_dfs(
    cfg: dict,
    *,
    force_partition: bool = False,
    force_flat: bool = False,
) -> Dict[str, pd.DataFrame]:
    store_dir = Path(cfg["paths"].get("store", str(_DEFAULT_STORE)))
    if not store_dir.exists():
        raise SystemExit(f"Store directory not found: {store_dir}")

    symbols = list_from_config(cfg["data"].get("symbols")) or []
    use_partition = force_partition or (not force_flat and should_use_partition_layout(cfg))

    if use_partition:
        ticker_dfs = _load_from_partitioned(cfg, store_dir, symbols)
    else:
        ticker_dfs = _load_from_flat(store_dir, symbols)

    if not ticker_dfs:
        raise SystemExit("No data loaded from store")
    return {sym: ensure_utc_sorted_index(df) for sym, df in ticker_dfs.items() if len(df) > 0}


def apply_prepare_filters(ticker_dfs: Dict[str, pd.DataFrame], cfg: dict) -> Dict[str, pd.DataFrame]:
    """Apply pre-split filters configured in [prepare]."""
    out = {sym: ensure_utc_sorted_index(df) for sym, df in ticker_dfs.items()}
    skip_opening_minutes = int(cfg["prepare"].get("skip_opening_minutes", 0))
    if skip_opening_minutes <= 0:
        return out

    cutoff_minutes_since_midnight = 9 * 60 + 30 + skip_opening_minutes
    filtered: Dict[str, pd.DataFrame] = {}
    for sym, df in out.items():
        idx_et = df.index.tz_localize("UTC").tz_convert("America/New_York")
        minutes_since_midnight = idx_et.hour * 60 + idx_et.minute
        filtered[sym] = df[minutes_since_midnight >= cutoff_minutes_since_midnight]
    return filtered


def split_ticker_dfs_by_fraction(
    ticker_dfs: Dict[str, pd.DataFrame],
    *,
    val_frac: float,
    test_frac: float,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split each ticker chronologically into train/val/test fractions."""
    train_dfs: Dict[str, pd.DataFrame] = {}
    val_dfs: Dict[str, pd.DataFrame] = {}
    test_dfs: Dict[str, pd.DataFrame] = {}

    for sym, df in ticker_dfs.items():
        n = len(df)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val - n_test
        if n_train <= 0:
            continue
        train_dfs[sym] = df.iloc[:n_train].copy()
        val_dfs[sym] = df.iloc[n_train : n_train + n_val].copy()
        test_dfs[sym] = df.iloc[n_train + n_val :].copy()

    return train_dfs, val_dfs, test_dfs


def _load_from_flat(store_dir: Path, symbols: list[str]) -> Dict[str, pd.DataFrame]:
    from kvant.kdata.store import OHLCVStore

    available = sorted(p.stem for p in store_dir.glob("*.csv"))
    if not available:
        raise SystemExit(f"No CSV files found in {store_dir}")

    selected = symbols if symbols else available
    missing = [s for s in selected if s not in available]
    if missing:
        raise SystemExit(f"Tickers not found in {store_dir}: {missing}")

    store = OHLCVStore(store_dir)
    return store.load_all(selected)


def _load_from_partitioned(cfg: dict, store_dir: Path, symbols: list[str]) -> Dict[str, pd.DataFrame]:
    from kvant.kdata.hf.month_store import MonthPartitionedStore

    prepare_cfg = cfg.get("prepare", {})
    train_start = prepare_cfg.get("train_start_month", "2025-01")
    train_end = prepare_cfg.get("train_end_month", "2025-11")
    test_start = prepare_cfg.get("test_start_month", "2025-12")
    test_end = prepare_cfg.get("test_end_month", "2026-03")

    store = MonthPartitionedStore(store_dir)
    train_data = store.load_range(symbols, train_start, train_end)
    test_data = store.load_range(symbols, test_start, test_end)

    combined: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        parts = []
        train_df = train_data.get(sym)
        test_df = test_data.get(sym)
        if train_df is not None and not train_df.empty:
            parts.append(train_df)
        if test_df is not None and not test_df.empty:
            parts.append(test_df)
        if parts:
            combined[sym] = pd.concat(parts).sort_index()
    return combined

