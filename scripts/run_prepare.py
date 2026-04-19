"""
scripts/run_prepare.py — Entry-point for data preparation.

Supports both legacy flat CSV layout (data/1m/{ticker}.csv) and new
month-partitioned layout (data/1m/{YYYY-MM}/{ticker}.csv).

With month-partitioned layout, uses configurable train/test month ranges from config.

Usage
-----
  python scripts/run_prepare.py
  python scripts/run_prepare.py --config pipeline.toml
  python scripts/run_prepare.py --use-partition     # Force new layout
  python scripts/run_prepare.py --use-flat          # Force legacy layout
"""
import argparse
import logging
from pathlib import Path

from kvant.utils.pipeline_config import list_from_config, load_pipeline_config

logger = logging.getLogger(__name__)

# Default store written by run_weekly_update.py
_DEFAULT_STORE = Path(__file__).resolve().parents[1] / "data" / "1m"
_DEFAULT_INTERVAL = "1m"


def _should_use_partition_layout(cfg: dict) -> bool:
    """Determine if should use month-partitioned layout."""
    hf_config = cfg.get("hf_config", {})
    dataset_id = hf_config.get("dataset_id", "")
    return bool(dataset_id)


def main():
    parser = argparse.ArgumentParser(description="Prepare an ML experiment dataset.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    parser.add_argument(
        "--use-partition",
        action="store_true",
        help="Force month-partitioned layout",
    )
    parser.add_argument(
        "--use-flat",
        action="store_true",
        help="Force legacy flat CSV layout",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg, cfg_path = load_pipeline_config(args.config)

    store_dir = Path(cfg["paths"].get("store", str(_DEFAULT_STORE)))
    if not store_dir.exists():
        raise SystemExit(f"Store directory not found: {store_dir}")

    symbols = list_from_config(cfg["data"].get("symbols"))
    interval = cfg["data"].get("interval", _DEFAULT_INTERVAL)
    val_frac = float(cfg["prepare"].get("val_frac", 0.15))
    test_frac = float(cfg["prepare"].get("test_frac", 0.15))

    # Decide which layout to use
    use_partition = args.use_partition or (
        not args.use_flat and _should_use_partition_layout(cfg)
    )

    print(f"Store    : {store_dir.resolve()}")
    print(f"Config   : {cfg_path}")
    print(f"Tickers  : {len(symbols)}")
    print(f"Layout   : {'month-partitioned' if use_partition else 'flat CSV'}\n")

    # Load DataFrames from the local store
    if use_partition:
        ticker_dfs = _load_from_partitioned(cfg, store_dir, symbols)
    else:
        ticker_dfs = _load_from_flat(store_dir, symbols)

    if not ticker_dfs:
        raise SystemExit("No data loaded from store")

    # Optionally drop bars from the first N minutes after NYSE open (9:30 ET).
    # Applied before the train/val/test split so all splits are filtered uniformly.
    skip_opening_minutes = int(cfg["prepare"].get("skip_opening_minutes", 0))
    if skip_opening_minutes > 0:
        cutoff_minutes_since_midnight = 9 * 60 + 30 + skip_opening_minutes
        for sym in list(ticker_dfs.keys()):
            df = ticker_dfs[sym]
            idx_et = df.index.tz_convert("America/New_York")
            minutes_since_midnight = idx_et.hour * 60 + idx_et.minute
            ticker_dfs[sym] = df[minutes_since_midnight >= cutoff_minutes_since_midnight]
        print(f"Skipping first {skip_opening_minutes} min after open (bars before 09:{30 + skip_opening_minutes:02d} ET dropped)")

    from kvant.experiment.prepare import build_default_components, prepare_experiment, PREPARED_DATA_ROOT

    # Split each ticker chronologically
    train_dfs, val_dfs, test_dfs = {}, {}, {}
    for sym, df in ticker_dfs.items():
        n       = len(df)
        n_test  = max(1, int(n * test_frac))
        n_val   = max(1, int(n * val_frac))
        n_train = n - n_val - n_test
        if n_train <= 0:
            print(f"  Skipping {sym}: only {n} bars — not enough for the split fractions.")
            continue
        train_dfs[sym] = df.iloc[:n_train].copy()
        val_dfs[sym]   = df.iloc[n_train: n_train + n_val].copy()
        test_dfs[sym]  = df.iloc[n_train + n_val:].copy()
        print(f"  {sym}: {n_train} train / {n_val} val / {n_test} test bars")

    if not train_dfs:
        raise SystemExit("No tickers had enough data after splitting.")

    sampler, fe, labeler, cfg = build_default_components(
        interval=interval,
        volatility_scaled_barrier=bool(cfg["prepare"].get("volatility_scaled_barrier", True)),
        vol_scale_min=float(cfg["prepare"].get("vol_scale_min", 0.5)),
        vol_scale_max=float(cfg["prepare"].get("vol_scale_max", 2.0)),
        lookback_L=int(cfg["prepare"].get("lookback", 20)),
        width_minutes=int(cfg["prepare"].get("width_minutes", 20)),
        height_pct=float(cfg["prepare"].get("height_pct", 0.5)),
        target_bars_per_day=int(cfg["prepare"].get("target_bars_per_day", 195)),
    )

    PREPARED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = prepare_experiment(
        out_root=PREPARED_DATA_ROOT,
        cfg=cfg,
        sampler=sampler,
        fe=fe,
        labeler=labeler,
        ticker_dfs_train=train_dfs,
        ticker_dfs_val=val_dfs,
        ticker_dfs_test=test_dfs,
        experiment_id=cfg.experiment_name,
    )
    (PREPARED_DATA_ROOT / "last_experiment.txt").write_text(manifest.exp_dir.name)


def _load_from_flat(store_dir: Path, symbols: list[str]) -> dict:
    """Load from legacy flat CSV layout."""
    from kvant.kdata.store import OHLCVStore

    # Discover available tickers from CSV filenames
    available = sorted(p.stem for p in store_dir.glob("*.csv"))
    if not available:
        raise SystemExit(f"No CSV files found in {store_dir}")

    symbols = symbols if symbols else available
    missing = [s for s in symbols if s not in available]
    if missing:
        raise SystemExit(f"Tickers not found in {store_dir}: {missing}")

    print("Loading from flat CSV files...")
    store = OHLCVStore(store_dir)
    return store.load_all(symbols)


def _load_from_partitioned(cfg: dict, store_dir: Path, symbols: list[str]) -> dict:
    """Load from new month-partitioned layout using configured month ranges."""
    from kvant.kdata.hf.month_store import MonthPartitionedStore

    prepare_cfg = cfg.get("prepare", {})
    train_start = prepare_cfg.get("train_start_month", "2025-01")
    train_end = prepare_cfg.get("train_end_month", "2025-11")
    test_start = prepare_cfg.get("test_start_month", "2025-12")
    test_end = prepare_cfg.get("test_end_month", "2026-03")

    print(f"Loading from month partitions...")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Test:  {test_start} to {test_end}")

    store = MonthPartitionedStore(store_dir)

    # Load training months
    train_data = store.load_range(symbols, train_start, train_end)

    # Load test months
    test_data = store.load_range(symbols, test_start, test_end)

    # Combine train + test (will be split later)
    combined = {}
    for sym in symbols:
        train_df = train_data.get(sym, None)
        test_df = test_data.get(sym, None)

        dfs = []
        if train_df is not None and not train_df.empty:
            dfs.append(train_df)
        if test_df is not None and not test_df.empty:
            dfs.append(test_df)

        if dfs:
            combined[sym] = __import__("pandas").concat(dfs).sort_index()

    return combined


if __name__ == "__main__":
    main()


    # Optionally drop bars from the first N minutes after NYSE open (9:30 ET).
    # Applied before the train/val/test split so all splits are filtered uniformly.
    skip_opening_minutes = int(cfg["prepare"].get("skip_opening_minutes", 0))
    if skip_opening_minutes > 0:
        cutoff_minutes_since_midnight = 9 * 60 + 30 + skip_opening_minutes
        for sym in list(ticker_dfs.keys()):
            df = ticker_dfs[sym]
            idx_et = df.index.tz_convert("America/New_York")
            minutes_since_midnight = idx_et.hour * 60 + idx_et.minute
            ticker_dfs[sym] = df[minutes_since_midnight >= cutoff_minutes_since_midnight]
        print(f"Skipping first {skip_opening_minutes} min after open (bars before 09:{30 + skip_opening_minutes:02d} ET dropped)")

    from kvant.experiment.prepare import build_default_components, prepare_experiment, PREPARED_DATA_ROOT

    # Split each ticker chronologically
    train_dfs, val_dfs, test_dfs = {}, {}, {}
    for sym, df in ticker_dfs.items():
        n       = len(df)
        n_test  = max(1, int(n * test_frac))
        n_val   = max(1, int(n * val_frac))
        n_train = n - n_val - n_test
        if n_train <= 0:
            print(f"  Skipping {sym}: only {n} bars — not enough for the split fractions.")
            continue
        train_dfs[sym] = df.iloc[:n_train].copy()
        val_dfs[sym]   = df.iloc[n_train: n_train + n_val].copy()
        test_dfs[sym]  = df.iloc[n_train + n_val:].copy()
        print(f"  {sym}: {n_train} train / {n_val} val / {n_test} test bars")

    if not train_dfs:
        raise SystemExit("No tickers had enough data after splitting.")

    sampler, fe, labeler, cfg = build_default_components(
        interval=interval,
        volatility_scaled_barrier=bool(cfg["prepare"].get("volatility_scaled_barrier", True)),
        vol_scale_min=float(cfg["prepare"].get("vol_scale_min", 0.5)),
        vol_scale_max=float(cfg["prepare"].get("vol_scale_max", 2.0)),
        lookback_L=int(cfg["prepare"].get("lookback", 20)),
        width_minutes=int(cfg["prepare"].get("width_minutes", 20)),
        height_pct=float(cfg["prepare"].get("height_pct", 0.5)),
        target_bars_per_day=int(cfg["prepare"].get("target_bars_per_day", 195)),
    )

    PREPARED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = prepare_experiment(
        out_root=PREPARED_DATA_ROOT,
        cfg=cfg,
        sampler=sampler,
        fe=fe,
        labeler=labeler,
        ticker_dfs_train=train_dfs,
        ticker_dfs_val=val_dfs,
        ticker_dfs_test=test_dfs,
        experiment_id=cfg.experiment_name,
    )
    (PREPARED_DATA_ROOT / "last_experiment.txt").write_text(manifest.exp_dir.name)


if __name__ == "__main__":
    main()
