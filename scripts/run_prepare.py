"""
scripts/run_prepare.py — Entry-point for data preparation.

By default loads every ticker CSV found in data/1m (written by
run_weekly_update.py) and runs the full preparation pipeline.

Usage
-----
  # Use all tickers in data/1m (default)
  python scripts/run_prepare.py

  # Restrict to specific tickers
  python scripts/run_prepare.py --symbols AAPL MSFT

  # Override store directory or split fractions
  python scripts/run_prepare.py --store data/1m --val-frac 0.15 --test-frac 0.15
"""
import argparse
from pathlib import Path

from kvant.utils.pipeline_config import list_from_config, load_pipeline_config


# Default store written by run_weekly_update.py
_DEFAULT_STORE = Path(__file__).resolve().parents[1] / "data" / "1m"
_DEFAULT_INTERVAL = "1m"


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Path to pipeline TOML config.")
    pre_args, remaining = pre_parser.parse_known_args()
    cfg, cfg_path = load_pipeline_config(pre_args.config)

    default_symbols = list_from_config(cfg["data"].get("symbols"))
    if default_symbols == []:
        default_symbols = None

    parser = argparse.ArgumentParser(
        description="Prepare an ML experiment dataset.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--store", default=cfg["paths"].get("store", str(_DEFAULT_STORE)), metavar="DIR",
        help="Directory with per-ticker CSVs from run_weekly_update.py.",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=default_symbols,
        help="Restrict to these tickers. Default: config symbols or all CSVs in --store.",
    )
    parser.add_argument("--interval", type=str, default=cfg["data"].get("interval", _DEFAULT_INTERVAL))
    parser.add_argument("--val-frac",  type=float, default=float(cfg["prepare"].get("val_frac", 0.15)))
    parser.add_argument("--test-frac", type=float, default=float(cfg["prepare"].get("test_frac", 0.15)))
    parser.add_argument("--lookback", type=int, default=int(cfg["prepare"].get("lookback", 20)))
    parser.add_argument("--width-minutes", type=int, default=int(cfg["prepare"].get("width_minutes", 20)))
    parser.add_argument("--height-pct", type=float, default=float(cfg["prepare"].get("height_pct", 0.5)))
    parser.add_argument(
        "--target-bars-per-day",
        type=int,
        default=int(cfg["prepare"].get("target_bars_per_day", 195)),
    )
    parser.add_argument(
        "--volatility-scaled-barrier",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg["prepare"].get("volatility_scaled_barrier", True)),
    )
    parser.add_argument("--vol-scale-min", type=float, default=float(cfg["prepare"].get("vol_scale_min", 0.5)))
    parser.add_argument("--vol-scale-max", type=float, default=float(cfg["prepare"].get("vol_scale_max", 2.0)))
    args = parser.parse_args(remaining)

    store_dir = Path(args.store)
    if not store_dir.exists():
        raise SystemExit(f"Store directory not found: {store_dir}")

    # Discover available tickers from CSV filenames
    available = sorted(p.stem for p in store_dir.glob("*.csv"))
    if not available:
        raise SystemExit(f"No CSV files found in {store_dir}")

    symbols = args.symbols if args.symbols else available
    missing = [s for s in symbols if s not in available]
    if missing:
        raise SystemExit(f"Tickers not found in {store_dir}: {missing}")

    print(f"Store    : {store_dir.resolve()}")
    print(f"Config   : {cfg_path}")
    print(f"Tickers  : {symbols}")

    # Load DataFrames from the local store
    from kvant.kdata.store import OHLCVStore
    store = OHLCVStore(store_dir)
    ticker_dfs = store.load_all(symbols)

    from kvant.experiment.prepare import build_default_components, prepare_experiment, PREPARED_DATA_ROOT

    # Split each ticker chronologically
    train_dfs, val_dfs, test_dfs = {}, {}, {}
    for sym, df in ticker_dfs.items():
        n       = len(df)
        n_test  = max(1, int(n * args.test_frac))
        n_val   = max(1, int(n * args.val_frac))
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
        interval=args.interval,
        volatility_scaled_barrier=args.volatility_scaled_barrier,
        vol_scale_min=args.vol_scale_min,
        vol_scale_max=args.vol_scale_max,
        lookback_L=args.lookback,
        width_minutes=args.width_minutes,
        height_pct=args.height_pct,
        target_bars_per_day=args.target_bars_per_day,
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
