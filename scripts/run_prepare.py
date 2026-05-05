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

from kvant.utils.pipeline_config import load_pipeline_config
from kvant.utils.prepare_pipeline import (
    apply_prepare_filters,
    load_pipeline_ticker_dfs,
    should_use_partition_layout,
    split_ticker_dfs_by_fraction,
)

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = "1m"


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

    store_dir = Path(cfg["paths"].get("store", str(Path(__file__).resolve().parents[1] / "data" / "1m")))
    if not store_dir.exists():
        raise SystemExit(f"Store directory not found: {store_dir}")

    symbols = cfg["data"].get("symbols", [])
    interval = cfg["data"].get("interval", _DEFAULT_INTERVAL)
    val_frac = float(cfg["prepare"].get("val_frac", 0.15))
    test_frac = float(cfg["prepare"].get("test_frac", 0.15))
    brokerage_fee = float(cfg.get("trading", {}).get("brokerage_fee", 0.0008))
    num_workers = int(cfg["prepare"].get("num_workers", 1))

    # Decide which layout to use
    use_partition = args.use_partition or (not args.use_flat and should_use_partition_layout(cfg))

    print(f"Store    : {store_dir.resolve()}")
    print(f"Config   : {cfg_path}")
    print(f"Tickers  : {len(symbols)}")
    print(f"Layout   : {'month-partitioned' if use_partition else 'flat CSV'}\n")

    # Load DataFrames from the local store
    ticker_dfs = load_pipeline_ticker_dfs(
        cfg,
        force_partition=args.use_partition,
        force_flat=args.use_flat,
    )

    # Optionally drop bars from the first N minutes after NYSE open (9:30 ET).
    # Applied before the train/val/test split so all splits are filtered uniformly.
    skip_opening_minutes = int(cfg["prepare"].get("skip_opening_minutes", 0))
    ticker_dfs = apply_prepare_filters(ticker_dfs, cfg)
    if skip_opening_minutes > 0:
        print(
            f"Skipping first {skip_opening_minutes} min after open "
            f"(bars before 09:{30 + skip_opening_minutes:02d} ET dropped)"
        )

    from kvant.experiment.prepare import build_default_components, prepare_experiment, PREPARED_DATA_ROOT

    # Split each ticker chronologically
    train_dfs, val_dfs, test_dfs = split_ticker_dfs_by_fraction(
        ticker_dfs,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    for sym, train_df in train_dfs.items():
        print(f"  {sym}: {len(train_df)} train / {len(val_dfs[sym])} val / {len(test_dfs[sym])} test bars")

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
        brokerage_fee=brokerage_fee,
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
        num_workers=num_workers,
    )
    (PREPARED_DATA_ROOT / "last_experiment.txt").write_text(manifest.exp_dir.name)


if __name__ == "__main__":
    main()
