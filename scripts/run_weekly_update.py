"""
scripts/run_weekly_update.py — Fetch Mon–Fri 1-minute bars from Yahoo Finance
and append them to the local CSV store. Maintains a status.toml tracking which
days have been retrieved and which are missing.

Designed to run on Saturday, Sunday, or Friday after NYSE close. Can also run
mid-week to pick up completed days so far. Safe to run multiple times —
already-fetched days are skipped automatically.

Usage
-----
  python scripts/run_weekly_update.py
  python scripts/run_weekly_update.py --config pipeline.toml
"""
import argparse
from pathlib import Path

from kvant.utils.pipeline_config import list_from_config, load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekly 1-minute data update — fetches Mon–Fri and appends to local CSVs."
    )
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()
    cfg, cfg_path = load_pipeline_config(args.config)

    from kvant.kdata.retriever import YahooRetriever
    from kvant.kdata.store import OHLCVStore

    symbols = list_from_config(cfg["data"].get("symbols"))
    store = str(cfg["paths"].get("store", "data/1m"))
    interval = str(cfg["data"].get("interval", "1m"))
    prepost = bool(cfg["data"].get("prepost", False))

    store_path = Path(store).resolve()

    # Default to all tickers already in the store
    if not symbols:
        existing = sorted(p.stem for p in store_path.glob("*.csv"))
        if not existing:
            raise SystemExit(f"No CSV files found in {store_path} and no symbols configured.")
        symbols = existing
        print(f"Auto-detected tickers from store: {symbols}")

    retriever = YahooRetriever(interval=interval, prepost=prepost)
    store_obj = OHLCVStore(store)

    print(f"Symbols : {', '.join(symbols)}")
    print(f"Config  : {cfg_path}")
    print(f"Store   : {store_path}")
    print(f"Interval: {interval}  |  prepost: {prepost}\n")

    report = store_obj.weekly_update(symbols, retriever, interval=interval)
    print(report)

    # Show path to status file for convenience
    print(f"\nStatus file: {store_path / 'status.toml'}")


if __name__ == "__main__":
    main()
