"""
scripts/run_weekly_update.py — Fetch Mon–Fri 1-minute bars from Yahoo Finance
and append them to the local CSV store. Maintains a status.toml tracking which
days have been retrieved and which are missing.

Designed to run on Saturday, Sunday, or Friday after NYSE close. Can also run
mid-week to pick up completed days so far. Safe to run multiple times —
already-fetched days are skipped automatically.

Usage
-----
  python scripts/run_weekly_update.py --symbols AAPL MSFT NVDA
  python scripts/run_weekly_update.py --symbols AAPL --store data/1m --prepost
  python scripts/run_weekly_update.py --symbols AAPL --store data/1m --interval 1m
"""
import argparse
from pathlib import Path

from kvant.utils.pipeline_config import list_from_config, load_pipeline_config


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Path to pipeline TOML config.")
    pre_args, remaining = pre_parser.parse_known_args()
    cfg, cfg_path = load_pipeline_config(pre_args.config)

    default_symbols = list_from_config(cfg["data"].get("symbols"))
    if default_symbols == []:
        default_symbols = None

    parser = argparse.ArgumentParser(
        description="Weekly 1-minute data update — fetches Mon–Fri and appends to local CSVs.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--symbols", nargs="+", default=default_symbols, metavar="TICKER",
        help="Yahoo Finance ticker symbols. Default: config symbols or all tickers already in --store.",
    )
    parser.add_argument(
        "--store", default=cfg["paths"].get("store", "data/1m"), metavar="DIR",
        help="Directory where per-ticker CSVs and status.toml are kept.",
    )
    parser.add_argument(
        "--interval", default=cfg["data"].get("interval", "1m"),
        help="Bar interval to fetch.",
    )
    parser.add_argument(
        "--prepost", action=argparse.BooleanOptionalAction,
        default=bool(cfg["data"].get("prepost", False)),
        help="Include pre- and post-market bars.",
    )
    args = parser.parse_args(remaining)

    from kvant.kdata.retriever import YahooRetriever
    from kvant.kdata.store import OHLCVStore

    store_path = Path(args.store).resolve()

    # Default to all tickers already in the store
    if args.symbols is None:
        existing = sorted(p.stem for p in store_path.glob("*.csv"))
        if not existing:
            raise SystemExit(f"No CSV files found in {store_path} and no --symbols given.")
        args.symbols = existing
        print(f"Auto-detected tickers from store: {args.symbols}")

    retriever = YahooRetriever(interval=args.interval, prepost=args.prepost)
    store = OHLCVStore(args.store)

    print(f"Symbols : {', '.join(args.symbols)}")
    print(f"Config  : {cfg_path}")
    print(f"Store   : {store_path}")
    print(f"Interval: {args.interval}  |  prepost: {args.prepost}\n")

    report = store.weekly_update(args.symbols, retriever, interval=args.interval)
    print(report)

    # Show path to status file for convenience
    print(f"\nStatus file: {store_path / 'status.toml'}")


if __name__ == "__main__":
    main()
