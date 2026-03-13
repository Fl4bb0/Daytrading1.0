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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekly 1-minute data update — fetches Mon–Fri and appends to local CSVs."
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True, metavar="TICKER",
        help="Yahoo Finance ticker symbols, e.g. AAPL MSFT NVDA.",
    )
    parser.add_argument(
        "--store", default="data/1m", metavar="DIR",
        help="Directory where per-ticker CSVs and status.toml are kept. Default: data/1m",
    )
    parser.add_argument(
        "--interval", default="1m",
        help="Bar interval to fetch. Default: 1m",
    )
    parser.add_argument(
        "--prepost", action="store_true",
        help="Include pre- and post-market bars.",
    )
    args = parser.parse_args()

    from kvant.kdata.retriever import YahooRetriever
    from kvant.kdata.store import OHLCVStore

    retriever = YahooRetriever(interval=args.interval, prepost=args.prepost)
    store = OHLCVStore(args.store)

    store_path = Path(args.store).resolve()
    print(f"Symbols : {', '.join(args.symbols)}")
    print(f"Store   : {store_path}")
    print(f"Interval: {args.interval}  |  prepost: {args.prepost}\n")

    report = store.weekly_update(args.symbols, retriever, interval=args.interval)
    print(report)

    # Show path to status file for convenience
    print(f"\nStatus file: {store_path / 'status.toml'}")


if __name__ == "__main__":
    main()
