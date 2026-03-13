"""
note: this requires premium alpha vantage to get intraday data

scripts/run_daily_update.py — Quota-aware daily 1m sync for one ticker.

Examples
--------
python scripts/run_daily_update.py --ticker AAPL
python scripts/run_daily_update.py --ticker NVDA --store data/1m --state sync_state.json
python scripts/run_daily_update.py --ticker AMD --budget 25 --no-roll-forward
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one daily sync job for a single ticker.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to update or onboard.")
    parser.add_argument("--store", default="data/1m", help="Directory with per-ticker CSV files.")
    parser.add_argument("--state", default="sync_state.json", help="Sync-state filename under --store.")
    parser.add_argument("--budget", type=int, default=25, help="Max requests per day. Default: 25")
    parser.add_argument("--no-roll-forward", action="store_true", help="Skip window roll-forward when onboarding.")
    parser.add_argument("--prepost", action="store_true", help="Use Yahoo pre/post bars for recent day fetches.")
    args = parser.parse_args()

    from kvant.kdata.alpha_vantage_retriever import AlphaVantageError, AlphaVantagePlanError
    from kvant.kdata.retriever import AlphaVantageRetriever, HybridRetriever, YahooRetriever
    from kvant.kdata.sync import DailyTickerSync

    retriever = HybridRetriever(
        yahoo=YahooRetriever(interval="1m", period="7d", prepost=args.prepost),
        alpha=AlphaVantageRetriever(interval="1m"),
        recent_days=7,
    )
    syncer = DailyTickerSync(
        store_dir=args.store,
        retriever=retriever,
        state_file=args.state,
        budget_limit=args.budget,
        interval="1m",
    )

    try:
        report = syncer.run(args.ticker, roll_forward=not args.no_roll_forward)
    except AlphaVantagePlanError as exc:
        print(f"Alpha Vantage plan error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except AlphaVantageError as exc:
        print(f"Alpha Vantage error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    store_path = Path(args.store).resolve()
    print(f"Ticker  : {args.ticker}")
    print(f"Store   : {store_path}")
    print(f"State   : {store_path / args.state}")
    print(f"Mode    : {report.mode}")
    print(f"Requests: {report.requests_used} (remaining: {syncer.remaining_requests()})")
    if report.notes:
        print("Notes   :")
        for note in report.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()

