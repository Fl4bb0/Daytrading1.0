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

from kvant.utils.pipeline_config import load_pipeline_config


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Path to pipeline TOML config.")
    pre_args, remaining = pre_parser.parse_known_args()
    cfg, cfg_path = load_pipeline_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Run one daily sync job for a single ticker.", parents=[pre_parser])
    parser.add_argument("--ticker", required=True, help="Ticker symbol to update or onboard.")
    parser.add_argument("--store", default=cfg["paths"].get("store", "data/1m"), help="Directory with per-ticker CSV files.")
    parser.add_argument("--state", default=cfg["paths"].get("sync_state", "sync_state.json"), help="Sync-state filename under --store.")
    parser.add_argument("--budget", type=int, default=int(cfg["data"].get("daily_budget", 25)), help="Max requests per day.")
    parser.add_argument("--no-roll-forward", action="store_true", help="Skip window roll-forward when onboarding.")
    parser.add_argument(
        "--prepost",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg["data"].get("prepost", False)),
        help="Use Yahoo pre/post bars for recent day fetches.",
    )
    parser.add_argument("--interval", default=cfg["data"].get("interval", "1m"), help="Bar interval, e.g. 1m.")
    args = parser.parse_args(remaining)

    from kvant.kdata.alpha_vantage_retriever import AlphaVantageError, AlphaVantagePlanError
    from kvant.kdata.retriever import AlphaVantageRetriever, HybridRetriever, YahooRetriever
    from kvant.kdata.sync import DailyTickerSync

    retriever = HybridRetriever(
        yahoo=YahooRetriever(interval=args.interval, period="7d", prepost=args.prepost),
        alpha=AlphaVantageRetriever(interval=args.interval),
        recent_days=7,
    )
    syncer = DailyTickerSync(
        store_dir=args.store,
        retriever=retriever,
        state_file=args.state,
        budget_limit=args.budget,
        interval=args.interval,
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
    print(f"Config  : {cfg_path}")
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

