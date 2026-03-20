"""
note: this requires premium alpha vantage to get intraday data

scripts/run_daily_update.py — Quota-aware daily 1m sync for one ticker.

Examples
--------
python scripts/run_daily_update.py
python scripts/run_daily_update.py --config pipeline.toml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from kvant.utils.pipeline_config import list_from_config, load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily sync jobs for configured tickers.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    args = parser.parse_args()
    cfg, cfg_path = load_pipeline_config(args.config)

    tickers = list_from_config(cfg["data"].get("symbols"))
    if not tickers:
        raise SystemExit("No symbols configured in [data].symbols in pipeline.toml")

    store = cfg["paths"].get("store", "data/1m")
    state = cfg["paths"].get("sync_state", "sync_state.json")
    budget = int(cfg["data"].get("daily_budget", 25))
    interval = str(cfg["data"].get("interval", "1m"))
    prepost = bool(cfg["data"].get("prepost", False))
    roll_forward = bool(cfg.get("daily", {}).get("roll_forward", True))
    recent_days = int(cfg.get("daily", {}).get("recent_days", 7))

    from kvant.kdata.alpha_vantage_retriever import AlphaVantageError, AlphaVantagePlanError
    from kvant.kdata.retriever import AlphaVantageRetriever, HybridRetriever, YahooRetriever
    from kvant.kdata.sync import DailyTickerSync

    retriever = HybridRetriever(
        yahoo=YahooRetriever(interval=interval, period="7d", prepost=prepost),
        alpha=AlphaVantageRetriever(interval=interval),
        recent_days=recent_days,
    )
    syncer = DailyTickerSync(
        store_dir=store,
        retriever=retriever,
        state_file=state,
        budget_limit=budget,
        interval=interval,
    )

    store_path = Path(store).resolve()
    print(f"Config  : {cfg_path}")
    print(f"Store   : {store_path}")
    print(f"State   : {store_path / state}")
    print(f"Tickers : {', '.join(tickers)}\n")

    for ticker in tickers:
        try:
            report = syncer.run(ticker, roll_forward=roll_forward)
        except AlphaVantagePlanError as exc:
            print(f"[{ticker}] Alpha Vantage plan error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        except AlphaVantageError as exc:
            print(f"[{ticker}] Alpha Vantage error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        print(f"Ticker  : {ticker}")
        print(f"Mode    : {report.mode}")
        print(f"Requests: {report.requests_used} (remaining: {syncer.remaining_requests()})")
        if report.notes:
            print("Notes   :")
            for note in report.notes:
                print(f"  - {note}")
        print("")


if __name__ == "__main__":
    main()

