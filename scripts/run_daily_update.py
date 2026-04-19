"""
scripts/run_daily_update.py — Daily 1m data sync for configured tickers.

Supports both legacy flat CSV layout (data/1m/{ticker}.csv) and new
month-partitioned layout (data/1m/{YYYY-MM}/{ticker}.csv).

With hybrid pipeline enabled (hf_config.dataset_id set), performs:
  - Yahoo incremental updates to current month partition only
  - Does NOT run HF backfill (use scripts/run_hf_backfill.py for that)

Examples
--------
python scripts/run_daily_update.py
python scripts/run_daily_update.py --config pipeline.toml
python scripts/run_daily_update.py --use-partition     # Force new layout
python scripts/run_daily_update.py --use-flat          # Force legacy layout
"""
from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path
import sys

import pandas as pd

from kvant.utils.pipeline_config import list_from_config, load_pipeline_config

logger = logging.getLogger(__name__)


def _should_use_partition_layout(cfg: dict) -> bool:
    """Determine if should use month-partitioned layout."""
    hf_config = cfg.get("hf_config", {})
    dataset_id = hf_config.get("dataset_id", "")
    return bool(dataset_id)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily 1m data sync for configured tickers.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    parser.add_argument(
        "--use-partition",
        action="store_true",
        help="Force month-partitioned layout (requires HF dataset)",
    )
    parser.add_argument(
        "--use-flat",
        action="store_true",
        help="Force legacy flat CSV layout",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg, cfg_path = load_pipeline_config(args.config)

    tickers = list_from_config(cfg["data"].get("symbols"))
    if not tickers:
        raise SystemExit("No symbols configured in [data].symbols in pipeline.toml")

    store_dir = cfg["paths"].get("store", "data/1m")

    # Decide which layout to use
    use_partition = args.use_partition or (
        not args.use_flat and _should_use_partition_layout(cfg)
    )

    print(f"Config  : {cfg_path}")
    print(f"Store   : {Path(store_dir).resolve()}")
    print(f"Tickers : {len(tickers)}")
    print(f"Layout  : {'month-partitioned' if use_partition else 'flat CSV'}")
    print()

    if use_partition:
        # New hybrid pipeline with month partitions (Yahoo only)
        _run_hybrid_daily_update(cfg, store_dir, tickers)
    else:
        # Legacy flat CSV layout (existing code)
        _run_legacy_daily_update(cfg, store_dir, tickers)


def _run_legacy_daily_update(cfg: dict, store_dir: str, tickers: list[str]) -> None:
    """Legacy daily update using flat CSV layout."""
    print("Using legacy flat CSV layout...\n")

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
        store_dir=store_dir,
        retriever=retriever,
        state_file=state,
        budget_limit=budget,
        interval=interval,
    )

    store_path = Path(store_dir).resolve()
    print(f"State   : {store_path / state}\n")

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


def _run_hybrid_daily_update(
    cfg: dict, store_dir: str, tickers: list[str]
) -> None:
    """New hybrid daily update using month-partitioned layout.

    Appends Yahoo data to current month. Does NOT run HF backfill
    (use scripts/run_hf_backfill.py for that).
    """
    from kvant.kdata.hf.month_store import MonthPartitionedStore
    from kvant.kdata.retriever import YahooRetriever

    print("Using month-partitioned layout with Yahoo incremental updates...\n")

    # Initialize retriever and store
    hf_config = cfg.get("hf_config", {})
    dataset_id = hf_config.get("dataset_id", "")

    if not dataset_id:
        print(
            "ERROR: hf_config.dataset_id not set in pipeline.toml\n"
            "Set it to enable hybrid pipeline, or use --use-flat for legacy layout."
        )
        raise SystemExit(1)

    yahoo_retriever = YahooRetriever(
        interval=cfg["data"].get("interval", "1m"),
        period="7d",
        prepost=cfg["data"].get("prepost", False),
    )

    store = MonthPartitionedStore(store_dir)
    interval = cfg["data"].get("interval", "1m")

    # Get current month
    now = pd.Timestamp.now(tz="UTC")
    current_month = now.strftime("%Y-%m")

    print(f"Current month: {current_month}")
    print(f"Appending Yahoo data to current month...\n")

    # Daily Yahoo update for each ticker
    for ticker in tickers:
        try:
            # Load current month
            df_current = store.load_month(ticker, current_month)

            # Get last timestamp
            last_ts = df_current.index[-1] if not df_current.empty else None

            # Fetch Yahoo from last_ts to now
            if last_ts:
                start = last_ts + timedelta(seconds=1)
            else:
                # No data yet, fetch last 7 days
                start = now - timedelta(days=7)

            logger.info(f"[{ticker}] Fetching Yahoo from {start} to {now}")
            df_yahoo = yahoo_retriever.get_history(
                ticker,
                start=start,
                end=now,
                interval=interval,
                as_pandas=True,
            )

            if df_yahoo.empty:
                logger.info(f"[{ticker}] No new data")
                print(f"{ticker}: No new data")
            else:
                # Append to current month
                report = store.append_month(ticker, current_month, df_yahoo, skip_existing=True)
                logger.info(f"[{ticker}] Appended {report['appended']}, skipped {report['skipped']}")
                print(
                    f"{ticker}: appended={report['appended']}, "
                    f"skipped={report['skipped']}, total={report['total']}"
                )

        except Exception as e:
            logger.error(f"[{ticker}] Error: {e}", exc_info=True)
            print(f"{ticker}: ERROR - {e}")


if __name__ == "__main__":
    main()

