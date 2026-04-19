"""
kdata.import_month — Import one month of HuggingFace 1m data to local partitions.

Provides programmatic API and CLI interface for importing monthly data shards.

Usage (CLI)
-----------
  python -m kvant.kdata.import_month 2025-03
  python -m kvant.kdata.import_month 2025-03 --symbols AAPL MSFT
  python -m kvant.kdata.import_month 2025-03 --config pipeline.toml

Behavior
--------
- Fetches data for requested month from HuggingFace
- Writes to data/1m/{YYYY-MM}/{ticker}.csv
- Skips if month already exists (idempotent)
- Logs progress and summary
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Optional

from kvant.kdata.hf.huggingface_retriever import HuggingFaceRetriever
from kvant.kdata.hf.month_store import MonthPartitionedStore
from kvant.utils.pipeline_config import list_from_config, load_pipeline_config

logger = logging.getLogger(__name__)


def import_month(
    year_month: str,
    symbols: Optional[list[str]] = None,
    retriever: Optional[HuggingFaceRetriever] = None,
    store: Optional[MonthPartitionedStore] = None,
    config_path: str = "pipeline.toml",
) -> dict:
    """
    Import one month of 1m data from HuggingFace to local month partition.

    Parameters
    ----------
    year_month : str
        Month in 'YYYY-MM' format (e.g., '2025-03').
    symbols : list[str], optional
        Symbols to import. If None, use all from config.
    retriever : HuggingFaceRetriever, optional
        If None, initialized from config.
    store : MonthPartitionedStore, optional
        If None, initialized from config.
    config_path : str
        Path to pipeline.toml.

    Returns
    -------
    dict
        Summary: {
          "month": "...",
          "status": "complete" | "skipped",
          "timestamp": ISO datetime,
          "symbols": [...],
          "results": {symbol: {status, appended, skipped, ...}},
        }

    Raises
    ------
    ValueError
        If year_month format is invalid.
    """
    # Validate month format
    try:
        datetime.strptime(year_month, "%Y-%m")
    except ValueError:
        raise ValueError(f"Invalid month format. Use YYYY-MM. Got: {year_month}")

    # Load config
    cfg, cfg_path_resolved = load_pipeline_config(config_path)

    # Initialize retriever if not provided
    if retriever is None:
        hf_config = cfg.get("hf_config", {})
        dataset_id = hf_config.get("dataset_id", "")
        cache_dir = hf_config.get("cache_dir", "~/.cache/huggingface")
        retriever = HuggingFaceRetriever(
            dataset_id=dataset_id,
            cache_dir=cache_dir,
        )

    # Initialize store if not provided
    if store is None:
        paths = cfg.get("paths", {})
        store_dir = paths.get("store", "data/1m")
        store = MonthPartitionedStore(store_dir)

    # Get symbols if not provided
    if symbols is None:
        symbols = list_from_config(cfg["data"].get("symbols", []))

    logger.info(f"Importing {year_month} for {len(symbols)} symbols...")

    # Check if month already fully imported
    existing_months = store.list_months()
    if year_month in existing_months:
        existing_syms = set(store.list_symbols_in_month(year_month))
        requested_syms = set(symbols)
        if existing_syms >= requested_syms:
            logger.info(f"Month {year_month} already fully imported; skipping.")
            return {
                "month": year_month,
                "status": "skipped",
                "reason": "already_exists",
                "timestamp": datetime.utcnow().isoformat(),
            }

    # Import each symbol
    results = {}
    for sym in symbols:
        try:
            logger.info(f"  Fetching {sym}/{year_month}...")
            df = retriever.get_month_shard(sym, year_month)

            if df.empty:
                results[sym] = {"status": "no_data"}
                logger.warning(f"    No data for {sym}/{year_month}")
                continue

            report = store.append_month(sym, year_month, df, skip_existing=True)
            results[sym] = {"status": "ok", **report}
            logger.info(
                f"    {sym}: appended={report['appended']}, "
                f"skipped={report['skipped']}, total={report['total']}"
            )

        except Exception as e:
            results[sym] = {"status": "error", "error": str(e)}
            logger.error(f"    Error fetching {sym}/{year_month}: {e}", exc_info=True)

    return {
        "month": year_month,
        "status": "complete",
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": symbols,
        "results": results,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Import one month of 1m data from HuggingFace to local partitions.",
    )
    parser.add_argument(
        "year_month",
        help="Month in YYYY-MM format (e.g., 2025-03)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to import (default: all from config)",
    )
    parser.add_argument(
        "--config",
        default="pipeline.toml",
        help="Path to pipeline.toml",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run import
    result = import_month(
        args.year_month,
        symbols=args.symbols,
        config_path=args.config,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
