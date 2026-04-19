"""
scripts/run_hf_backfill.py — One-shot backfill of HuggingFace historical 1m data.

Imports all HuggingFace months for all configured symbols to local month partitions.
Idempotent: safe to re-run; skips months already present.

Usage
-----
python scripts/run_hf_backfill.py
python scripts/run_hf_backfill.py --config pipeline.toml
python scripts/run_hf_backfill.py --symbols AAPL MSFT
python scripts/run_hf_backfill.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from kvant.kdata.hf.import_month import import_month
from kvant.kdata.hf.month_store import _month_range
from kvant.utils.pipeline_config import list_from_config, load_pipeline_config

logger = logging.getLogger(__name__)


def _is_valid_month_str(month: str) -> bool:
    """Return True if month is YYYY-MM."""
    if not isinstance(month, str) or len(month) != 7:
        return False
    try:
        datetime.strptime(month, "%Y-%m")
        return True
    except ValueError:
        return False


def _parse_backfill_windows(data_sources_cfg: dict) -> List[Tuple[str, str]]:
    """
    Parse and validate [data_sources].backfill_time_windows from config.

    Expected TOML shape:
      backfill_time_windows = [
          { start = "2025-01", end = "2025-03" },
          { start = "2025-06", end = "2025-08" }
      ]
    """
    raw = data_sources_cfg.get("backfill_time_windows", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "data_sources.backfill_time_windows must be a list of {start, end} objects"
        )

    windows: List[Tuple[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"data_sources.backfill_time_windows[{i}] must be a table/object with 'start' and 'end'"
            )
        start = item.get("start")
        end = item.get("end")
        if not _is_valid_month_str(start) or not _is_valid_month_str(end):
            raise ValueError(
                f"Invalid month in backfill_time_windows[{i}]: start={start!r}, end={end!r}. "
                "Use YYYY-MM."
            )
        if start > end:
            raise ValueError(
                f"Invalid window order in backfill_time_windows[{i}]: start={start} > end={end}"
            )
        windows.append((start, end))
    return windows


def _resolve_backfill_months(
    *,
    data_sources_cfg: dict,
    cli_start_month: Optional[str],
    cli_end_month: Optional[str],
    hf_end_exclusive: str,
) -> tuple[List[str], List[dict]]:
    """
    Resolve months to import.

    Priority:
      1) CLI --start-month/--end-month (single continuous range)
      2) Config data_sources.backfill_time_windows (possibly multiple windows)
      3) Fallback default continuous range (2020-01 .. hf_end_exclusive[:7])

    Returns:
      (months_to_import, selected_windows_for_summary)
    """
    # CLI override mode
    if cli_start_month is not None or cli_end_month is not None:
        start_month = cli_start_month or "2020-01"
        end_month = cli_end_month or hf_end_exclusive[:7]
        months = _month_range(start_month, end_month)
        return months, [{"start": start_month, "end": end_month, "source": "cli_override"}]

    # Config windows mode
    windows = _parse_backfill_windows(data_sources_cfg)
    if windows:
        all_months: List[str] = []
        summary_windows: List[dict] = []
        for start, end in windows:
            months = _month_range(start, end)
            all_months.extend(months)
            summary_windows.append({"start": start, "end": end, "source": "config_window"})
        # de-dup + sorted for deterministic iteration if windows overlap
        uniq_months = sorted(set(all_months))
        return uniq_months, summary_windows

    # Fallback continuous range
    start_month = "2020-01"
    end_month = hf_end_exclusive[:7]
    months = _month_range(start_month, end_month)
    return months, [{"start": start_month, "end": end_month, "source": "default_range"}]


def run_hf_backfill(
    config_path: str = "pipeline.toml",
    symbols: Optional[list[str]] = None,
    dry_run: bool = False,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> dict:
    """
    Import HuggingFace months for symbols.

    Month selection:
      - Uses CLI start/end if passed.
      - Else uses [data_sources].backfill_time_windows if configured.
      - Else falls back to default range.
    """
    cfg, cfg_path_resolved = load_pipeline_config(config_path)

    if symbols is None:
        symbols = list_from_config(cfg["data"].get("symbols", []))

    data_sources = cfg.get("data_sources", {})
    hf_end_exclusive = data_sources.get("hf_end_exclusive", "2026-01-01")

    months_to_import, selected_windows = _resolve_backfill_months(
        data_sources_cfg=data_sources,
        cli_start_month=start_month,
        cli_end_month=end_month,
        hf_end_exclusive=hf_end_exclusive,
    )

    logger.info(f"Config  : {cfg_path_resolved}")
    logger.info(f"Symbols : {len(symbols)}")
    logger.info(f"Dry-run : {dry_run}")
    logger.info(f"Windows : {selected_windows}")
    logger.info(f"Months  : {len(months_to_import)}")
    logger.info("")

    results = {}
    for i, month in enumerate(months_to_import, 1):
        logger.info(f"[{i}/{len(months_to_import)}] {month}")

        if dry_run:
            logger.info("  (DRY-RUN: skipping write)")
            result = {
                "month": month,
                "status": "skipped",
                "reason": "dry_run",
            }
        else:
            result = import_month(
                month,
                symbols=symbols,
                config_path=config_path,
            )

        results[month] = result

        if "results" in result:
            ok_count = sum(1 for r in result["results"].values() if r.get("status") == "ok")
            error_count = sum(1 for r in result["results"].values() if r.get("status") == "error")
            no_data_count = sum(1 for r in result["results"].values() if r.get("status") == "no_data")
            logger.info(f"  OK: {ok_count}, Errors: {error_count}, No data: {no_data_count}")

    summary = {
        "status": "complete",
        "timestamp": datetime.utcnow().isoformat(),
        "config": str(cfg_path_resolved),
        "dry_run": dry_run,
        "selected_windows": selected_windows,
        "months_processed": len(months_to_import),
        "symbols": symbols,
        "month_results": results,
    }

    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="One-shot backfill of HuggingFace 1m data.",
    )
    parser.add_argument(
        "--config",
        default="pipeline.toml",
        help="Path to pipeline.toml",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to backfill (default: all from config)",
    )
    parser.add_argument(
        "--start-month",
        default=None,
        help="Start month in YYYY-MM format (default: from config)",
    )
    parser.add_argument(
        "--end-month",
        default=None,
        help="End month in YYYY-MM format (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log but don't write",
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

    # Run backfill
    result = run_hf_backfill(
        config_path=args.config,
        symbols=args.symbols,
        dry_run=args.dry_run,
        start_month=args.start_month,
        end_month=args.end_month,
    )

    print("\n" + "=" * 80)
    print("BACKFILL SUMMARY")
    print("=" * 80)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
