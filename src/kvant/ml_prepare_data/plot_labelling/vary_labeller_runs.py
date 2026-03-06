from __future__ import annotations

import json
import pickle
import shutil
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

from kvant.kdata.retriever import HuggingFaceRetriever, YahooRetriever
# your existing components
from kvant.ml_prepare_data.prepare_experiment import (
    prepare_experiment,
    ExperimentConfig,
)
from kvant.ml_prepare_data.features.feature_engineering import OHLCVFeatures
from kvant.ml_prepare_data.labelling.tripple_bar import TripleBarrierLabeler
from kvant.ml_prepare_data.samplers.sampling import IdentitySampler


# IMPORTANT: use your real import path for this
# (user example: exp = PreparedExperiment(exp_dir))
from kvant.ml_prepare_data.data_loading import PreparedExperiment  # adjust if needed


TB_CLASSES = ("0", "1", "2")


def _stable_sweep_exp_id(prefix: str, payload: dict) -> str:
    """
    Stable (deterministic) id so reruns overwrite same folder if desired.
    Kept short to be path-friendly.
    """
    b = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    h = hashlib.sha256(b).hexdigest()[:16]
    return f"{prefix}_{h}"


# def _load_json(path: Path) -> dict:
#     return json.loads(path.read_text())


# def _empty_class_counts() -> Dict[str, int]:
#     return {k: 0 for k in TB_CLASSES}

# from typing import Any, Dict
#
# TB_CLASSES = ("0", "1", "2")  # keep your existing convention

def _empty_class_counts() -> Dict[str, int]:
    return {k: 0 for k in TB_CLASSES}


def _extract_per_ticker_counts_from_prepared(exp: PreparedExperiment) -> dict[str, Any]:
    """
    Extract per-ticker class counts for train/val/test using the dataset's `summary()`.

    This relies on:
      - exp.get_datasets() producing IndexWindowDataset instances
      - IndexWindowDataset.summary(display=False) returning:
          {
            "overall": {"n": int, "y_counts": {0:int,1:int,2:int}, "first_ts": str|None, "last_ts": str|None},
            "per_ticker": {
               "<TICKER>": {"tid": int, "n": int, "y_counts": {0:int,1:int,2:int}, "first_ts": str|None, "last_ts": str|None}
            }
          }

    Note: index_*.npy is already filtered by (label != -1) and (pos >= lookback_L),
          so counts are over valid targets only.
    """
    ds_train, ds_val, ds_test = exp.get_datasets()
    datasets = {"train": ds_train, "val": ds_val, "test": ds_test}

    tickers_all: list[str] = list(exp.store.tickers_all)

    # Initialize output structure with all tickers (even if absent in a split)
    per_ticker: dict[str, Any] = {
        t: {
            "train": {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
            "val":   {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
            "test":  {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
        }
        for t in tickers_all
    }

    totals: dict[str, Any] = {
        "train": {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
        "val":   {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
        "test":  {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
    }

    # Fill from dataset summaries
    for split, ds in datasets.items():
        s = ds.summary(display=False)

        # overall / totals
        overall = s.get("overall", {}) or {}
        totals[split]["n"] = int(overall.get("n", 0) or 0)
        totals[split]["first_ts"] = overall.get("first_ts", None)
        totals[split]["last_ts"] = overall.get("last_ts", None)

        y_counts_overall = overall.get("y_counts", {}) or {}
        for cls_int in (0, 1, 2):
            k = str(cls_int)
            if k not in TB_CLASSES:
                continue
            totals[split]["class_counts"][k] = int(y_counts_overall.get(cls_int, 0) or 0)

        # per ticker
        per = s.get("per_ticker", {}) or {}
        for ticker, row in per.items():
            # be defensive if summary includes tickers not in tickers_all for some reason
            if ticker not in per_ticker:
                per_ticker[ticker] = {
                    "train": {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
                    "val":   {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
                    "test":  {"class_counts": _empty_class_counts(), "n": 0, "first_ts": None, "last_ts": None},
                }

            per_ticker[ticker][split]["n"] = int(row.get("n", 0) or 0)
            per_ticker[ticker][split]["first_ts"] = row.get("first_ts", None)
            per_ticker[ticker][split]["last_ts"] = row.get("last_ts", None)

            y_counts = row.get("y_counts", {}) or {}
            for cls_int in (0, 1, 2):
                k = str(cls_int)
                if k not in TB_CLASSES:
                    continue
                per_ticker[ticker][split]["class_counts"][k] = int(y_counts.get(cls_int, 0) or 0)

    return {
        "tickers_all": tickers_all,
        "per_ticker": per_ticker,
        "totals": totals,
    }


def run_sweep_and_save_pkl(
    *,
    ticker_data_train: Dict[str, Any],
    ticker_data_val: Dict[str, Any],
    ticker_data_test: Dict[str, Any],
    out_root_prepared: Path,
    out_pkl_path: Path,
    width_minutes_grid: List[int],
    height_grid: List[float],
    lookback_L: int = 200,
    subsample_every: int = 1,
    drop_time_exit_label: bool = False,
) -> Path:

    sampler = IdentitySampler(subsample_every=subsample_every)
    fe = OHLCVFeatures(cols=("open", "high", "low", "close", "volume"), log1p_volume=True)

    runs: list[dict[str, Any]] = []
    grid = [(w, h) for w in width_minutes_grid for h in height_grid]

    for (w, h) in grid:
        # --- build config ---
        labeler = TripleBarrierLabeler(
            name=f"tb_w{w}_h{h}",
            width_minutes=int(w),
            height=float(h),
            drop_time_exit_label=bool(drop_time_exit_label),
        )

        cfg = ExperimentConfig(
            experiment_name="sweep_tb_label_stats",
            sampler=asdict(sampler),
            feature_engineer=asdict(fe),
            labeler=asdict(labeler),
            lookback_L=int(lookback_L),
        )

        sweep_payload = {
            "sampler": asdict(sampler),
            "feature_engineer": asdict(fe),
            "labeler": asdict(labeler),
            "lookback_L": int(lookback_L),
        }
        exp_id = _stable_sweep_exp_id("tmp_sweep", sweep_payload)
        exp_dir = out_root_prepared / exp_id

        # --- run standard preparation (writes to disk) ---
        prepare_experiment(
            out_root=out_root_prepared,
            cfg=cfg,
            sampler=sampler,
            fe=fe,
            labeler=labeler,
            ticker_dfs_train=ticker_data_train,
            ticker_dfs_val=ticker_data_val,
            ticker_dfs_test=ticker_data_test,
            experiment_id=exp_id,
        )

        # --- load via PreparedExperiment (as requested) ---
        # We don't depend on its internal structure for stats; we just ensure it loads.
        exp = PreparedExperiment(exp_dir)
        # _ds_train, _ds_val, _ds_test = exp.get_datasets()

        # --- extract stats from written artifacts ---
        stats = _extract_per_ticker_counts_from_prepared(exp)


        runs.append({
            "params": {
                "width_minutes": int(w),
                "height": float(h),
                "drop_time_exit_label": bool(drop_time_exit_label),
            },
            "experiment_id": exp_id,
            "stats": stats,
        })

        # --- cleanup ---
        shutil.rmtree(exp_dir, ignore_errors=True)

    payload = {
        "schema_version": 1,
        "tb_classes": list(TB_CLASSES),
        "grid": {
            "width_minutes": list(width_minutes_grid),
            "height": list(height_grid),
        },
        "lookback_L": int(lookback_L),
        "subsample_every": int(subsample_every),
        "drop_time_exit_label": bool(drop_time_exit_label),
        "runs": runs,
    }

    out_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    return out_pkl_path


def _default_sweep_kwargs(out_root_prepared: Path, out_pkl_path: Path) -> dict:
    return dict(
        out_root_prepared=out_root_prepared,
        out_pkl_path=out_pkl_path,
        width_minutes_grid=[60, 120, 180],
        height_grid=[0.015, 0.02, 0.03],
        lookback_L=5,
        subsample_every=1,
        drop_time_exit_label=False,
    )


def main_hf():
    """Run labeller sweep using HuggingFace minute-bar data."""
    from kvant.ml_prepare_data import prepared_data_root

    hf = HuggingFaceRetriever()
    downloaded_splits = hf.get_splits(preset="small")
    ticker_data_train, ticker_data_val, ticker_data_test = hf.get_ticker_data(
        [], downloaded_dataset=downloaded_splits[-1]
    )

    out_root_prepared = prepared_data_root
    out_pkl_path = Path("prepared") / "sweep_tb_label_stats.pkl"

    pkl_path = run_sweep_and_save_pkl(
        ticker_data_train=ticker_data_train,
        ticker_data_val=ticker_data_val,
        ticker_data_test=ticker_data_test,
        **_default_sweep_kwargs(out_root_prepared, out_pkl_path),
    )
    print("Wrote:", pkl_path)


def main_yahoo(
    symbols: List[str] = None,
    period: str = "6mo",
    interval: str = "1d",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """Run labeller sweep using Yahoo Finance data.

    Args:
        symbols:   Ticker symbols to fetch. Defaults to a small liquid set.
        period:    yfinance period string, e.g. ``"6mo"``, ``"2y"``.
        interval:  Bar interval, e.g. ``"1d"``, ``"1h"``, ``"5m"``.
        val_frac:  Fraction of each ticker's history reserved for validation.
        test_frac: Fraction reserved for testing (taken from the end).
    """
    import pandas as pd
    from kvant.ml_prepare_data import prepared_data_root

    if symbols is None:
        symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    all_data = YahooRetriever().get_ticker_data(symbols, period=period, interval=interval)

    ticker_data_train: Dict[str, pd.DataFrame] = {}
    ticker_data_val: Dict[str, pd.DataFrame] = {}
    ticker_data_test: Dict[str, pd.DataFrame] = {}

    for sym, df in all_data.items():
        n = len(df)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val - n_test
        if n_train <= 0:
            print(f"Skipping {sym}: not enough rows ({n}) for the requested split fractions.")
            continue
        ticker_data_train[sym] = df.iloc[:n_train]
        ticker_data_val[sym] = df.iloc[n_train: n_train + n_val]
        ticker_data_test[sym] = df.iloc[n_train + n_val:]

    if not ticker_data_train:
        raise RuntimeError("No tickers had enough data to form a train split.")

    out_root_prepared = prepared_data_root
    out_pkl_path = Path("prepared") / "sweep_tb_label_stats.pkl"

    pkl_path = run_sweep_and_save_pkl(
        ticker_data_train=ticker_data_train,
        ticker_data_val=ticker_data_val,
        ticker_data_test=ticker_data_test,
        **_default_sweep_kwargs(out_root_prepared, out_pkl_path),
    )
    print("Wrote:", pkl_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Labeller parameter sweep.")
    parser.add_argument(
        "source",
        choices=["hf", "yahoo"],
        help="Data source: 'hf' for HuggingFace, 'yahoo' for Yahoo Finance.",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Ticker symbols (yahoo only).")
    parser.add_argument("--period", default="6mo", help="yfinance period string (yahoo only).")
    parser.add_argument("--interval", default="1d", help="Bar interval (yahoo only).")
    parser.add_argument("--val-frac", type=float, default=0.15, dest="val_frac", help="Validation fraction (yahoo only).")
    parser.add_argument("--test-frac", type=float, default=0.15, dest="test_frac", help="Test fraction (yahoo only).")
    args = parser.parse_args()

    if args.source == "hf":
        main_hf()
    else:
        main_yahoo(
            symbols=args.symbols,
            period=args.period,
            interval=args.interval,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )

