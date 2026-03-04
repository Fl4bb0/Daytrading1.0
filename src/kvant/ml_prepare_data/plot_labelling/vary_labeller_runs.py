from __future__ import annotations

import json
import pickle
import shutil
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np

from kvant.kdata.hf_minute_data import DownloadedDatasetSplit
# your existing components
from src.kvant.ml_prepare_data.prepare_experiment import (
    prepare_experiment,
    ExperimentConfig,
    OHLCVFeatures,
)
from kvant.ml_prepare_data.labelling.tripple_bar import TripleBarrierLabeler
from kvant.ml_prepare_data.samplers import IdentitySampler

from src.kvant.kdata.hf_minute_data import (
    get_huggingface_top_4_tiny_splits,
    get_ticker_data,
)

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
    downloaded_splits : list[DownloadedDatasetSplit],
    out_root_prepared: Path,
    out_pkl_path: Path,
    width_minutes_grid: List[int],
    height_grid: List[float],
    lookback_L: int = 200,
    subsample_every: int = 1,
    drop_time_exit_label: bool = False,
) -> Path:
    if downloaded_splits is None:
        print("Downloaded splits are none, giving them default.")
        assert False
        # downloaded_splits = get_huggingface_top_4_tiny_splits()

    dataset_split = downloaded_splits[-1]
    ticker_data_train, ticker_data_val, ticker_data_test = get_ticker_data(dataset_split)

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


def main():
    from kvant.ml_prepare_data import prepared_data_root
    from kvant.ml_prepare_data.prepare_experiment import get_huggingface_top_5_small_splits

    out_root_prepared = prepared_data_root # Path("prepared")  # same as you use normally
    out_pkl_path = Path("prepared") / "sweep_tb_label_stats.pkl"
    downloaded_splits = get_huggingface_top_5_small_splits()

    # requested start
    width_minutes_grid = [60, 120, 180]
    height_grid = [0.015, 0.02, 0.03]  # if you want more resolution: [0.2, 0.25, 0.3]

    pkl_path = run_sweep_and_save_pkl(
        downloaded_splits=downloaded_splits,
        out_root_prepared=out_root_prepared,
        out_pkl_path=out_pkl_path,
        width_minutes_grid=width_minutes_grid,
        height_grid=height_grid,
        lookback_L=5,
        subsample_every=1,
        drop_time_exit_label=False,
    )
    print("Wrote:", pkl_path)


if __name__ == "__main__":
    main()