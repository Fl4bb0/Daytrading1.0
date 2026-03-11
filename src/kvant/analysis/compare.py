"""
analysis.compare — Compare multiple evaluation runs (different models / features).

compare_evaluations(eval_dirs, labels=None)  → Dict[str, pd.DataFrame]

Returned keys
-------------
"classification_metrics"  — metric × run wide table
"return_stats"            — stat × run wide table
"trade_stats"             — stacked table with a leading 'run' column
"equity_curves"           — timestamp × one cumulative_pnl_pct column per run
"label_distribution"      — stacked label distribution with a leading 'run' column
"run_meta"                — one row per run (concatenated run_meta.csv)
"confusion_matrices"      — dict[run_label → DataFrame]  (nested, not a flat DataFrame)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from kvant.analysis.loader import load_evaluation


def compare_evaluations(
    eval_dirs: List[Union[Path, str]],
    labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Load and compare evaluation results from multiple runs.

    Parameters
    ----------
    eval_dirs : list of paths, each produced by ``evaluate_experiment()``.
    labels    : human-readable run names; defaults to the directory names.

    Returns
    -------
    dict with keys described in the module docstring.
    """
    eval_dirs = [Path(d) for d in eval_dirs]
    if labels is None:
        labels = [d.name for d in eval_dirs]

    if len(labels) != len(eval_dirs):
        raise ValueError("`labels` length must match `eval_dirs` length.")

    loaded = {label: load_evaluation(d) for label, d in zip(labels, eval_dirs)}

    return {
        "classification_metrics": _compare_classification_metrics(loaded),
        "return_stats":           _compare_return_stats(loaded),
        "trade_stats":            _compare_trade_stats(loaded),
        "equity_curves":          _compare_equity_curves(loaded),
        "label_distribution":     _compare_label_distribution(loaded),
        "run_meta":               _compare_run_meta(loaded),
        "confusion_matrices":     {label: dfs["confusion_matrix"] for label, dfs in loaded.items()},
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compare_classification_metrics(
    loaded: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Wide table: rows = (class, metric), columns = run labels.
    Example columns: class, metric, run_A, run_B, …
    """
    frames = []
    for run_label, dfs in loaded.items():
        df = dfs["classification_metrics"].copy()
        # Melt all numeric columns into long form
        id_cols     = ["class"]
        value_cols  = [c for c in df.columns if c not in id_cols]
        melted = df.melt(id_vars=id_cols, value_vars=value_cols,
                         var_name="metric", value_name=run_label)
        frames.append(melted.set_index(["class", "metric"]))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).reset_index()
    return combined


def _compare_return_stats(
    loaded: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Wide table: rows = stat name, columns = run labels."""
    frames = []
    for run_label, dfs in loaded.items():
        df = dfs["return_stats"].copy()
        # return_stats is a single row; transpose to stat → value
        col = df.iloc[0].rename(run_label)
        frames.append(col)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined.index.name = "stat"
    return combined.reset_index()


def _compare_trade_stats(
    loaded: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Stacked table with a leading 'run' column."""
    parts = []
    for run_label, dfs in loaded.items():
        df = dfs["trade_stats"].copy()
        df.insert(0, "run", run_label)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _compare_equity_curves(
    loaded: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    One ``cumulative_pnl_pct_<run>`` column per run, aligned on timestamp.
    Rows without trades (empty equity curves) result in a NaN column.
    """
    merged: Optional[pd.DataFrame] = None

    for run_label, dfs in loaded.items():
        df = dfs["equity_curve"].copy()
        if df.empty or "timestamp" not in df.columns:
            continue
        df = (
            df[["timestamp", "cumulative_pnl_pct"]]
            .rename(columns={"cumulative_pnl_pct": f"cumulative_pnl_pct_{run_label}"})
        )
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="timestamp", how="outer")

    if merged is None:
        return pd.DataFrame()

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    # Forward-fill so each curve is defined for the full shared timeline
    pnl_cols = [c for c in merged.columns if c != "timestamp"]
    merged[pnl_cols] = merged[pnl_cols].ffill()
    return merged


def _compare_label_distribution(
    loaded: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Stacked label distribution with a leading 'run' column."""
    parts = []
    for run_label, dfs in loaded.items():
        df = dfs["label_distribution"].copy()
        df.insert(0, "run", run_label)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _compare_run_meta(
    loaded: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Concatenated run_meta rows with a leading 'run' column."""
    parts = []
    for run_label, dfs in loaded.items():
        df = dfs["run_meta"].copy()
        df.insert(0, "run", run_label)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
