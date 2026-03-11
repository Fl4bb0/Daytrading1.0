"""
analysis.loader — Load all evaluation CSVs produced by evaluation.runner.

Public API
----------
load_evaluation(eval_dir)  → Dict[str, pd.DataFrame]

The returned dict keys match the CSV stem names:
    "predictions", "classification_metrics", "trade_stats",
    "return_stats", "equity_curve", "label_distribution",
    "confusion_matrix", "run_meta"
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

EXPECTED_CSVS = [
    "predictions",
    "classification_metrics",
    "trade_stats",
    "return_stats",
    "equity_curve",
    "label_distribution",
    "confusion_matrix",
    "run_meta",
]


def load_evaluation(eval_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all evaluation CSVs from *eval_dir* into a dict of DataFrames.

    Parameters
    ----------
    eval_dir : Path produced by ``evaluate_experiment()``.

    Returns
    -------
    Dict mapping CSV stem → DataFrame.

    Raises
    ------
    FileNotFoundError  if *eval_dir* does not exist.
    ValueError         if any expected CSV is missing.
    """
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    missing = [
        stem for stem in EXPECTED_CSVS
        if not (eval_dir / f"{stem}.csv").exists()
    ]
    if missing:
        raise ValueError(
            f"Missing CSV files in {eval_dir}: {missing}\n"
            "Make sure evaluate_experiment() ran successfully."
        )

    dfs: Dict[str, pd.DataFrame] = {}
    for stem in EXPECTED_CSVS:
        path = eval_dir / f"{stem}.csv"
        df   = pd.read_csv(path)
        # Parse timestamp columns where present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        dfs[stem] = df

    return dfs
