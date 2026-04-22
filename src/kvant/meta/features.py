"""
meta.features — Build causal, side-aware ranking features from prediction rows.

These features sit on top of the base classifier outputs and are intended for
trade ranking, not raw price forecasting.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

META_FEATURE_COLUMNS = [
    "side_confidence",
    "side",
    "ticker_side_prior_score",
    "ticker_side_prior_n",
]
META_TARGET_COLUMN = "target_signed_net_pnl"

_DIRECTIONAL_LABELS = {0, 2}
_SIDE_BY_LABEL = {0: -1.0, 2: 1.0}


def _prepare_rows(df: pd.DataFrame, *, source: str, fee: float) -> pd.DataFrame:
    work = df.copy()
    timestamps = work["timestamp"] if "timestamp" in work.columns else pd.Series([None] * len(work), index=work.index)
    close_times = work["bar_close_time"] if "bar_close_time" in work.columns else pd.Series([None] * len(work), index=work.index)
    tickers = work["ticker"] if "ticker" in work.columns else pd.Series([""] * len(work), index=work.index)

    work["timestamp"] = pd.to_datetime(timestamps, errors="coerce")
    work["bar_close_time"] = pd.to_datetime(close_times, errors="coerce")
    work["ticker"] = tickers.astype(str)
    work["y_pred"] = pd.to_numeric(work.get("y_pred"), errors="coerce")
    work["pnl_fraction"] = pd.to_numeric(work.get("pnl_fraction"), errors="coerce")

    side = work["y_pred"].map(_SIDE_BY_LABEL).astype(float)
    work["side"] = side

    side_confidence = np.full(len(work), np.nan, dtype=float)
    short_mask = work["y_pred"] == 0
    buy_mask = work["y_pred"] == 2

    if "prob_SHORT" in work.columns:
        side_confidence[short_mask.to_numpy()] = pd.to_numeric(
            work.loc[short_mask, "prob_SHORT"],
            errors="coerce",
        ).to_numpy(dtype=float)
    if "prob_BUY" in work.columns:
        side_confidence[buy_mask.to_numpy()] = pd.to_numeric(
            work.loc[buy_mask, "prob_BUY"],
            errors="coerce",
        ).to_numpy(dtype=float)
    work["side_confidence"] = side_confidence

    target = side.to_numpy(dtype=float) * work["pnl_fraction"].to_numpy(dtype=float)
    target = target - (2.0 * float(fee))
    target[~np.isin(work["y_pred"].to_numpy(dtype=float), list(_DIRECTIONAL_LABELS))] = np.nan
    target[~np.isfinite(work["pnl_fraction"].to_numpy(dtype=float))] = np.nan
    work[META_TARGET_COLUMN] = target

    work["_source"] = source
    work["_source_rank"] = 0 if source == "history" else 1
    work["_row_id"] = np.arange(len(work), dtype=np.int64)
    work["_entry_order"] = np.arange(len(work), dtype=np.int64)
    return work


def add_meta_features(
    pred_df: pd.DataFrame,
    *,
    history_df: Optional[pd.DataFrame] = None,
    fee: float = 0.0,
    shrinkage_k: float = 10.0,
) -> pd.DataFrame:
    """
    Add minimal causal meta features to *pred_df*.

    The historical prior only updates once a prior trade's ``bar_close_time`` is
    at or before the current row's entry ``timestamp``.
    """
    shrinkage_k = float(shrinkage_k)
    if shrinkage_k < 0.0:
        raise ValueError("shrinkage_k must be >= 0")

    current = _prepare_rows(pred_df, source="current", fee=float(fee))
    combined_parts = [current]
    if history_df is not None and len(history_df) > 0:
        combined_parts.insert(0, _prepare_rows(history_df, source="history", fee=float(fee)))

    combined = pd.concat(combined_parts, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["timestamp", "_source_rank", "_entry_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    side_values = current["side"].to_numpy(dtype=float, copy=True)
    side_conf_values = current["side_confidence"].to_numpy(dtype=float, copy=True)
    prior_score_values = np.full(len(current), np.nan, dtype=float)
    prior_n_values = np.full(len(current), np.nan, dtype=float)
    target_values = current[META_TARGET_COLUMN].to_numpy(dtype=float, copy=True)

    local_sum: dict[tuple[str, int], float] = defaultdict(float)
    local_count: dict[tuple[str, int], int] = defaultdict(int)
    global_sum: dict[int, float] = defaultdict(float)
    global_count: dict[int, int] = defaultdict(int)
    pending: list[tuple[pd.Timestamp, int, str, int, float]] = []
    pending_counter = 0

    def resolve_until(ts: pd.Timestamp) -> None:
        while pending and pending[0][0] <= ts:
            _, _, ticker, label, value = heapq.heappop(pending)
            key = (ticker, label)
            local_sum[key] += float(value)
            local_count[key] += 1
            global_sum[label] += float(value)
            global_count[label] += 1

    for _, row in combined.iterrows():
        entry_ts = row["timestamp"]
        if not pd.isna(entry_ts):
            resolve_until(entry_ts)

        label_raw = row["y_pred"]
        label = int(label_raw) if not pd.isna(label_raw) else None
        if row["_source"] == "current" and label in _DIRECTIONAL_LABELS:
            row_id = int(row["_row_id"])
            key = (str(row["ticker"]), int(label))
            local_n = int(local_count.get(key, 0))
            local_sum_value = float(local_sum.get(key, 0.0))
            global_n = int(global_count.get(int(label), 0))
            global_mean = (
                float(global_sum.get(int(label), 0.0)) / float(global_n)
                if global_n > 0
                else 0.0
            )
            denom = float(local_n) + shrinkage_k
            score = global_mean if denom <= 0.0 else (local_sum_value + shrinkage_k * global_mean) / denom
            prior_score_values[row_id] = float(score)
            prior_n_values[row_id] = float(local_n)

        if label in _DIRECTIONAL_LABELS:
            target_value = float(row[META_TARGET_COLUMN])
            exit_ts = row["bar_close_time"]
            if np.isfinite(target_value) and not pd.isna(exit_ts):
                heapq.heappush(
                    pending,
                    (pd.Timestamp(exit_ts), pending_counter, str(row["ticker"]), int(label), target_value),
                )
                pending_counter += 1

    out = pred_df.copy()
    out["side_confidence"] = side_conf_values
    out["side"] = side_values
    out["ticker_side_prior_score"] = prior_score_values
    out["ticker_side_prior_n"] = prior_n_values
    out[META_TARGET_COLUMN] = target_values
    return out


def build_meta_training_frame(
    pred_df: pd.DataFrame,
    *,
    history_df: Optional[pd.DataFrame] = None,
    fee: float = 0.0,
    shrinkage_k: float = 10.0,
) -> pd.DataFrame:
    """Return the trainable directional subset with all feature columns present."""
    meta_df = add_meta_features(
        pred_df,
        history_df=history_df,
        fee=fee,
        shrinkage_k=shrinkage_k,
    )
    keep_mask = meta_df["y_pred"].isin(sorted(_DIRECTIONAL_LABELS))
    keep_mask &= pd.to_numeric(meta_df[META_TARGET_COLUMN], errors="coerce").notna()
    for col in META_FEATURE_COLUMNS:
        keep_mask &= pd.to_numeric(meta_df[col], errors="coerce").notna()
    return meta_df.loc[keep_mask].copy()
