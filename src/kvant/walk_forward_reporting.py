from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from kvant.evaluation.runner import (
    _compute_directional_calibration,
    _compute_directional_drift,
    _overall_directional_summary,
    _save_equity_curve,
)
from kvant.training.metrics import (
    classification_metrics,
    compute_action_profit_stats,
    compute_return_stats,
    per_ticker_trade_stats,
)

_LABEL_NAMES = ["SHORT", "HOLD", "BUY"]
_LABEL_IDS = [0, 1, 2]


def write_walk_forward_aggregate(
    *,
    aggregate_dir: Path,
    fold_rows: List[dict],
    fee: float,
    execution_priority: str,
    top_k_per_timestamp: Optional[int],
    ticker_cooldown_minutes: int,
) -> Path:
    aggregate_dir = Path(aggregate_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fold_rows).to_csv(aggregate_dir / "fold_summary.csv", index=False)

    pred_parts: List[pd.DataFrame] = []
    run_meta_rows: List[dict] = []
    for row in fold_rows:
        eval_dir = Path(row["eval_dir"])
        pred_path = eval_dir / "predictions.csv"
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
        if "bar_close_time" in pred_df.columns:
            pred_df["bar_close_time"] = pd.to_datetime(pred_df["bar_close_time"], errors="coerce")
        pred_df["fold_id"] = row["fold_id"]
        pred_parts.append(pred_df)

        meta_path = eval_dir / "run_meta.csv"
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            if not meta_df.empty:
                meta_row = meta_df.iloc[0].to_dict()
                meta_row["fold_id"] = row["fold_id"]
                run_meta_rows.append(meta_row)

    if not pred_parts:
        raise SystemExit("No fold prediction outputs were found to aggregate.")

    pred_df = pd.concat(pred_parts, ignore_index=True)
    pred_df = pred_df.sort_values(["timestamp", "fold_id", "ticker"], kind="stable").reset_index(drop=True)
    pred_df.to_csv(aggregate_dir / "predictions.csv", index=False)

    y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)

    report = classification_report(
        y_true,
        y_pred,
        labels=_LABEL_IDS,
        target_names=_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm_rows = []
    for key in [*_LABEL_NAMES, "macro avg", "weighted avg"]:
        if key in report:
            row = {"class": key}
            row.update(report[key])
            cm_rows.append(row)
    cm_rows.append({"class": "overall", **classification_metrics(y_true, y_pred)})
    pd.DataFrame(cm_rows).to_csv(aggregate_dir / "classification_metrics.csv", index=False)

    metas = _prediction_rows_to_metas(pred_df)
    ticker_codes, ticker_uniques = pd.factorize(pred_df["ticker"], sort=True)
    tids = ticker_codes.astype(np.int64)
    ticker_map = {int(idx): str(ticker) for idx, ticker in enumerate(ticker_uniques)}

    ts_stats = per_ticker_trade_stats(y_pred=y_pred, metas=metas, tids=tids)
    act_stats = compute_action_profit_stats(y_pred=y_pred, metas=metas, tids=tids)
    trade_rows = []
    for tid_int, ticker_sym in ticker_map.items():
        row: dict = {"tid": tid_int, "ticker": ticker_sym}
        row.update(ts_stats.get(tid_int, {
            "n_trades": 0,
            "bruto_profit_pct/avg": float("nan"),
            "accuracy_call_put/avg": float("nan"),
        }))
        row.update(act_stats.get(tid_int, {
            "buy/n_trades": 0,
            "buy/profit_pct/avg_per_trade": float("nan"),
            "buy/profit_pct/total": 0.0,
            "short/n_trades": 0,
            "short/profit_pct/avg_per_trade": float("nan"),
            "short/profit_pct/total": 0.0,
        }))
        trade_rows.append(row)
    pd.DataFrame(trade_rows).to_csv(aggregate_dir / "trade_stats.csv", index=False)

    ret_stats = compute_return_stats(y_pred=y_pred, metas=metas)
    ret_stats.update(_overall_directional_summary(pred_df))
    ret_stats["n_folds"] = int(len({row["fold_id"] for row in fold_rows}))
    pd.DataFrame([ret_stats]).to_csv(aggregate_dir / "return_stats.csv", index=False)

    _save_equity_curve(
        pred_df,
        aggregate_dir / "equity_curve.csv",
        fee=fee,
        execution_priority=execution_priority,
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
    )

    dist_rows = []
    for ticker_sym in sorted(ticker_map.values()):
        sub = pred_df[pred_df["ticker"] == ticker_sym]
        for label_id, label_name in zip(_LABEL_IDS, _LABEL_NAMES):
            dist_rows.append(
                {
                    "ticker": ticker_sym,
                    "label": label_name,
                    "label_id": label_id,
                    "y_true_count": int((sub["y_true"] == label_id).sum()),
                    "y_pred_count": int((sub["y_pred"] == label_id).sum()),
                }
            )
    pd.DataFrame(dist_rows).to_csv(aggregate_dir / "label_distribution.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=_LABEL_IDS)
    cm_df = pd.DataFrame(cm, index=_LABEL_NAMES, columns=_LABEL_NAMES)
    cm_df.index.name = "true \\ pred"
    cm_df.to_csv(aggregate_dir / "confusion_matrix.csv")

    _compute_directional_drift(pred_df).to_csv(aggregate_dir / "directional_drift.csv", index=False)
    calib = _compute_directional_calibration(pred_df)
    if calib is not None:
        calib.to_csv(aggregate_dir / "directional_calibration.csv", index=False)

    pd.DataFrame(run_meta_rows).to_csv(aggregate_dir / "fold_run_meta.csv", index=False)
    aggregate_run_meta = {
        "timestamp_run": datetime.now(tz=timezone.utc).isoformat(),
        "n_folds": int(len({row["fold_id"] for row in fold_rows})),
        "n_samples": int(len(pred_df)),
        "n_tickers": int(pred_df["ticker"].nunique()),
        "execution_priority": execution_priority,
        "top_k_per_timestamp": "" if top_k_per_timestamp is None else int(top_k_per_timestamp),
        "ticker_cooldown_minutes": int(ticker_cooldown_minutes),
        "fee": float(fee),
    }
    if run_meta_rows:
        first_meta = run_meta_rows[0]
        for key in ("model_name", "model_class_name", "split", "meta_enabled", "meta_train_split"):
            if key in first_meta:
                aggregate_run_meta[key] = first_meta[key]
    pd.DataFrame([aggregate_run_meta]).to_csv(aggregate_dir / "run_meta.csv", index=False)
    return aggregate_dir.resolve()


def _prediction_rows_to_metas(pred_df: pd.DataFrame) -> list[Optional[dict]]:
    metas: list[Optional[dict]] = []
    for row in pred_df.itertuples(index=False):
        pnl = getattr(row, "pnl_fraction", np.nan)
        if pd.isna(pnl):
            pnl = None
        bar_close_time = getattr(row, "bar_close_time", pd.NaT)
        metas.append(
            {
                "label": int(getattr(row, "y_true")),
                "pnl_fraction": None if pnl is None else float(pnl),
                "bar_close_time": None if pd.isna(bar_close_time) else pd.Timestamp(bar_close_time).isoformat(),
            }
        )
    return metas

