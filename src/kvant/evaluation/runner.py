"""
evaluation.runner — End-to-end inference + statistics, saving results as CSV.

evaluate_experiment(exp_dir, model_path, model_cls, out_dir, split="test")

Output CSVs (written to out_dir/)
-----------------------------------
predictions.csv         — per-sample: timestamp, ticker, y_true, y_pred, [proba cols]
classification_metrics.csv — accuracy / precision / recall / F1 per class + macro avg
trade_stats.csv         — per-ticker trade statistics (profit, accuracy, counts)
return_stats.csv        — overall split-level return / profit statistics
equity_curve.csv        — cumulative portfolio PnL over time
label_distribution.csv  — per-ticker count of y_true and y_pred per class
confusion_matrix.csv    — 3×3 confusion matrix (rows=true, cols=pred)
directional_drift.csv   — per-ticker directional drift diagnostics
directional_calibration.csv — confidence-binned directional calibration (if probabilities exist)
run_meta.csv            — experiment metadata (model, split, timestamp, counts)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from kvant.evaluation.loader import load_split
from kvant.models.base import KvantModel
from kvant.training.metrics import (
    classification_metrics,
    per_ticker_trade_stats,
    compute_return_stats,
    compute_action_profit_stats,
)

# Label names (must match label convention: 0=SHORT, 1=HOLD, 2=BUY)
_LABEL_NAMES = ["SHORT", "HOLD", "BUY"]
_LABEL_IDS   = [0, 1, 2]


def evaluate_experiment(
    exp_dir: Path,
    model_path: Path,
    model_cls: Type[KvantModel],
    out_dir: Path,
    split: str = "test",
    tickers: Optional[List[str]] = None,
) -> Path:
    """
    Load artifacts, run inference, compute statistics, and save all results as CSV.

    Parameters
    ----------
    exp_dir    : Path to a prepared experiment directory.
    model_path : Path to a saved model checkpoint (directory or file).
    model_cls  : KvantModel subclass — must implement ``load(path)``.
    out_dir    : Directory where all CSV files will be written.
    split      : Which split to evaluate — ``"train"``, ``"val"``, or ``"test"``.
    tickers    : Optional allowlist of ticker symbols to evaluate.

    Returns
    -------
    out_dir (as resolved Path)
    """
    exp_dir    = Path(exp_dir)
    model_path = Path(model_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load artifacts from disk
    # ------------------------------------------------------------------
    X, y, timestamps, tids, metas, ticker_map = load_split(exp_dir, split, tickers)
    n_samples  = int(len(y))
    n_tickers  = int(len(ticker_map))

    # ------------------------------------------------------------------
    # 2. Load model and run inference
    # ------------------------------------------------------------------
    model    = model_cls.load(model_path)
    y_pred   = model.predict(X)

    proba: Optional[np.ndarray] = None
    try:
        proba = model.predict_proba(X)
    except NotImplementedError:
        pass

    # Ticker symbol per sample
    ticker_labels = np.array([ticker_map[int(tid)] for tid in tids])

    # Readable timestamps
    try:
        ts_pd = pd.to_datetime(timestamps)
    except Exception:
        ts_pd = pd.RangeIndex(n_samples)

    # ------------------------------------------------------------------
    # 3. predictions.csv
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "timestamp": ts_pd,
        "ticker":    ticker_labels,
        "y_true":    y,
        "y_pred":    y_pred,
        "y_true_name": [_LABEL_NAMES[int(v)] if int(v) in _LABEL_IDS else str(v) for v in y],
        "y_pred_name": [_LABEL_NAMES[int(v)] if int(v) in _LABEL_IDS else str(v) for v in y_pred],
    })
    if proba is not None and proba.shape[1] == 3:
        pred_df["prob_SHORT"] = proba[:, 0]
        pred_df["prob_HOLD"]  = proba[:, 1]
        pred_df["prob_BUY"]   = proba[:, 2]

    # Add pnl_fraction from metadata for downstream convenience
    pnl_col = [
        m.get("pnl_fraction") if isinstance(m, dict) else None
        for m in metas
    ]
    pred_df["pnl_fraction"] = pnl_col
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 4. classification_metrics.csv
    # ------------------------------------------------------------------
    report = classification_report(
        y, y_pred,
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
    # Also add overall accuracy
    overall = classification_metrics(y, y_pred)
    cm_rows.append({"class": "overall", **overall})
    pd.DataFrame(cm_rows).to_csv(out_dir / "classification_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 5. trade_stats.csv
    # ------------------------------------------------------------------
    ts_stats   = per_ticker_trade_stats(y_pred=y_pred, metas=metas, tids=tids)
    act_stats  = compute_action_profit_stats(y_pred=y_pred, metas=metas, tids=tids)

    trade_rows = []
    for tid_int, ticker_sym in ticker_map.items():
        row: dict = {"tid": tid_int, "ticker": ticker_sym}
        if tid_int in ts_stats:
            row.update(ts_stats[tid_int])
        else:
            row.update({"n_trades": 0, "bruto_profit_pct/avg": float("nan"), "accuracy_call_put/avg": float("nan")})
        if tid_int in act_stats:
            row.update(act_stats[tid_int])
        else:
            row.update({
                "buy/n_trades": 0, "buy/profit_pct/avg_per_trade": float("nan"),
                "buy/profit_pct/total": 0.0,
                "short/n_trades": 0, "short/profit_pct/avg_per_trade": float("nan"),
                "short/profit_pct/total": 0.0,
            })
        trade_rows.append(row)

    pd.DataFrame(trade_rows).to_csv(out_dir / "trade_stats.csv", index=False)

    # ------------------------------------------------------------------
    # 6. return_stats.csv
    # ------------------------------------------------------------------
    ret_stats = compute_return_stats(y_pred=y_pred, metas=metas)
    ret_stats.update(_overall_directional_summary(pred_df))
    pd.DataFrame([ret_stats]).to_csv(out_dir / "return_stats.csv", index=False)

    # ------------------------------------------------------------------
    # 7. equity_curve.csv
    # ------------------------------------------------------------------
    _save_equity_curve(pred_df, out_dir / "equity_curve.csv")

    # ------------------------------------------------------------------
    # 8. label_distribution.csv
    # ------------------------------------------------------------------
    dist_rows = []
    for ticker_sym in sorted(ticker_map.values()):
        mask = pred_df["ticker"] == ticker_sym
        sub  = pred_df[mask]
        for label_id, label_name in zip(_LABEL_IDS, _LABEL_NAMES):
            dist_rows.append({
                "ticker":       ticker_sym,
                "label":        label_name,
                "label_id":     label_id,
                "y_true_count": int((sub["y_true"] == label_id).sum()),
                "y_pred_count": int((sub["y_pred"] == label_id).sum()),
            })
    pd.DataFrame(dist_rows).to_csv(out_dir / "label_distribution.csv", index=False)

    # ------------------------------------------------------------------
    # 9. confusion_matrix.csv
    # ------------------------------------------------------------------
    cm = confusion_matrix(y, y_pred, labels=_LABEL_IDS)
    cm_df = pd.DataFrame(cm, index=_LABEL_NAMES, columns=_LABEL_NAMES)
    cm_df.index.name = "true \\ pred"
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    # ------------------------------------------------------------------
    # 10. directional drift + calibration reports
    # ------------------------------------------------------------------
    _compute_directional_drift(pred_df).to_csv(out_dir / "directional_drift.csv", index=False)

    calib = _compute_directional_calibration(pred_df)
    if calib is not None:
        calib.to_csv(out_dir / "directional_calibration.csv", index=False)

    # ------------------------------------------------------------------
    # 11. run_meta.csv
    # ------------------------------------------------------------------
    experiment_id = exp_dir.name
    run_meta = {
        "experiment_id":   experiment_id,
        "model_name":      model_cls.__name__,
        "model_path":      str(model_path),
        "split":           split,
        "timestamp_run":   datetime.now(tz=timezone.utc).isoformat(),
        "n_samples":       n_samples,
        "n_tickers":       n_tickers,
        "tickers":         ",".join(sorted(ticker_map.values())),
    }
    pd.DataFrame([run_meta]).to_csv(out_dir / "run_meta.csv", index=False)

    print(
        f"[evaluate_experiment] Saved {len(list(out_dir.glob('*.csv')))} CSVs → {out_dir}\n"
        f"  split={split}, n_samples={n_samples}, n_tickers={n_tickers}"
    )
    return out_dir.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_equity_curve(pred_df: pd.DataFrame, out_path: Path) -> None:
    """
    Build and save a cumulative portfolio PnL equity curve.

    Only BUY (y_pred=2) and SHORT (y_pred=0) predictions that have a
    ``pnl_fraction`` value in the metadata contribute a trade.
    The curve is sorted by timestamp and PnL is accumulated in sequence.
    """
    rows = []
    for _, row in pred_df.sort_values("timestamp").iterrows():
        yp       = int(row["y_true"])       # use y_true for ideal curve?  No: use y_pred.
        yp       = int(row["y_pred"])
        pnl_frac = row.get("pnl_fraction")
        if not isinstance(pnl_frac, (int, float)) or np.isnan(float(pnl_frac)):
            continue
        if yp not in (0, 2):
            continue
        signed_pnl = (-1.0 if yp == 0 else 1.0) * float(pnl_frac)
        rows.append({
            "timestamp":      row["timestamp"],
            "ticker":         row["ticker"],
            "action":         _LABEL_NAMES[yp],
            "trade_pnl_pct":  signed_pnl * 100.0,
        })

    if not rows:
        pd.DataFrame(columns=[
            "timestamp", "ticker", "action", "trade_pnl_pct", "cumulative_pnl_pct"
        ]).to_csv(out_path, index=False)
        return

    eq_df = pd.DataFrame(rows)
    eq_df["cumulative_pnl_pct"] = eq_df["trade_pnl_pct"].cumsum()
    eq_df.to_csv(out_path, index=False)


def _direction_from_label(s: pd.Series) -> pd.Series:
    return s.map({0: -1, 1: 0, 2: 1}).astype(float)


def _overall_directional_summary(pred_df: pd.DataFrame) -> dict:
    d = pred_df[pred_df["y_true"].isin([0, 2])]
    if len(d) == 0:
        return {
            "directional_true_n": 0,
            "directional_accuracy": 0.0,
            "directional_opposite_rate": 0.0,
        }
    opposite = ((d["y_true"] == 0) & (d["y_pred"] == 2)) | ((d["y_true"] == 2) & (d["y_pred"] == 0))
    return {
        "directional_true_n": int(len(d)),
        "directional_accuracy": float((d["y_true"] == d["y_pred"]).mean()),
        "directional_opposite_rate": float(opposite.mean()),
    }


def _compute_directional_drift(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = [("ALL", pred_df), *[(str(t), g) for t, g in pred_df.groupby("ticker")]]
    for ticker, sub in grouped:
        dir_true = _direction_from_label(sub["y_true"])
        dir_pred = _direction_from_label(sub["y_pred"])
        true_dir_mask = sub["y_true"].isin([0, 2])
        pred_dir_mask = sub["y_pred"].isin([0, 2])

        directional = sub[true_dir_mask]
        opposite = ((directional["y_true"] == 0) & (directional["y_pred"] == 2)) | ((directional["y_true"] == 2) & (directional["y_pred"] == 0))

        signed_pnl_pct = np.nan
        has_pnl = sub["pnl_fraction"].notna() & pred_dir_mask
        if bool(has_pnl.any()):
            signed = np.where(sub.loc[has_pnl, "y_pred"] == 0, -1.0, 1.0) * sub.loc[has_pnl, "pnl_fraction"].astype(float)
            signed_pnl_pct = float(np.mean(signed) * 100.0)

        rows.append({
            "ticker": ticker,
            "n": int(len(sub)),
            "n_true_directional": int(true_dir_mask.sum()),
            "n_pred_directional": int(pred_dir_mask.sum()),
            "true_short_rate": float((sub["y_true"] == 0).mean()),
            "true_buy_rate": float((sub["y_true"] == 2).mean()),
            "pred_short_rate": float((sub["y_pred"] == 0).mean()),
            "pred_buy_rate": float((sub["y_pred"] == 2).mean()),
            "direction_bias_pred_minus_true": float((dir_pred - dir_true).mean()),
            "directional_accuracy": float((directional["y_true"] == directional["y_pred"]).mean()) if len(directional) else 0.0,
            "directional_opposite_rate": float(opposite.mean()) if len(directional) else 0.0,
            "avg_signed_pnl_pct_pred": signed_pnl_pct,
        })
    return pd.DataFrame(rows)


def _compute_directional_calibration(pred_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    pcols = ["prob_SHORT", "prob_HOLD", "prob_BUY"]
    if not all(c in pred_df.columns for c in pcols):
        return None

    d = pred_df[pred_df["y_pred"].isin([0, 2])].copy()
    if len(d) == 0:
        return pd.DataFrame(columns=[
            "ticker", "confidence_bin", "n", "avg_confidence",
            "directional_accuracy", "directional_opposite_rate", "avg_signed_pnl_pct",
        ])

    conf = np.where(d["y_pred"] == 0, d["prob_SHORT"], d["prob_BUY"])
    d["confidence"] = conf.astype(float)
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    d["confidence_bin"] = pd.cut(d["confidence"], bins=bins, include_lowest=True)

    rows = []
    grouped = [("ALL", d), *[(str(t), g) for t, g in d.groupby("ticker")]]
    for ticker, sub in grouped:
        for b, g in sub.groupby("confidence_bin", observed=False):
            if len(g) == 0:
                continue
            opposite = ((g["y_true"] == 0) & (g["y_pred"] == 2)) | ((g["y_true"] == 2) & (g["y_pred"] == 0))
            has_pnl = g["pnl_fraction"].notna()
            avg_signed_pnl_pct = np.nan
            if bool(has_pnl.any()):
                signed = np.where(g.loc[has_pnl, "y_pred"] == 0, -1.0, 1.0) * g.loc[has_pnl, "pnl_fraction"].astype(float)
                avg_signed_pnl_pct = float(np.mean(signed) * 100.0)
            rows.append({
                "ticker": ticker,
                "confidence_bin": str(b),
                "n": int(len(g)),
                "avg_confidence": float(g["confidence"].mean()),
                "directional_accuracy": float((g["y_true"] == g["y_pred"]).mean()),
                "directional_opposite_rate": float(opposite.mean()),
                "avg_signed_pnl_pct": avg_signed_pnl_pct,
            })
    return pd.DataFrame(rows)

