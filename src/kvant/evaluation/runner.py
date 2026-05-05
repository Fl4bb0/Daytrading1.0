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

from dataclasses import dataclass
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
_EXECUTION_PRIORITIES = {"first_seen", "model_confidence", "meta_score"}


@dataclass
class PredictionArtifacts:
    X: np.ndarray
    pred_df: pd.DataFrame
    y: np.ndarray
    y_pred: np.ndarray
    y_pred_raw: np.ndarray
    tids: np.ndarray
    metas: List[Optional[dict]]
    ticker_map: dict[int, str]
    n_samples: int
    n_tickers: int


def _build_prediction_artifacts(
    exp_dir: Path,
    model_path: Path,
    model_cls: Type[KvantModel],
    *,
    split: str,
    tickers: Optional[List[str]],
    required_buy_probability: float,
    required_sell_probability: float,
    model: Optional[KvantModel],
) -> PredictionArtifacts:
    """Load a split, run base-model inference, and return aligned prediction rows."""
    X, y, timestamps, tids, metas, ticker_map = load_split(exp_dir, split, tickers)
    n_samples = int(len(y))
    n_tickers = int(len(ticker_map))

    model = model if model is not None else model_cls.load(model_path)
    y_pred_raw = model.predict(X)

    proba: Optional[np.ndarray] = None
    try:
        proba = model.predict_proba(X)
    except NotImplementedError:
        pass

    y_pred = _apply_action_probability_thresholds(
        y_pred=y_pred_raw,
        proba=proba,
        required_sell_probability=required_sell_probability,
        required_buy_probability=required_buy_probability,
    )

    ticker_labels = np.array([ticker_map[int(tid)] for tid in tids])
    try:
        ts_pd = pd.to_datetime(timestamps)
    except Exception:
        ts_pd = pd.RangeIndex(n_samples)

    pred_df = pd.DataFrame(
        {
            "timestamp": ts_pd,
            "ticker": ticker_labels,
            "y_true": y,
            "y_pred_raw": y_pred_raw,
            "y_pred": y_pred,
            "y_true_name": [_LABEL_NAMES[int(v)] if int(v) in _LABEL_IDS else str(v) for v in y],
            "y_pred_raw_name": [_LABEL_NAMES[int(v)] if int(v) in _LABEL_IDS else str(v) for v in y_pred_raw],
            "y_pred_name": [_LABEL_NAMES[int(v)] if int(v) in _LABEL_IDS else str(v) for v in y_pred],
        }
    )
    if proba is not None and proba.shape[1] == 3:
        pred_df["prob_SHORT"] = proba[:, 0]
        pred_df["prob_HOLD"] = proba[:, 1]
        pred_df["prob_BUY"] = proba[:, 2]

    pred_df["pnl_fraction"] = [
        m.get("pnl_fraction") if isinstance(m, dict) else None
        for m in metas
    ]
    pred_df["bar_close_time"] = pd.to_datetime(
        [
            m.get("bar_close_time") if isinstance(m, dict) else None
            for m in metas
        ],
        errors="coerce",
    )

    return PredictionArtifacts(
        X=X,
        pred_df=pred_df,
        y=y,
        y_pred=y_pred,
        y_pred_raw=y_pred_raw,
        tids=tids,
        metas=metas,
        ticker_map=ticker_map,
        n_samples=n_samples,
        n_tickers=n_tickers,
    )


def build_prediction_frame(
    exp_dir: Path,
    model_path: Path,
    model_cls: Type[KvantModel],
    *,
    split: str = "test",
    tickers: Optional[List[str]] = None,
    required_buy_probability: float = 0.0,
    required_sell_probability: float = 0.0,
    model: Optional[KvantModel] = None,
) -> pd.DataFrame:
    """Public helper for generating aligned prediction rows without full evaluation."""
    return _build_prediction_artifacts(
        exp_dir=exp_dir,
        model_path=model_path,
        model_cls=model_cls,
        split=split,
        tickers=tickers,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        model=model,
    ).pred_df


def evaluate_experiment(
    exp_dir: Path,
    model_path: Path,
    model_cls: Type[KvantModel],
    out_dir: Path,
    split: str = "test",
    tickers: Optional[List[str]] = None,
    fee: float = 0.0,
    n_pools: int = 10,
    required_buy_probability: float = 0.0,
    required_sell_probability: float = 0.0,
    allow_short: bool = True,
    execution_priority: str = "model_confidence",
    top_k_per_timestamp: Optional[int] = None,
    ticker_cooldown_minutes: int = 0,
    model: Optional[KvantModel] = None,
    meta_model: Optional[object] = None,
    meta_model_path: Optional[Path] = None,
    meta_history_pred_df: Optional[pd.DataFrame] = None,
    meta_shrinkage_k: float = 10.0,
    meta_train_split: Optional[str] = None,
    meta_min_score_buy: Optional[float] = None,
    meta_min_score_short: Optional[float] = None,
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
    fee        : One-way brokerage fee as a fraction (e.g. 0.0008 = 0.08 %).
                 Used to add net-of-fees columns to the equity curve.
    n_pools    : Number of equal capital pools for position sizing (default 10).
                 Each trade uses 1/n_pools of total capital. Trades arriving
                 when all pools are occupied are skipped.
    execution_priority : How same-timestamp candidate trades compete for free
                 capital pools. ``"first_seen"`` preserves existing order;
                 ``"model_confidence"`` prioritizes higher predicted-side
                 class probability first; ``"meta_score"`` prioritizes
                 the optional meta regressor's output first.
    top_k_per_timestamp : Optional cap on how many candidate trades can be
                 considered per timestamp after ranking. ``None`` disables
                 the cap.
    ticker_cooldown_minutes : Minimum wait time before the same ticker can
                 open another trade. ``0`` disables the cooldown.

    Returns
    -------
    out_dir (as resolved Path)
    """
    exp_dir = Path(exp_dir)
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    execution_priority = str(execution_priority)
    if execution_priority not in _EXECUTION_PRIORITIES:
        raise ValueError(
            f"execution_priority must be one of {sorted(_EXECUTION_PRIORITIES)}, "
            f"got {execution_priority!r}"
        )
    if top_k_per_timestamp is not None and int(top_k_per_timestamp) <= 0:
        raise ValueError("top_k_per_timestamp must be > 0 when provided")
    ticker_cooldown_minutes = int(ticker_cooldown_minutes)
    if ticker_cooldown_minutes < 0:
        raise ValueError("ticker_cooldown_minutes must be >= 0")

    model = model if model is not None else model_cls.load(model_path)
    artifacts = _build_prediction_artifacts(
        exp_dir=exp_dir,
        model_path=model_path,
        model_cls=model_cls,
        split=split,
        tickers=tickers,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        model=model,
    )
    pred_df = artifacts.pred_df
    X = artifacts.X
    y = artifacts.y
    y_pred = artifacts.y_pred
    y_pred_raw = artifacts.y_pred_raw
    tids = artifacts.tids
    metas = artifacts.metas
    ticker_map = artifacts.ticker_map
    n_samples = artifacts.n_samples
    n_tickers = artifacts.n_tickers
    n_thresholded_to_hold = int(np.sum((y_pred_raw != y_pred) & np.isin(y_pred_raw, [0, 2])))
    n_meta_thresholded_to_hold = 0
    n_short_blocked_by_policy = 0

    if meta_model is not None:
        from kvant.meta import add_meta_features

        pred_df = add_meta_features(
            pred_df,
            history_df=meta_history_pred_df,
            fee=fee,
            shrinkage_k=meta_shrinkage_k,
        )
        pred_df["meta_score"] = meta_model.predict(pred_df)
        pred_df["y_pred_pre_meta"] = pred_df["y_pred"].astype(int)
        pred_df["y_pred_pre_meta_name"] = pred_df["y_pred_name"]
        y_pred = _apply_meta_score_thresholds(
            y_pred=pred_df["y_pred"].to_numpy(dtype=np.int64),
            meta_score=pred_df["meta_score"].to_numpy(dtype=float),
            min_score_short=meta_min_score_short,
            min_score_buy=meta_min_score_buy,
        )
        n_meta_thresholded_to_hold = int(
            np.sum(
                (pred_df["y_pred_pre_meta"].to_numpy(dtype=np.int64) != y_pred)
                & np.isin(pred_df["y_pred_pre_meta"].to_numpy(dtype=np.int64), [0, 2])
            )
        )
        pred_df["y_pred"] = y_pred
        pred_df["y_pred_name"] = [
            _LABEL_NAMES[int(v)] if int(v) in _LABEL_IDS else str(v)
            for v in y_pred
        ]

    pred_df["y_pred_pre_policy"] = pred_df["y_pred"].astype(int)
    pred_df["y_pred_pre_policy_name"] = pred_df["y_pred_name"]
    if not bool(allow_short):
        y_pred = _apply_short_execution_policy(y_pred, allow_short=allow_short)
        n_short_blocked_by_policy = int(
            np.sum(pred_df["y_pred_pre_policy"].to_numpy(dtype=np.int64) == 0)
        )
        pred_df["y_pred"] = y_pred
        pred_df["y_pred_name"] = [
            _LABEL_NAMES[int(v)] if int(v) in _LABEL_IDS else str(v)
            for v in y_pred
        ]

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
    _save_equity_curve(
        pred_df,
        out_dir / "equity_curve.csv",
        fee=fee,
        n_pools=n_pools,
        execution_priority=execution_priority,
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
    )

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
    # 11. ensemble_member_comparison.csv (ensemble/council only)
    # ------------------------------------------------------------------
    from kvant.models.ensemble import AveragingEnsembleModel
    if isinstance(model, AveragingEnsembleModel):
        ensemble_member_df = _build_ensemble_member_comparison(
            model=model,
            X=X,
            y=y,
            metas=metas,
            required_buy_probability=required_buy_probability,
            required_sell_probability=required_sell_probability,
            allow_short=allow_short,
        )
        ensemble_member_df.to_csv(out_dir / "ensemble_member_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # 12. run_meta.csv
    # ------------------------------------------------------------------
    experiment_id = exp_dir.name
    run_meta = {
        "experiment_id":   experiment_id,
        "model_name":      model.name,
        "model_class_name": type(model).__name__,
        "model_path":      str(model_path),
        "split":           split,
        "timestamp_run":   datetime.now(tz=timezone.utc).isoformat(),
        "n_samples":       n_samples,
        "n_tickers":       n_tickers,
        "tickers":         ",".join(sorted(ticker_map.values())),
        "required_buy_probability": float(required_buy_probability),
        "required_sell_probability": float(required_sell_probability),
        "allow_short": bool(allow_short),
        "execution_priority": execution_priority,
        "top_k_per_timestamp": "" if top_k_per_timestamp is None else int(top_k_per_timestamp),
        "ticker_cooldown_minutes": ticker_cooldown_minutes,
        "n_thresholded_to_hold": n_thresholded_to_hold,
        "n_meta_thresholded_to_hold": n_meta_thresholded_to_hold,
        "n_short_blocked_by_policy": n_short_blocked_by_policy,
        "meta_enabled": bool(meta_model is not None),
        "meta_model_path": "" if meta_model_path is None else str(meta_model_path),
        "meta_train_split": "" if meta_train_split is None else str(meta_train_split),
        "meta_shrinkage_k": "" if meta_model is None else float(meta_shrinkage_k),
        "meta_min_score_buy": "" if meta_min_score_buy is None else float(meta_min_score_buy),
        "meta_min_score_short": "" if meta_min_score_short is None else float(meta_min_score_short),
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

def _save_equity_curve(
    pred_df: pd.DataFrame,
    out_path: Path,
    fee: float = 0.0,
    n_pools: int = 10,
    execution_priority: str = "model_confidence",
    top_k_per_timestamp: Optional[int] = None,
    ticker_cooldown_minutes: int = 0,
) -> None:
    """
    Build and save a cumulative portfolio PnL equity curve.

    Only BUY (y_pred=2) and SHORT (y_pred=0) predictions that have a
    ``pnl_fraction`` value in the metadata contribute a trade.

    Capital allocation uses *n_pools* equal-sized pools.  Each trade
    occupies one pool from entry (timestamp) until exit (bar_close_time).
    A new trade is skipped if all pools are occupied.  Each trade's
    portfolio impact is ``trade_pnl_pct / n_pools``.

    The output ``timestamp`` is the equity event time: the trade's exit
    timestamp. Entry and exit timestamps are also written explicitly. Both
    gross (theoretical, no pool limit) and portfolio-level columns (with
    pool allocation and optional fees) are included. When several trades
    share the same entry timestamp, ``execution_priority`` decides which
    ones claim scarce pools first. ``top_k_per_timestamp`` and
    ``ticker_cooldown_minutes`` can further reduce execution churn.
    """
    if execution_priority not in _EXECUTION_PRIORITIES:
        raise ValueError(
            f"execution_priority must be one of {sorted(_EXECUTION_PRIORITIES)}, "
            f"got {execution_priority!r}"
        )
    if top_k_per_timestamp is not None and int(top_k_per_timestamp) <= 0:
        raise ValueError("top_k_per_timestamp must be > 0 when provided")
    ticker_cooldown_minutes = int(ticker_cooldown_minutes)
    if ticker_cooldown_minutes < 0:
        raise ValueError("ticker_cooldown_minutes must be >= 0")
    round_trip_fee_pct = 2.0 * float(fee) * 100.0   # percentage points

    _empty_cols = [
        "timestamp", "entry_timestamp", "exit_timestamp", "ticker", "action",
        "trade_pnl_pct", "cumulative_pnl_pct",
        "portfolio_pnl_pct", "cumulative_portfolio_pnl_pct",
        "portfolio_pnl_net_pct", "cumulative_portfolio_pnl_net_pct",
        "pools_busy", "skipped",
    ]

    candidates = _collect_candidate_trades(
        pred_df,
        execution_priority=execution_priority,
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
    )

    if not candidates:
        pd.DataFrame(columns=_empty_cols).to_csv(out_path, index=False)
        return

    # Simulate pool allocation
    # Each pool is "busy" until the trade's bar_close_time.
    pool_free_at: list[pd.Timestamp] = [pd.Timestamp.min] * n_pools
    rows = []

    for event_order, row in enumerate(candidates):
        yp = int(row["y_pred"])
        entry_ts = pd.Timestamp(row["timestamp"])
        exit_ts = pd.Timestamp(row.get("bar_close_time"))
        if pd.isna(entry_ts) or pd.isna(exit_ts):
            continue

        signed_pnl = (-1.0 if yp == 0 else 1.0) * float(row["pnl_fraction"])
        gross_pct = signed_pnl * 100.0

        # Count how many pools are busy at this entry time
        busy = sum(1 for t in pool_free_at if t > entry_ts)

        # Try to claim a pool (pick the one that freed up earliest)
        pool_idx = None
        for i, free_at in enumerate(pool_free_at):
            if free_at <= entry_ts:
                pool_idx = i
                break

        skipped = pool_idx is None
        if not skipped and exit_ts is not None and not pd.isna(exit_ts):
            pool_free_at[pool_idx] = exit_ts

        portfolio_pnl = 0.0 if skipped else gross_pct / n_pools
        portfolio_pnl_net = 0.0 if skipped else (gross_pct - round_trip_fee_pct) / n_pools

        rows.append({
            "timestamp":          exit_ts,
            "entry_timestamp":    entry_ts,
            "exit_timestamp":     exit_ts,
            "ticker":             row["ticker"],
            "action":             _LABEL_NAMES[yp],
            "trade_pnl_pct":      gross_pct,
            "portfolio_pnl_pct":  portfolio_pnl,
            "portfolio_pnl_net_pct": portfolio_pnl_net,
            "pools_busy":         busy,
            "skipped":            skipped,
            "_event_order":       event_order,
        })

    if not rows:
        pd.DataFrame(columns=_empty_cols).to_csv(out_path, index=False)
        return

    eq_df = pd.DataFrame(rows)
    eq_df = eq_df.sort_values(["timestamp", "_event_order"], kind="mergesort").reset_index(drop=True)
    # Gross cumulative (theoretical, no pool limit)
    eq_df["cumulative_pnl_pct"] = eq_df["trade_pnl_pct"].cumsum()
    # Portfolio cumulative (pool-allocated)
    eq_df["cumulative_portfolio_pnl_pct"] = eq_df["portfolio_pnl_pct"].cumsum()
    eq_df["cumulative_portfolio_pnl_net_pct"] = eq_df["portfolio_pnl_net_pct"].cumsum()
    eq_df = eq_df.drop(columns=["_event_order"])
    eq_df.to_csv(out_path, index=False)


def _build_ensemble_member_comparison(
    *,
    model,
    X: np.ndarray,
    y: np.ndarray,
    metas: List[Optional[dict]],
    required_buy_probability: float,
    required_sell_probability: float,
    allow_short: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    members = list(getattr(model, "members", []))
    member_names = list(getattr(model, "member_names", []))

    for idx, member in enumerate(members):
        member_name = (
            str(member_names[idx])
            if idx < len(member_names)
            else str(getattr(member, "name", f"member_{idx}"))
        )
        y_pred_raw = np.asarray(member.predict(X), dtype=np.int64)

        member_proba: Optional[np.ndarray] = None
        try:
            member_proba = member.predict_proba(X)
        except NotImplementedError:
            member_proba = None

        y_pred = _apply_action_probability_thresholds(
            y_pred=y_pred_raw,
            proba=member_proba,
            required_sell_probability=required_sell_probability,
            required_buy_probability=required_buy_probability,
        )
        y_pred = _apply_short_execution_policy(y_pred, allow_short=allow_short)

        metrics = classification_metrics(y, y_pred)
        ret_stats = compute_return_stats(y_pred=y_pred, metas=metas)
        directional = _overall_directional_summary(
            pd.DataFrame({"y_true": y, "y_pred": y_pred})
        )
        rows.append(
            {
                "member_name": member_name,
                "member_class_name": type(member).__name__,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "directional_accuracy": float(directional.get("directional_accuracy", 0.0)),
                "directional_opposite_rate": float(directional.get("directional_opposite_rate", 0.0)),
                "accuracy_call_put/avg": float(ret_stats.get("accuracy_call_put/avg", 0.0)),
                "bruto_profit_pct/avg": float(ret_stats.get("bruto_profit_pct/avg", 0.0)),
                "directional_true_n": float(directional.get("directional_true_n", 0.0)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "rank",
                "member_name",
                "member_class_name",
                "accuracy",
                "directional_accuracy",
                "directional_opposite_rate",
                "accuracy_call_put/avg",
                "bruto_profit_pct/avg",
                "directional_true_n",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        [
            "bruto_profit_pct/avg",
            "directional_accuracy",
            "accuracy",
            "accuracy_call_put/avg",
        ],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _collect_candidate_trades(
    pred_df: pd.DataFrame,
    *,
    execution_priority: str,
    top_k_per_timestamp: Optional[int],
    ticker_cooldown_minutes: int,
) -> list[pd.Series]:
    """Return eligible trades ordered for execution at each timestamp."""
    pnl_frac = pd.to_numeric(pred_df.get("pnl_fraction"), errors="coerce")
    candidate_mask = pred_df["y_pred"].isin([0, 2]) & pnl_frac.notna()
    if not candidate_mask.any():
        return []

    candidates = pred_df.loc[candidate_mask].copy()
    candidates["_candidate_order"] = np.arange(len(candidates), dtype=np.int64)
    candidates = candidates.sort_values(["timestamp", "_candidate_order"], kind="mergesort")

    if execution_priority != "first_seen":
        score = _execution_priority_score(candidates, execution_priority=execution_priority)
        if score.notna().all():
            candidates["_execution_score"] = score
            candidates = candidates.sort_values(
                ["timestamp", "_execution_score", "_candidate_order"],
                ascending=[True, False, True],
                kind="mergesort",
            )

    if top_k_per_timestamp is not None:
        candidates = candidates.groupby("timestamp", group_keys=False).head(int(top_k_per_timestamp))

    if ticker_cooldown_minutes > 0:
        candidates = _apply_ticker_cooldown(candidates, ticker_cooldown_minutes=ticker_cooldown_minutes)

    return [row for _, row in candidates.iterrows()]


def _execution_priority_score(
    candidate_df: pd.DataFrame,
    *,
    execution_priority: str,
) -> pd.Series:
    """Return the score used to prioritize simultaneous trade candidates."""
    if execution_priority == "meta_score":
        if "meta_score" not in candidate_df.columns:
            return pd.Series(np.nan, index=candidate_df.index, dtype=float)
        return pd.to_numeric(candidate_df["meta_score"], errors="coerce")

    score = pd.Series(np.nan, index=candidate_df.index, dtype=float)

    if "prob_SHORT" in candidate_df.columns:
        short_mask = candidate_df["y_pred"].astype(int) == 0
        score.loc[short_mask] = pd.to_numeric(
            candidate_df.loc[short_mask, "prob_SHORT"],
            errors="coerce",
        )

    if "prob_BUY" in candidate_df.columns:
        buy_mask = candidate_df["y_pred"].astype(int) == 2
        score.loc[buy_mask] = pd.to_numeric(
            candidate_df.loc[buy_mask, "prob_BUY"],
            errors="coerce",
        )

    return score


def _apply_ticker_cooldown(
    candidate_df: pd.DataFrame,
    *,
    ticker_cooldown_minutes: int,
) -> pd.DataFrame:
    """Drop trades that arrive too soon after a prior entry in the same ticker."""
    if candidate_df.empty or ticker_cooldown_minutes <= 0:
        return candidate_df

    cooldown = pd.Timedelta(minutes=int(ticker_cooldown_minutes))
    keep_rows: list[pd.Series] = []
    last_entry_by_ticker: dict[str, pd.Timestamp] = {}

    for _, row in candidate_df.iterrows():
        ticker = str(row["ticker"])
        entry_ts = row["timestamp"]
        last_entry = last_entry_by_ticker.get(ticker)
        if last_entry is not None and entry_ts < last_entry + cooldown:
            continue
        last_entry_by_ticker[ticker] = entry_ts
        keep_rows.append(row)

    if not keep_rows:
        return candidate_df.iloc[0:0].copy()

    return pd.DataFrame(keep_rows).reset_index(drop=True)


def _apply_action_probability_thresholds(
    y_pred: np.ndarray,
    proba: Optional[np.ndarray],
    required_sell_probability: float,
    required_buy_probability: float,
) -> np.ndarray:
    """Demote low-confidence SHORT/BUY predictions to HOLD using side-specific thresholds."""
    out = np.asarray(y_pred, dtype=np.int64).copy()
    if proba is None or not isinstance(proba, np.ndarray) or proba.ndim != 2 or proba.shape[1] < 3:
        return out

    sell_thr = float(required_sell_probability)
    buy_thr = float(required_buy_probability)

    if sell_thr > 0:
        short_mask = out == 0
        out[short_mask & (proba[:, 0] < sell_thr)] = 1

    if buy_thr > 0:
        buy_mask = out == 2
        out[buy_mask & (proba[:, 2] < buy_thr)] = 1

    return out


def _apply_meta_score_thresholds(
    y_pred: np.ndarray,
    meta_score: np.ndarray,
    *,
    min_score_short: Optional[float],
    min_score_buy: Optional[float],
) -> np.ndarray:
    """Demote directional predictions below side-specific meta-score thresholds to HOLD."""
    out = np.asarray(y_pred, dtype=np.int64).copy()
    score = np.asarray(meta_score, dtype=float)
    if score.ndim != 1 or score.shape[0] != out.shape[0]:
        raise ValueError("meta_score must be a 1D array aligned with y_pred")

    if min_score_short is not None:
        short_mask = out == 0
        out[short_mask & (score < float(min_score_short))] = 1

    if min_score_buy is not None:
        buy_mask = out == 2
        out[buy_mask & (score < float(min_score_buy))] = 1

    return out


def _apply_short_execution_policy(y_pred: np.ndarray, *, allow_short: bool) -> np.ndarray:
    """Demote SHORT predictions to HOLD when short execution is disabled."""
    out = np.asarray(y_pred, dtype=np.int64).copy()
    if not bool(allow_short):
        out[out == 0] = 1
    return out


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
