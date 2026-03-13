"""
training.metrics — Evaluation metrics used across all models.

All functions accept plain numpy arrays and return plain Python scalars
or dicts, keeping the interface framework-agnostic.

classification_metrics(y_true, y_pred)              → Dict[str, float]
per_ticker_trade_stats(y_pred, metas, tids)         → Dict[int, Dict]
compute_return_stats(y_pred, metas, tids=None)      → Dict[str, Any]
compute_action_profit_stats(y_pred, metas, tids)    → Dict[int, Dict]

Label convention (default)
--------------------------
  0 = SHORT / sell
  1 = HOLD  / abstain
  2 = BUY   / long
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Return accuracy and (optionally) macro-averaged precision/recall/F1."""
    if len(y_true) == 0:
        return {"accuracy": 0.0}
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


# ---------------------------------------------------------------------------
# Trade / profit stats
# ---------------------------------------------------------------------------

def per_ticker_trade_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Per-ticker stats for call/put predictions that have label metadata.

    A trade is counted only when ``y_pred`` is 0 or 2 AND the metadata
    entry contains a ``pnl_fraction`` field.

    Returns
    -------
    { tid: { "n_trades", "bruto_profit_pct/avg", "accuracy_call_put/avg" } }
    """
    assert len(y_pred) == len(metas) == len(tids)
    by_tid: Dict[int, Dict[str, list]] = defaultdict(lambda: {"pct_change": [], "acc": []})

    for i in range(len(y_pred)):
        m = metas[i]
        if m is None:
            continue
        yp = int(y_pred[i])
        if yp not in (0, 2):
            continue
        pnl_frac = m.get("pnl_fraction")
        if not isinstance(pnl_frac, (int, float)):
            continue
        tid = int(tids[i])
        signed = (-1.0 if yp == 0 else 1.0) * float(pnl_frac)
        by_tid[tid]["pct_change"].append(signed)
        by_tid[tid]["acc"].append(m.get("label") == yp)

    out: Dict[int, Dict[str, Any]] = {}
    for tid, d in by_tid.items():
        pc, ac = d["pct_change"], d["acc"]
        out[tid] = {
            "n_trades":               int(len(pc)),
            "bruto_profit_pct/avg":   float(np.mean(pc) * 100.0) if pc else 0.0,
            "accuracy_call_put/avg":  float(np.mean(ac)) if ac else 0.0,
        }
    return out


def compute_return_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Overall (split-level) metadata-based return statistics."""
    assert len(y_pred) == len(metas)
    pct_change: list = []
    acc_call_put: list = []

    for i, yp in enumerate(y_pred):
        m = metas[i]
        if m is None:
            continue
        yp = int(yp)
        if yp in (0, 2):
            pnl_frac = m.get("pnl_fraction")
            if not isinstance(pnl_frac, (int, float)):
                continue
            pct_change.append((-1 if yp == 0 else 1) * float(pnl_frac))
            acc_call_put.append(m.get("label") == yp)

    return {
        "n":                     int(len(metas)),
        "n_with_metadata":       int(sum(m is not None for m in metas)),
        "accuracy_call_put/avg": float(np.mean(acc_call_put)) if acc_call_put else 0.0,
        "bruto_profit_pct/avg":  float(np.mean(pct_change) * 100.0) if pct_change else 0.0,
    }


def compute_action_profit_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Per-ticker profit split by action:
      BUY   (y_pred == 2) → signed profit = +pnl_fraction
      SHORT (y_pred == 0) → signed profit = -pnl_fraction

    Returns
    -------
    { tid: {
        "buy/n_trades", "buy/profit_pct/avg_per_trade", "buy/profit_pct/total",
        "short/n_trades", "short/profit_pct/avg_per_trade", "short/profit_pct/total",
    }}
    """
    assert len(y_pred) == len(metas) == len(tids)
    buy_pnls: Dict[int, list]   = defaultdict(list)
    short_pnls: Dict[int, list] = defaultdict(list)

    for i in range(len(y_pred)):
        m = metas[i]
        if m is None:
            continue
        pnl_frac = m.get("pnl_fraction")
        if not isinstance(pnl_frac, (int, float)):
            continue
        tid, yp = int(tids[i]), int(y_pred[i])
        if yp == 2:
            buy_pnls[tid].append(float(pnl_frac))
        elif yp == 0:
            short_pnls[tid].append(-float(pnl_frac))

    out: Dict[int, Dict[str, Any]] = {}
    for tid in set(buy_pnls) | set(short_pnls):
        b = buy_pnls.get(tid, [])
        s = short_pnls.get(tid, [])
        out[tid] = {
            "buy/n_trades":                int(len(b)),
            "buy/profit_pct/avg_per_trade": float(np.mean(b) * 100.0) if b else float("nan"),
            "buy/profit_pct/total":         float(np.sum(b) * 100.0)  if b else 0.0,
            "short/n_trades":               int(len(s)),
            "short/profit_pct/avg_per_trade": float(np.mean(s) * 100.0) if s else float("nan"),
            "short/profit_pct/total":         float(np.sum(s) * 100.0)  if s else 0.0,
        }
    return out
