from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {"accuracy": 0.0}
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def per_ticker_trade_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Computes per-ticker trade stats based on metadata.

    A "trade" is only counted when prediction is call/put (0 or 2) AND metadata has pnl_fraction.

    Returns:
      { tid: {
          "n_trades": int,
          "bruto_profit_pct/avg": float,          # avg profit per trade in percent
          "accuracy_call_put/avg": float,         # accuracy restricted to call/put trades with metadata
        }, ... }
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

        pnl_frac = m.get("pnl_fraction", None)
        true_label = m.get("label", None)
        if not isinstance(pnl_frac, (int, float)):
            continue

        tid = int(tids[i])
        signed = (-1.0 if yp == 0 else 1.0) * float(pnl_frac)
        by_tid[tid]["pct_change"].append(signed)
        by_tid[tid]["acc"].append(true_label == yp)

    out: Dict[int, Dict[str, Any]] = {}
    for tid, d in by_tid.items():
        pct_change = d["pct_change"]
        acc = d["acc"]
        out[tid] = {
            "n_trades": int(len(pct_change)),
            "bruto_profit_pct/avg": float(np.mean(pct_change) * 100.0) if pct_change else 0.0,
            "accuracy_call_put/avg": float(np.mean(acc)) if acc else 0.0,
        }
    return out


def compute_return_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Overall (split-level) metadata-based statistics.
    """
    assert len(y_pred) == len(metas)
    if tids is not None:
        assert len(tids) == len(y_pred)

    out: Dict[str, Any] = {}
    n_total = len(metas)
    n_with_meta = sum(m is not None for m in metas)

    out["n"] = int(n_total)
    out["n_with_metadata"] = int(n_with_meta)

    pct_change = []
    acc_call_put = []

    for i, yp in enumerate(y_pred):
        m = metas[i]
        if m is None:
            continue

        yp = int(yp)
        if yp in (0, 2):  # call / put
            pnl_frac = m.get("pnl_fraction", None)
            true_label = m.get("label", None)
            if not isinstance(pnl_frac, (int, float)):
                continue

            pct_change.append((-1 if yp == 0 else 1) * float(pnl_frac))
            acc_call_put.append(true_label == yp)

    out["accuracy_call_put/avg"] = float(np.mean(acc_call_put)) if acc_call_put else 0.0
    out["bruto_profit_pct/avg"] = float(np.mean(pct_change) * 100.0) if pct_change else 0.0
    return out



def compute_action_profit_stats(
    *,
    y_pred: np.ndarray,
    metas: List[Optional[dict]],
    tids: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-ticker profit stats split by action:

      Action BUY:   y_pred == 2  -> signed profit = +pnl_fraction
      Action SHORT: y_pred == 0  -> signed profit = -pnl_fraction

    All values are in percent (%), derived from pnl_fraction * 100.

    Returns:
      { tid: {
          "buy/n_trades": int,
          "buy/profit_pct/avg_per_trade": float (nan if no trades),
          "buy/profit_pct/total": float (0 if no trades),

          "short/n_trades": int,
          "short/profit_pct/avg_per_trade": float (nan if no trades),
          "short/profit_pct/total": float (0 if no trades),
      }}
    """
    assert len(y_pred) == len(metas) == len(tids)

    # accumulate signed pnl fractions
    buy_pnls = defaultdict(list)    # tid -> list[pnl_frac]
    short_pnls = defaultdict(list)  # tid -> list[-pnl_frac]

    for i in range(len(y_pred)):
        m = metas[i]
        if m is None:
            continue

        pnl_frac = m.get("pnl_fraction", None)
        if not isinstance(pnl_frac, (int, float)):
            continue

        tid = int(tids[i])
        yp = int(y_pred[i])

        if yp == 2:
            buy_pnls[tid].append(float(pnl_frac))
        elif yp == 0:
            short_pnls[tid].append(-float(pnl_frac))

    out: Dict[int, Dict[str, Any]] = {}
    all_tids = set(buy_pnls.keys()) | set(short_pnls.keys())

    for tid in all_tids:
        b = buy_pnls.get(tid, [])
        s = short_pnls.get(tid, [])

        out[tid] = {
            "buy/n_trades": int(len(b)),
            "buy/profit_pct/avg_per_trade": float(np.mean(b) * 100.0) if len(b) else float("nan"),
            "buy/profit_pct/total": float(np.sum(b) * 100.0) if len(b) else 0.0,

            "short/n_trades": int(len(s)),
            "short/profit_pct/avg_per_trade": float(np.mean(s) * 100.0) if len(s) else float("nan"),
            "short/profit_pct/total": float(np.sum(s) * 100.0) if len(s) else 0.0,
        }

    return out