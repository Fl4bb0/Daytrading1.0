from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from kvant.ml_prepare_data.data_loading import PreparedStore
from .predict import predict
from .metrics import classification_metrics, compute_action_profit_stats


@dataclass(frozen=True)
class EvalConfig:
    compute_per_ticker_accuracy: bool = True
    compute_profit_stats: bool = True
    labels: tuple[int, ...] = (0, 1, 2)


class ExperimentEvaluator:
    def __init__(
        self,
        *,
        store: PreparedStore,
        device: torch.device,
        cfg: EvalConfig = EvalConfig(),
        logger: Optional[Any] = None,
    ):
        self.store = store
        self.device = device
        self.cfg = cfg
        self.logger = logger

    def evaluate_split(
        self,
        split: str,
        model: torch.nn.Module,
        loader: DataLoader,
        *,
        step: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
        pred_out = predict(model, loader, self.device)
        y_true = pred_out["y_true"]
        y_pred = pred_out["y_pred"]
        tid = pred_out["tid"].astype(np.int64, copy=False)
        tpos = pred_out["tpos"].astype(np.int64, copy=False)

        metrics: Dict[str, Any] = {}
        per_ticker_rows: List[Dict[str, Any]] = []

        # split-level scalars
        cls = classification_metrics(y_true, y_pred)
        for k, v in cls.items():
            metrics[f"{split}/{k}"] = v

        # confusion counts for heatmap
        cm = confusion_matrix(y_true, y_pred, labels=list(self.cfg.labels)).astype(np.int64, copy=False)

        # profit stats need metadata
        per_tid_profit: Dict[int, Dict[str, Any]] = {}
        if self.cfg.compute_profit_stats:
            index = np.stack([tid, tpos], axis=1).astype(np.int32, copy=False)
            metas = self.store.metadata_for_index(index)
            per_tid_profit = compute_action_profit_stats(y_pred=y_pred, metas=metas, tids=tid)

        # per-ticker accuracy (+ profit stats columns)
        if self.cfg.compute_per_ticker_accuracy:
            for t in np.unique(tid):
                mask = (tid == t)
                n_t = int(mask.sum())
                acc_t = float((y_true[mask] == y_pred[mask]).mean()) if n_t > 0 else 0.0

                ticker = self.store.tickers_all[int(t)]
                p = per_tid_profit.get(int(t), {})

                per_ticker_rows.append(
                    {
                        "epoch": int(step) if step is not None else None,
                        "split": split,
                        "tid": int(t),
                        "ticker": str(ticker),
                        "acc": acc_t,
                        "n": n_t,

                        # buy-only
                        "buy_n_trades": int(p.get("buy/n_trades", 0)),
                        "buy_profit_avg_per_trade_pct": float(p.get("buy/profit_pct/avg_per_trade", float("nan"))),
                        "buy_profit_total_pct": float(p.get("buy/profit_pct/total", 0.0)),

                        # short-only
                        "short_n_trades": int(p.get("short/n_trades", 0)),
                        "short_profit_avg_per_trade_pct": float(p.get("short/profit_pct/avg_per_trade", float("nan"))),
                        "short_profit_total_pct": float(p.get("short/profit_pct/total", 0.0)),
                    }
                )

        return metrics, per_ticker_rows, cm

    def evaluate_all(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, Optional[DataLoader]],
        *,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        all_metrics: Dict[str, Any] = {}
        all_rows: List[Dict[str, Any]] = {}
        confusion_counts: Dict[str, np.ndarray] = {}

        rows_out: List[Dict[str, Any]] = []

        for split, loader in loaders.items():
            if loader is None or len(loader.dataset) == 0:
                continue

            m, rows, cm = self.evaluate_split(split, model, loader, step=step)
            all_metrics.update(m)
            rows_out.extend(rows)
            confusion_counts[split] = cm

        # special payloads for logger
        all_metrics["_per_ticker_rows"] = rows_out
        all_metrics["_confusion_counts"] = confusion_counts

        if self.logger is not None:
            self.logger.log(all_metrics, step=step)

        return all_metrics