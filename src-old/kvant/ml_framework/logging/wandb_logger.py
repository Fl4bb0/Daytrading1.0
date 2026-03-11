from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, List
from collections import defaultdict

import numpy as np
import wandb
import matplotlib.pyplot as plt


LABEL_MEANINGS = {
    0: "triple-bar up > 2.5%",
    1: "triple-bar flat",
    2: "triple-bar down > 2.5%",
}

CLASS_NAMES = [
    "Up > 2.5% (y=0)",
    "Flat (y=1)",
    "Down > 2.5% (y=2)",
]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _plot_confusion_heatmap(cm: np.ndarray, title: str) -> plt.Figure:
    """
    cm: (C,C) int array. Produces a heatmap with:
      - cell text: "count\n(percent of row)"
    """
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig = plt.figure(figsize=(5.5, 4.8), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = 100.0 * float(cm_norm[i, j])
            ax.text(j, i, f"{count}\n{pct:.1f}%", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    return fig


class WandbLogger:
    def __init__(
        self,
        *,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **init_kwargs,
    ):
        self.run = wandb.init(project=project, name=name, config=config or {}, **init_kwargs)

        # ticker_label -> list of epoch dicts
        self._ticker_history: Dict[str, List[Dict[str, Any]]] = {}
        self._tickers_to_chart: List[str] = []

    def log_config(self, cfg: Any) -> None:
        if is_dataclass(cfg):
            wandb.config.update(asdict(cfg), allow_val_change=True)
        elif isinstance(cfg, dict):
            wandb.config.update(cfg, allow_val_change=True)

    def setup(self, *, exp: Any, loaders: Dict[str, Any]) -> None:
        # label meanings
        self.run.config.update({"label_meanings": LABEL_MEANINGS}, allow_val_change=True)
        label_table = wandb.Table(columns=["y", "meaning"])
        for y, meaning in LABEL_MEANINGS.items():
            label_table.add_data(int(y), str(meaning))
        wandb.log({"data/label_meanings": label_table})

        # tickers mapping
        tickers = getattr(exp.store, "tickers_all", None) or []
        ticker_table = wandb.Table(columns=["tid", "ticker", "ticker_label"])
        for tid, tkr in enumerate(tickers):
            ticker_table.add_data(int(tid), str(tkr), f"{tkr} (tid={tid})")
        wandb.log({"data/tickers": ticker_table})

        # first 10 tickers (or fewer)
        n_chart = min(10, len(tickers))
        self._tickers_to_chart = [f"{tickers[tid]} (tid={tid})" for tid in range(n_chart)]
        self.run.config.update({"wandb_chart_first_n_tickers": n_chart}, allow_val_change=True)

        # split distribution table (static)
        # (Assumes ds.summary exists; if you truly want to use ds.display instead, tell me what it returns.)
        dist_table = wandb.Table(
            columns=["split", "n", "first_ts", "last_ts", "y_count_0", "y_count_1", "y_count_2"]
        )
        for split, loader in loaders.items():
            if loader is None:
                continue
            ds = loader.dataset
            s = ds.summary(display=False)
            overall = s.get("overall", {}) or {}
            yc = overall.get("y_counts", {}) or {}
            dist_table.add_data(
                str(split),
                _safe_int(overall.get("n", 0)),
                overall.get("first_ts", None),
                overall.get("last_ts", None),
                _safe_int(yc.get(0, 0)),
                _safe_int(yc.get(1, 0)),
                _safe_int(yc.get(2, 0)),
            )
        wandb.log({"data/split_distribution": dist_table})

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        per_ticker_rows = metrics.pop("_per_ticker_rows", None)
        confusion_counts = metrics.pop("_confusion_counts", None)

        # log scalars
        wandb.log(metrics, step=step)

        if step is None:
            return

        # confusion matrices (graphical heatmaps)
        if isinstance(confusion_counts, dict):
            for split, cm in confusion_counts.items():
                if cm is None:
                    continue
                fig = _plot_confusion_heatmap(np.asarray(cm), title=f"Confusion matrix ({split})")
                wandb.log({f"charts/confusion_matrix/{split}": wandb.Image(fig)}, step=step)
                plt.close(fig)

        if not per_ticker_rows:
            return

        # 1) long-form per-ticker table
        table = wandb.Table(
            columns=[
                "epoch", "split", "tid", "ticker", "ticker_label",
                "acc", "n",
                "buy_n_trades", "buy_profit_avg_per_trade_pct", "buy_profit_total_pct",
                "short_n_trades", "short_profit_avg_per_trade_pct", "short_profit_total_pct",
            ]
        )

        # per-epoch values for charting: ticker_label -> split -> dict(values)
        by_ticker = defaultdict(lambda: defaultdict(dict))

        for r in per_ticker_rows:
            epoch = int(r.get("epoch", step) or step)
            split = str(r["split"])
            tid = int(r["tid"])
            ticker = str(r["ticker"])
            ticker_label = f"{ticker} (tid={tid})"

            acc = _to_float_or_nan(r.get("acc", np.nan))
            n = int(r.get("n", 0))

            buy_n_trades = int(r.get("buy_n_trades", 0))
            buy_avg = _to_float_or_nan(r.get("buy_profit_avg_per_trade_pct", np.nan))
            buy_total = _to_float_or_nan(r.get("buy_profit_total_pct", 0.0))

            short_n_trades = int(r.get("short_n_trades", 0))
            short_avg = _to_float_or_nan(r.get("short_profit_avg_per_trade_pct", np.nan))
            short_total = _to_float_or_nan(r.get("short_profit_total_pct", 0.0))

            table.add_data(
                epoch, split, tid, ticker, ticker_label,
                acc, n,
                buy_n_trades, buy_avg, buy_total,
                short_n_trades, short_avg, short_total,
            )

            by_ticker[ticker_label][split] = {
                "acc": acc,
                "buy_avg": buy_avg,
                "buy_total": buy_total,
                "short_avg": short_avg,
                "short_total": short_total,
            }

        wandb.log({"perf/per_ticker_table": table}, step=step)

        # 2) charts for first 10 tickers (train/val/test lines)
        for ticker_label in self._tickers_to_chart:
            d = by_ticker.get(ticker_label, {})
            if not d:
                continue

            hist = self._ticker_history.setdefault(ticker_label, [])
            hist.append(
                {
                    "epoch": int(step),
                    "acc_train": d.get("train", {}).get("acc", np.nan),
                    "acc_val": d.get("val", {}).get("acc", np.nan),
                    "acc_test": d.get("test", {}).get("acc", np.nan),

                    "buy_avg_train": d.get("train", {}).get("buy_avg", np.nan),
                    "buy_avg_val": d.get("val", {}).get("buy_avg", np.nan),
                    "buy_avg_test": d.get("test", {}).get("buy_avg", np.nan),

                    "short_avg_train": d.get("train", {}).get("short_avg", np.nan),
                    "short_avg_val": d.get("val", {}).get("short_avg", np.nan),
                    "short_avg_test": d.get("test", {}).get("short_avg", np.nan),

                    "buy_total_train": d.get("train", {}).get("buy_total", np.nan),
                    "buy_total_val": d.get("val", {}).get("buy_total", np.nan),
                    "buy_total_test": d.get("test", {}).get("buy_total", np.nan),

                    "short_total_train": d.get("train", {}).get("short_total", np.nan),
                    "short_total_val": d.get("val", {}).get("short_total", np.nan),
                    "short_total_test": d.get("test", {}).get("short_total", np.nan),
                }
            )

            xs = [h["epoch"] for h in hist]

            def line_series(key_train: str, key_val: str, key_test: str, title: str, chart_key: str):
                chart = wandb.plot.line_series(
                    xs=xs,
                    ys=[
                        [h[key_train] for h in hist],
                        [h[key_val] for h in hist],
                        [h[key_test] for h in hist],
                    ],
                    keys=["train", "val", "test"],
                    title=title,
                    xname="epoch",
                    split_table=True,
                )
                wandb.log({chart_key: chart}, step=step)

            # Accuracy
            line_series(
                "acc_train", "acc_val", "acc_test",
                f"Per-ticker accuracy: {ticker_label}",
                f"charts/per_ticker/acc/{ticker_label}",
            )

            # Per-trade profit (buy-only)
            line_series(
                "buy_avg_train", "buy_avg_val", "buy_avg_test",
                f"Buy-only profit (% avg per trade): {ticker_label}",
                f"charts/per_ticker/buy_profit_avg_per_trade_pct/{ticker_label}",
            )

            # Per-trade profit (short-only)
            line_series(
                "short_avg_train", "short_avg_val", "short_avg_test",
                f"Short-only profit (% avg per trade): {ticker_label}",
                f"charts/per_ticker/short_profit_avg_per_trade_pct/{ticker_label}",
            )

            # Total profit (buy-only)
            line_series(
                "buy_total_train", "buy_total_val", "buy_total_test",
                f"Buy-only profit (% total): {ticker_label}",
                f"charts/per_ticker/buy_profit_total_pct/{ticker_label}",
            )

            # Total profit (short-only)
            line_series(
                "short_total_train", "short_total_val", "short_total_test",
                f"Short-only profit (% total): {ticker_label}",
                f"charts/per_ticker/short_profit_total_pct/{ticker_label}",
            )

    def stop(self) -> None:
        self.run.finish()