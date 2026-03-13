"""
scripts/run_plot.py — Generate figures from evaluation CSVs produced by run_predict.py.

Figures saved to <eval-dir>/figures/
--------------------------------------
01_confusion_matrix.png        — heatmap (counts + normalised %)
02_classification_metrics.png  — precision / recall / F1 bar chart per class
03_label_distribution.png      — y_true vs y_pred counts per ticker and class
04_prediction_distribution.png — overall predicted class breakdown (pie + bar)
05_per_ticker_accuracy.png     — per-ticker accuracy bar chart
06_equity_curve.png            — cumulative PnL over time  (skipped if no trades)
07_trade_stats.png             — per-ticker buy/short trade count + avg profit
08_prob_distributions.png      — predicted class probability histograms (if available)

Usage
-----
  python scripts/run_plot.py --eval-dir prepared/<exp>/eval/<model>_<split>
  python scripts/run_plot.py --eval-dir prepared/<exp>/eval/<model>_test --show
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _load_csvs(eval_dir: Path) -> dict:
    import pandas as pd
    required = [
        "classification_metrics", "confusion_matrix", "label_distribution",
        "predictions", "trade_stats", "return_stats", "equity_curve", "run_meta",
    ]
    dfs = {}
    for name in required:
        path = eval_dir / f"{name}.csv"
        if path.exists():
            dfs[name] = pd.read_csv(path)
        else:
            print(f"  [warn] Missing {name}.csv — skipping related plots.")
    return dfs


def _save(fig, path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 01 — Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "confusion_matrix" not in dfs:
        return

    raw = dfs["confusion_matrix"]
    raw = raw.set_index(raw.columns[0])
    cm  = raw.values.astype(float)
    labels = list(raw.columns)

    total = cm.sum()
    cm_norm = cm / total if total > 0 else cm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Counts",         ".0f"),
        (axes[1], cm_norm, "Normalised (%)", ".1%"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                txt = f"{val:{fmt}}"
                colour = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=colour, fontsize=11)

    _save(fig, out_dir / "01_confusion_matrix.png", show)


# ---------------------------------------------------------------------------
# 02 — Classification metrics bar chart
# ---------------------------------------------------------------------------
def plot_classification_metrics(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "classification_metrics" not in dfs:
        return

    df = dfs["classification_metrics"]
    # Keep only per-class rows (SHORT, HOLD, BUY)
    class_rows = df[df["class"].isin(["SHORT", "HOLD", "BUY"])].copy()
    if class_rows.empty:
        return

    metrics = ["precision", "recall", "f1-score"]
    metrics = [m for m in metrics if m in class_rows.columns]
    classes = class_rows["class"].tolist()

    x     = np.arange(len(classes))
    width = 0.25
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        vals = class_rows[metric].fillna(0).tolist()
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=colour)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Overall accuracy line
    overall_row = df[df["class"] == "overall"]
    if not overall_row.empty and "accuracy" in overall_row.columns:
        acc = float(overall_row["accuracy"].iloc[0])
        ax.axhline(acc, linestyle="--", color="red", linewidth=1.2, label=f"Overall acc {acc:.2f}")

    ax.set_xticks(x + width); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score"); ax.set_title("Per-Class Classification Metrics", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir / "02_classification_metrics.png", show)


# ---------------------------------------------------------------------------
# 03 — Label distribution per ticker
# ---------------------------------------------------------------------------
def plot_label_distribution(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "label_distribution" not in dfs:
        return

    df      = dfs["label_distribution"]
    tickers = df["ticker"].unique()
    labels  = ["SHORT", "HOLD", "BUY"]
    colours_true = ["#d62728", "#7f7f7f", "#2ca02c"]
    colours_pred = ["#ff7f7f", "#c7c7c7", "#98df8a"]

    n_tickers = len(tickers)
    fig, axes = plt.subplots(1, n_tickers, figsize=(4 * n_tickers, 5), sharey=True)
    if n_tickers == 1:
        axes = [axes]

    fig.suptitle("Label Distribution per Ticker (True vs Predicted)", fontweight="bold")

    for ax, ticker in zip(axes, tickers):
        sub = df[df["ticker"] == ticker].set_index("label").reindex(labels).fillna(0)
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, sub["y_true_count"],  w, label="y_true",  color=colours_true)
        ax.bar(x + w/2, sub["y_pred_count"], w, label="y_pred", color=colours_pred)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(ticker); ax.set_xlabel("Class")
        if ax is axes[0]:
            ax.set_ylabel("Count")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir / "03_label_distribution.png", show)


# ---------------------------------------------------------------------------
# 04 — Overall prediction distribution
# ---------------------------------------------------------------------------
def plot_prediction_distribution(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "predictions" not in dfs:
        return

    df     = dfs["predictions"]
    labels = ["SHORT", "HOLD", "BUY"]
    ids    = [0, 1, 2]
    colours = ["#d62728", "#7f7f7f", "#2ca02c"]

    true_counts = [int((df["y_true"] == i).sum()) for i in ids]
    pred_counts = [int((df["y_pred"] == i).sum()) for i in ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Overall Prediction Distribution", fontweight="bold")

    # Bar chart
    x = np.arange(len(labels)); w = 0.35
    axes[0].bar(x - w/2, true_counts, w, label="y_true",  color=colours, alpha=0.9)
    axes[0].bar(x + w/2, pred_counts, w, label="y_pred", color=colours, alpha=0.5, edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Count"); axes[0].set_title("True vs Predicted Counts")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # Pie chart of predictions
    non_zero = [(c, l, col) for c, l, col in zip(pred_counts, labels, colours) if c > 0]
    if non_zero:
        counts, lbls, cols = zip(*non_zero)
        axes[1].pie(counts, labels=lbls, colors=cols, autopct="%1.1f%%", startangle=140)
        axes[1].set_title("Predicted Class Breakdown")
    else:
        axes[1].text(0.5, 0.5, "No predictions", ha="center", va="center")

    _save(fig, out_dir / "04_prediction_distribution.png", show)


# ---------------------------------------------------------------------------
# 05 — Per-ticker accuracy
# ---------------------------------------------------------------------------
def plot_per_ticker_accuracy(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    if "predictions" not in dfs:
        return

    df = dfs["predictions"]
    rows = []
    for ticker, grp in df.groupby("ticker"):
        acc = (grp["y_true"] == grp["y_pred"]).mean()
        rows.append({"ticker": ticker, "accuracy": acc, "n": len(grp)})

    import pandas as pd
    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, len(summary) * 1.5), 5))
    colours = ["#2ca02c" if v >= 0.5 else "#d62728" for v in summary["accuracy"]]
    bars = ax.bar(summary["ticker"], summary["accuracy"], color=colours)
    ax.axhline(1/3, linestyle="--", color="gray",   linewidth=1, label="Random baseline (33%)")
    ax.axhline(0.5, linestyle="--", color="orange", linewidth=1, label="50% line")

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{row['accuracy']:.2f}\n(n={row['n']})", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy"); ax.set_title("Per-Ticker Accuracy", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir / "05_per_ticker_accuracy.png", show)


# ---------------------------------------------------------------------------
# 06 — Equity curve
# ---------------------------------------------------------------------------
def plot_equity_curve(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    if "equity_curve" not in dfs:
        return

    df = dfs["equity_curve"]
    if df.empty or "cumulative_pnl_pct" not in df.columns or df["cumulative_pnl_pct"].isna().all():
        print("  [skip] equity_curve.csv has no trades — skipping plot.")
        return

    df = df.dropna(subset=["cumulative_pnl_pct"])
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(df)), df["cumulative_pnl_pct"], linewidth=1.5, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(df)), df["cumulative_pnl_pct"], 0,
                    where=df["cumulative_pnl_pct"] >= 0, alpha=0.2, color="green")
    ax.fill_between(range(len(df)), df["cumulative_pnl_pct"], 0,
                    where=df["cumulative_pnl_pct"] < 0, alpha=0.2, color="red")

    final = float(df["cumulative_pnl_pct"].iloc[-1])
    ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL (%)")
    ax.set_title(f"Equity Curve  (final: {final:+.2f}%)", fontweight="bold")
    ax.grid(alpha=0.3)
    _save(fig, out_dir / "06_equity_curve.png", show)


# ---------------------------------------------------------------------------
# 07 — Trade stats per ticker
# ---------------------------------------------------------------------------
def plot_trade_stats(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "trade_stats" not in dfs:
        return

    df = dfs["trade_stats"]
    if df.empty or df["n_trades"].sum() == 0:
        print("  [skip] No trades in trade_stats.csv — skipping plot.")
        return

    tickers = df["ticker"].tolist()
    x = np.arange(len(tickers))
    w = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-Ticker Trade Statistics", fontweight="bold")

    # Trade counts
    buy_counts   = df["buy/n_trades"].fillna(0)
    short_counts = df["short/n_trades"].fillna(0)
    axes[0].bar(x - w/2, buy_counts,   w, label="BUY trades",   color="#2ca02c")
    axes[0].bar(x + w/2, short_counts, w, label="SHORT trades", color="#d62728")
    axes[0].set_xticks(x); axes[0].set_xticklabels(tickers)
    axes[0].set_ylabel("Number of trades"); axes[0].set_title("Trade Counts")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # Avg profit per trade
    buy_profit   = df["buy/profit_pct/avg_per_trade"].fillna(0)
    short_profit = df["short/profit_pct/avg_per_trade"].fillna(0)
    axes[1].bar(x - w/2, buy_profit,   w, label="BUY avg profit %",   color="#2ca02c")
    axes[1].bar(x + w/2, short_profit, w, label="SHORT avg profit %", color="#d62728")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(tickers)
    axes[1].set_ylabel("Avg profit per trade (%)"); axes[1].set_title("Avg Profit per Trade")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    _save(fig, out_dir / "07_trade_stats.png", show)


# ---------------------------------------------------------------------------
# 08 — Probability distributions
# ---------------------------------------------------------------------------
def plot_prob_distributions(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    if "predictions" not in dfs:
        return

    df    = dfs["predictions"]
    pcols = [c for c in ["prob_SHORT", "prob_HOLD", "prob_BUY"] if c in df.columns]
    if not pcols:
        print("  [skip] No probability columns in predictions.csv — skipping plot.")
        return

    colours = {"prob_SHORT": "#d62728", "prob_HOLD": "#7f7f7f", "prob_BUY": "#2ca02c"}
    labels  = {"prob_SHORT": "SHORT", "prob_HOLD": "HOLD", "prob_BUY": "BUY"}

    fig, axes = plt.subplots(1, len(pcols), figsize=(5 * len(pcols), 4), sharey=False)
    if len(pcols) == 1:
        axes = [axes]

    fig.suptitle("Predicted Class Probability Distributions", fontweight="bold")
    for ax, col in zip(axes, pcols):
        ax.hist(df[col].dropna(), bins=20, color=colours[col], edgecolor="white", alpha=0.85)
        ax.set_xlabel("Probability"); ax.set_ylabel("Count")
        ax.set_title(f"P({labels[col]})")
        ax.grid(axis="y", alpha=0.3)

    _save(fig, out_dir / "08_prob_distributions.png", show)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive by default; overridden if --show
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate evaluation figures from run_predict.py CSVs.")
    parser.add_argument(
        "--eval-dir", required=True,
        help="Path to an eval directory produced by run_predict.py.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to save figures. Defaults to <eval-dir>/figures/",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show each figure interactively after saving.",
    )
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"Eval directory not found: {eval_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Eval dir : {eval_dir}")
    print(f"Figures  : {out_dir}\n")

    dfs = _load_csvs(eval_dir)

    plot_confusion_matrix(dfs, out_dir, args.show)
    plot_classification_metrics(dfs, out_dir, args.show)
    plot_label_distribution(dfs, out_dir, args.show)
    plot_prediction_distribution(dfs, out_dir, args.show)
    plot_per_ticker_accuracy(dfs, out_dir, args.show)
    plot_equity_curve(dfs, out_dir, args.show)
    plot_trade_stats(dfs, out_dir, args.show)
    plot_prob_distributions(dfs, out_dir, args.show)

    saved = list(out_dir.glob("*.png"))
    print(f"\nDone — {len(saved)} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
