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
09_directional_drift.png       — per-ticker directional drift / opposite-rate diagnostics
10_directional_calibration.png — confidence-binned directional accuracy (if available)

Usage
-----
  python scripts/run_plot.py
  python scripts/run_plot.py --config pipeline.toml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kvant.utils.ensemble import ensemble_slug, normalize_model_names
from kvant.utils.pipeline_config import load_pipeline_config


def _load_csvs(eval_dir: Path) -> dict:
    import pandas as pd
    required = [
        "classification_metrics", "confusion_matrix", "label_distribution",
        "predictions", "trade_stats", "return_stats", "equity_curve", "run_meta",
        "directional_drift", "directional_calibration",
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

    has_portfolio = "cumulative_portfolio_pnl_pct" in df.columns
    has_net = "cumulative_portfolio_pnl_net_pct" in df.columns

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df))

    import pandas as pd

    title_parts = []

    # Portfolio curve (pool-allocated, before fees)
    if has_portfolio:
        port = df["cumulative_portfolio_pnl_pct"]
        ax.plot(x, port, linewidth=1.5, color="#4C72B0", label="Portfolio (gross)")
        ax.fill_between(x, port, 0, where=port >= 0, alpha=0.1, color="green")
        ax.fill_between(x, port, 0, where=port < 0, alpha=0.1, color="red")
        final_port = float(port.iloc[-1])
        title_parts.append(f"portfolio: {final_port:+.2f}%")

    # Portfolio net-of-fees curve
    if has_net:
        net = df["cumulative_portfolio_pnl_net_pct"]
        ax.plot(x, net, linewidth=1.5, color="#DD8452", label="Portfolio (net of fees)")
        final_net = float(net.iloc[-1])
        title_parts.append(f"net: {final_net:+.2f}%")

    # Buy-and-hold benchmark (equal-weight across evaluated tickers)
    bnh_return = _buy_and_hold_return(df, dfs.get("predictions"))
    if bnh_return is not None:
        # Interpolate the B&H return linearly across trade indices so the
        # x-axis (Trade #) stays consistent with the other curves.
        bnh_line = [bnh_return * i / max(len(df) - 1, 1) for i in range(len(df))]
        ax.plot(x, bnh_line, linewidth=1.2, color="#7f7f7f", linestyle="--",
                label=f"Buy & Hold ({bnh_return:+.2f}%)")
        title_parts.append(f"B&H: {bnh_return:+.2f}%")

    # Show skipped-trade count if available
    if "skipped" in df.columns:
        n_skipped = int(df["skipped"].sum())
        n_total = len(df)
        title_parts.append(f"{n_skipped}/{n_total} skipped")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL (%)")
    ax.set_title(f"Equity Curve  ({',  '.join(title_parts)})", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir / "06_equity_curve.png", show)


def _buy_and_hold_return(eq_df, predictions_df=None) -> float | None:
    """
    Compute equal-weight buy-and-hold return over the active trading window.

    Window: first possible trade timestamp to last possible trade timestamp,
    derived from equity_curve rows. Universe: evaluated tickers from
    predictions.csv when available; falls back to equity curve tickers.
    """
    import pandas as pd

    store_dir = _PROJECT_ROOT / "data" / "1m"
    if not store_dir.exists():
        return None

    if "timestamp" not in eq_df.columns:
        return None

    trade_ts = pd.to_datetime(eq_df["timestamp"], errors="coerce", utc=True).dropna()
    if len(trade_ts) < 2:
        return None
    t_min = trade_ts.min()
    t_max = trade_ts.max()

    ticker_values = None
    if predictions_df is not None and "ticker" in predictions_df.columns:
        ticker_values = predictions_df["ticker"]
    elif "ticker" in eq_df.columns:
        ticker_values = eq_df["ticker"]
    if ticker_values is None:
        return None

    tickers = [str(t) for t in pd.Series(ticker_values).dropna().unique().tolist()]
    if not tickers:
        return None

    returns = []
    for ticker in tickers:
        raw = _load_price_history(store_dir, ticker)
        if raw is None or "close" not in raw.columns:
            continue
        sliced = raw.loc[t_min:t_max, "close"].dropna()
        if len(sliced) < 2:
            continue
        ret = (float(sliced.iloc[-1]) / float(sliced.iloc[0]) - 1.0) * 100.0
        returns.append(ret)

    if not returns:
        return None
    return sum(returns) / len(returns)


def _load_price_history(store_dir: Path, ticker: str):
    """Load flat and month-partitioned OHLCV files for one ticker."""
    import pandas as pd

    paths: list[Path] = []
    flat_path = store_dir / f"{ticker}.csv"
    if flat_path.exists():
        paths.append(flat_path)
    paths.extend(sorted(store_dir.glob(f"????-??/{ticker}.csv")))

    frames = []
    for path in paths:
        raw = pd.read_csv(path)
        if "timestamp" not in raw.columns:
            continue
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
        raw = raw.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        frames.append(raw)

    if not frames:
        return None

    out = pd.concat(frames, axis=0, sort=True).sort_index()
    return out[~out.index.duplicated(keep="last")]


# ---------------------------------------------------------------------------
# 06b — Equity curve vs time (with time-indexed B&H)
# ---------------------------------------------------------------------------
def plot_equity_curve_time(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np

    if "equity_curve" not in dfs:
        print("  [skip] No equity_curve data — skipping time-based equity curve.")
        return

    df = dfs["equity_curve"].copy()
    if df.empty:
        print("  [skip] equity_curve.csv is empty — skipping time-based equity curve.")
        return
    if "timestamp" not in df.columns or "cumulative_portfolio_pnl_pct" not in df.columns:
        print("  [skip] equity_curve.csv missing required columns — skipping time-based equity curve.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "cumulative_portfolio_pnl_pct"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) == 0:
        print("  [skip] equity_curve.csv has no valid rows after parsing — skipping time-based equity curve.")
        return

    ts = df["timestamp"].values
    port = df["cumulative_portfolio_pnl_pct"].values

    # --- Build time-indexed B&H curve from raw 1m CSVs ---
    store_dir = _PROJECT_ROOT / "data" / "1m"
    bnh_series = None
    if store_dir.exists() and "ticker" in df.columns:
        bnh_window = None
        tickers = df["ticker"].dropna().unique()
        if "predictions" in dfs and {"timestamp", "ticker"}.issubset(dfs["predictions"].columns):
            pred = dfs["predictions"].copy()
            pred["timestamp"] = pd.to_datetime(pred["timestamp"], errors="coerce", utc=True)
            pred = pred.dropna(subset=["timestamp", "ticker"])
            if len(pred) >= 2:
                bnh_window = (pred["timestamp"].min(), pred["timestamp"].max())
                tickers = pred["ticker"].dropna().unique()

        if bnh_window is None:
            bnh_window = (df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

        t_min, t_max = bnh_window
        cum_returns = []
        for ticker in tickers:
            raw = _load_price_history(store_dir, ticker)
            if raw is None or "close" not in raw.columns:
                continue
            sliced = raw.loc[t_min:t_max, "close"].dropna()
            if len(sliced) < 2:
                continue
            cum_ret = (sliced / float(sliced.iloc[0]) - 1.0) * 100.0
            cum_returns.append(cum_ret)

        if cum_returns:
            combined = pd.concat(cum_returns, axis=1, sort=True).sort_index().ffill()
            bnh_series = combined.mean(axis=1)

    has_net = "cumulative_portfolio_pnl_net_pct" in df.columns

    fig, ax = plt.subplots(figsize=(14, 5))
    title_parts = []

    # Portfolio gross — step plot, shading done separately without `where`
    ax.plot(ts, port, linewidth=1.5, color="#4C72B0",
            drawstyle="steps-post", label="Portfolio (gross)")
    ax.fill_between(ts, port, 0, alpha=0.1, color="red", step="post")

    title_parts.append(f"portfolio: {float(port[-1]):+.2f}%")

    # Portfolio net of fees
    if has_net:
        net = df["cumulative_portfolio_pnl_net_pct"].values
        ax.plot(ts, net, linewidth=1.5, color="#DD8452",
                drawstyle="steps-post", label="Portfolio (net of fees)")
        title_parts.append(f"net: {float(net[-1]):+.2f}%")

    # B&H over time (continuous)
    if bnh_series is not None and len(bnh_series) > 1:
        bnh_final = float(bnh_series.iloc[-1])
        ax.plot(bnh_series.index, bnh_series.values, linewidth=1.2,
                color="#7f7f7f", linestyle="--", label=f"Buy & Hold ({bnh_final:+.2f}%)")
        title_parts.append(f"B&H: {bnh_final:+.2f}%")

    if "skipped" in df.columns:
        n_skipped = int(df["skipped"].sum())
        title_parts.append(f"{n_skipped}/{len(df)} skipped")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PnL (%)")
    ax.set_title(f"Equity Curve  ({',  '.join(title_parts)})", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir / "06b_equity_curve_time.png", show)


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
# 09 — Directional drift diagnostics
# ---------------------------------------------------------------------------
def plot_directional_drift(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "directional_drift" not in dfs:
        return

    df = dfs["directional_drift"].copy()
    if df.empty:
        return
    df = df[df["ticker"] != "ALL"] if "ticker" in df.columns else df
    if df.empty:
        return

    tickers = df["ticker"].astype(str).tolist()
    x = np.arange(len(tickers))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Directional Drift Diagnostics", fontweight="bold")

    axes[0].bar(x - w / 2, df["true_buy_rate"],  w, label="true buy rate",  color="#98df8a")
    axes[0].bar(x + w / 2, df["pred_buy_rate"],  w, label="pred buy rate",  color="#2ca02c")
    axes[0].bar(x - w / 2, -df["true_short_rate"], w, label="true short rate", color="#ff9896")
    axes[0].bar(x + w / 2, -df["pred_short_rate"], w, label="pred short rate", color="#d62728")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(tickers)
    axes[0].set_title("Buy/Short Rate (short shown below zero)")
    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    opp = df["directional_opposite_rate"].fillna(0.0)
    bars = axes[1].bar(tickers, opp, color="#9467bd")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Opposite Direction Rate")
    axes[1].set_ylabel("Rate")
    axes[1].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, opp):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    _save(fig, out_dir / "09_directional_drift.png", show)


# ---------------------------------------------------------------------------
# 10 — Directional calibration curve-ish bars
# ---------------------------------------------------------------------------
def plot_directional_calibration(dfs: dict, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    if "directional_calibration" not in dfs:
        return
    df = dfs["directional_calibration"].copy()
    if df.empty:
        return

    if "ticker" in df.columns:
        df = df[df["ticker"] == "ALL"]
    if df.empty:
        return

    df = df.sort_values("avg_confidence")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(df["confidence_bin"].astype(str), df["directional_accuracy"], color="#4C72B0", alpha=0.9)
    ax.plot(df["confidence_bin"].astype(str), df["avg_confidence"], color="#dd8452", marker="o", label="avg confidence")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Directional Calibration (ALL tickers)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "10_directional_calibration.png", show)


# ---------------------------------------------------------------------------
# 11 — Backtest comparison: price + predictions vs buy-and-hold
# ---------------------------------------------------------------------------
def plot_backtest_comparison(dfs: dict, out_dir: Path, show: bool, max_tickers: int = 6) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if "predictions" not in dfs:
        return

    pred_df = dfs["predictions"].copy()
    if "timestamp" not in pred_df.columns or "ticker" not in pred_df.columns:
        return

    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True)
    pred_df = pred_df.sort_values("timestamp")

    # Try to load raw price data from data/1m/
    store_dir = _PROJECT_ROOT / "data" / "1m"
    if not store_dir.exists():
        print("  [skip] No data/1m/ directory found — skipping backtest comparison.")
        return

    # Restrict to tickers that actually generated trades (BUY / SHORT predictions)
    # in this eval split — HOLD-only tickers produce empty backtest panels.
    traded_mask = pred_df["y_pred_name"].isin(["BUY", "SHORT"])
    traded_counts = pred_df.loc[traded_mask, "ticker"].value_counts()
    if traded_counts.empty:
        print("  [skip] No BUY/SHORT predictions in this split — skipping backtest comparison.")
        return
    tickers = traded_counts.head(max_tickers).index.tolist()

    n = len(tickers)
    fig, axes = plt.subplots(n, 2, figsize=(18, 5 * n), squeeze=False)
    fig.suptitle("Backtest: Model Predictions vs Buy-and-Hold", fontsize=16, fontweight="bold", y=1.01)

    # HOLD signals are intentionally omitted from the price-panel overlay:
    # they don't open trades, so they only add visual noise.
    colour_map = {"SHORT": "#d62728", "BUY": "#2ca02c"}
    marker_map = {"SHORT": "v", "BUY": "^"}

    for row_idx, ticker in enumerate(tickers):
        ax_price = axes[row_idx, 0]
        ax_returns = axes[row_idx, 1]

        # Load raw 1m price data
        csv_path = store_dir / f"{ticker}.csv"
        if not csv_path.exists():
            ax_price.text(0.5, 0.5, f"No CSV for {ticker}", ha="center", va="center")
            ax_returns.text(0.5, 0.5, f"No CSV for {ticker}", ha="center", va="center")
            continue

        raw = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")

        # Filter to test period (timestamps in predictions)
        ticker_preds = pred_df[pred_df["ticker"] == ticker].copy()
        if ticker_preds.empty:
            continue

        t_min = ticker_preds["timestamp"].min() - pd.Timedelta(minutes=30)
        t_max = ticker_preds["timestamp"].max() + pd.Timedelta(minutes=30)
        price_slice = raw.loc[t_min:t_max].copy()

        if price_slice.empty or "close" not in price_slice.columns:
            continue

        close = price_slice["close"]

        # --- Left panel: price chart with prediction markers ---
        ax_price.plot(close.index, close.values, color="#4C72B0", linewidth=0.8, alpha=0.8, label="Close")

        for pred_name, colour in colour_map.items():
            subset = ticker_preds[ticker_preds["y_pred_name"] == pred_name]
            if subset.empty:
                continue
            # Match prediction timestamps to closest price
            matched_prices = []
            matched_times = []
            for ts in subset["timestamp"]:
                idx = close.index.searchsorted(ts)
                if idx < len(close):
                    matched_times.append(close.index[idx])
                    matched_prices.append(close.iloc[idx])
            if matched_prices:
                ax_price.scatter(
                    matched_times, matched_prices,
                    c=colour, marker=marker_map[pred_name],
                    s=30, alpha=0.7, label=f"Pred: {pred_name}", zorder=5,
                )

        ax_price.set_title(f"{ticker} — Price + Predictions", fontweight="bold")
        ax_price.set_ylabel("Price ($)")
        ax_price.legend(fontsize=7, loc="upper left")
        ax_price.grid(alpha=0.3)
        ax_price.tick_params(axis="x", rotation=30)

        # --- Right panel: cumulative returns comparison ---
        # Buy-and-hold: simple cumulative return from first to last bar
        bnh_returns = close.pct_change().fillna(0)
        bnh_cumulative = (1 + bnh_returns).cumprod() - 1

        # Model strategy: apply prediction at each signal timestamp
        # BUY → +1 position, SHORT → -1 position, HOLD → 0 (flat)
        position_map = {"BUY": 1.0, "SHORT": -1.0, "HOLD": 0.0}
        signal_series = pd.Series(0.0, index=close.index)
        for _, pred_row in ticker_preds.iterrows():
            ts = pred_row["timestamp"]
            idx = close.index.searchsorted(ts)
            if idx < len(close):
                signal_series.iloc[idx] = position_map.get(pred_row["y_pred_name"], 0.0)

        # Forward-fill signals until next signal
        position = signal_series.replace(0.0, np.nan)
        # Set first value to 0 (flat) if no signal yet
        if pd.isna(position.iloc[0]):
            position.iloc[0] = 0.0
        position = position.ffill().fillna(0.0)

        model_returns = position.shift(1).fillna(0.0) * bnh_returns
        model_cumulative = (1 + model_returns).cumprod() - 1

        ax_returns.plot(bnh_cumulative.index, bnh_cumulative.values * 100,
                        color="#4C72B0", linewidth=1.5, label="Buy & Hold")
        ax_returns.plot(model_cumulative.index, model_cumulative.values * 100,
                        color="#DD8452", linewidth=1.5, label="Model Strategy")
        ax_returns.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_returns.fill_between(model_cumulative.index, model_cumulative.values * 100, 0,
                                where=model_cumulative.values >= 0, alpha=0.15, color="green")
        ax_returns.fill_between(model_cumulative.index, model_cumulative.values * 100, 0,
                                where=model_cumulative.values < 0, alpha=0.15, color="red")

        final_bnh = float(bnh_cumulative.iloc[-1]) * 100
        final_model = float(model_cumulative.iloc[-1]) * 100
        ax_returns.set_title(
            f"{ticker} — Returns: Model {final_model:+.2f}% vs B&H {final_bnh:+.2f}%",
            fontweight="bold",
        )
        ax_returns.set_ylabel("Cumulative Return (%)")
        ax_returns.legend(fontsize=8)
        ax_returns.grid(alpha=0.3)
        ax_returns.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, out_dir / "11_backtest_comparison.png", show)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PREPARED_ROOT = _PROJECT_ROOT / "prepared"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive by default; overridden if --show
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate evaluation figures from run_predict.py CSVs.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    parser.add_argument(
        "--eval-dir",
        default="",
        help="Optional explicit evaluation directory. Overrides auto-detection.",
    )
    args = parser.parse_args()

    cfg, cfg_path = load_pipeline_config(args.config)
    plot_cfg = cfg.get("plot", {})
    predict_cfg = cfg.get("predict", {})
    paths_cfg = cfg.get("paths", {})

    show = bool(plot_cfg.get("show", False))
    ensemble_models = normalize_model_names(cfg.get("ensemble", {}).get("models"))
    if ensemble_models:
        model_name = ensemble_slug(ensemble_models)
    else:
        model_name = str(plot_cfg.get("model", predict_cfg.get("model", "conv1d")))
    split = str(plot_cfg.get("split", predict_cfg.get("split", "test")))
    exp_id = str(plot_cfg.get("experiment_id", "last"))
    walk_cfg = cfg.get("walk_forward", {})

    prepared_root = Path(paths_cfg.get("prepared_root", str(_PREPARED_ROOT)))

    if show:
        matplotlib.use("TkAgg")

    eval_dir_arg = str(args.eval_dir).strip()
    if eval_dir_arg:
        eval_dir = Path(eval_dir_arg)
        print(f"Using explicit eval dir: {eval_dir}")
    elif bool(walk_cfg.get("enabled", False)):
        last_wf_file = prepared_root / "last_walk_forward.txt"
        if not last_wf_file.exists():
            raise SystemExit(
                f"walk_forward.enabled=true but no {last_wf_file.name} found in {prepared_root}. "
                "Run scripts/run_walk_forward.py first or pass --eval-dir."
            )
        run_root = Path(last_wf_file.read_text().strip())
        manifest_path = run_root / "walk_forward_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"Walk-forward manifest not found: {manifest_path}. "
                "Run scripts/run_walk_forward.py first or pass --eval-dir."
            )
        import json
        manifest = json.loads(manifest_path.read_text())
        aggregate_dir = str(manifest.get("aggregate_dir", "")).strip()
        if not aggregate_dir:
            raise SystemExit(
                f"No aggregate_dir in {manifest_path}. "
                "Run scripts/run_walk_forward.py first or pass --eval-dir."
            )
        eval_dir = Path(aggregate_dir)
        print(f"Auto-detected walk-forward eval dir: {eval_dir}")
    else:
        if exp_id == "last":
            last_file = prepared_root / "last_experiment.txt"
            if not last_file.exists():
                raise SystemExit(f"No last_experiment.txt found in {prepared_root}.")
            exp_id = last_file.read_text().strip()
        eval_dir = prepared_root / exp_id / "eval" / f"{model_name}_{split}"
        print(f"Auto-detected eval dir: {eval_dir}")

    if not eval_dir.exists():
        raise SystemExit(f"Eval directory not found: {eval_dir}. Run run_predict.py first.")

    out_dir = eval_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Eval dir : {eval_dir}")
    print(f"Config   : {cfg_path}")
    print(f"Figures  : {out_dir}\n")

    dfs = _load_csvs(eval_dir)

    plot_confusion_matrix(dfs, out_dir, show)
    plot_classification_metrics(dfs, out_dir, show)
    plot_label_distribution(dfs, out_dir, show)
    plot_prediction_distribution(dfs, out_dir, show)
    plot_per_ticker_accuracy(dfs, out_dir, show)
    plot_equity_curve(dfs, out_dir, show)
    plot_equity_curve_time(dfs, out_dir, show)
    plot_trade_stats(dfs, out_dir, show)
    plot_prob_distributions(dfs, out_dir, show)
    plot_directional_drift(dfs, out_dir, show)
    plot_directional_calibration(dfs, out_dir, show)
    plot_backtest_comparison(dfs, out_dir, show)

    saved = list(out_dir.glob("*.png"))
    print(f"\nDone — {len(saved)} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
