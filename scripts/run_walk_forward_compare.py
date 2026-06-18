"""
scripts/run_walk_forward_compare.py — Compare walk-forward runs across parameter sets.

Usage
-----
  python scripts/run_walk_forward_compare.py
  python scripts/run_walk_forward_compare.py --run-ids wf_a wf_b wf_c
  python scripts/run_walk_forward_compare.py --max-runs 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kvant.experiment.prepare import PREPARED_DATA_ROOT
from kvant.utils.pipeline_config import load_pipeline_config


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_runs(walk_root: Path, run_ids: list[str] | None, max_runs: int) -> list[dict[str, Any]]:
    if run_ids:
        roots = [walk_root / run_id for run_id in run_ids]
    else:
        roots = sorted([p for p in walk_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:max_runs]

    loaded: list[dict[str, Any]] = []
    for run_root in roots:
        manifest_path = run_root / "walk_forward_manifest.json"
        if not manifest_path.exists():
            print(f"[skip] Missing manifest: {run_root}")
            continue

        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            print(f"[skip] Invalid JSON manifest: {manifest_path}")
            continue

        aggregate_dir_raw = str(manifest.get("aggregate_dir", "")).strip()
        if not aggregate_dir_raw:
            print(f"[skip] No aggregate_dir in manifest: {manifest_path}")
            continue

        aggregate_dir = Path(aggregate_dir_raw)
        fold_path = aggregate_dir / "fold_summary.csv"
        return_path = aggregate_dir / "return_stats.csv"
        class_path = aggregate_dir / "classification_metrics.csv"
        run_meta_path = aggregate_dir / "run_meta.csv"

        if not fold_path.exists() or not return_path.exists() or not class_path.exists():
            print(f"[skip] Missing aggregate CSVs for run: {run_root.name}")
            continue

        fold_df = pd.read_csv(fold_path)
        return_df = pd.read_csv(return_path)
        class_df = pd.read_csv(class_path)
        run_meta_df = pd.read_csv(run_meta_path) if run_meta_path.exists() else pd.DataFrame()

        loaded.append(
            {
                "run_id": run_root.name,
                "run_root": run_root,
                "manifest": manifest,
                "fold_summary": fold_df,
                "return_stats": return_df,
                "classification_metrics": class_df,
                "run_meta": run_meta_df,
            }
        )
    return loaded


def _build_run_summary(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run in runs:
        fold_df = run["fold_summary"]
        return_row = run["return_stats"].iloc[0].to_dict()
        class_df = run["classification_metrics"]
        overall_row = class_df[class_df["class"] == "overall"]
        overall_acc = _safe_float(overall_row.iloc[0]["accuracy"]) if not overall_row.empty else float("nan")

        manifest = run["manifest"]
        config_sig = manifest.get("config_signature", {}) if isinstance(manifest.get("config_signature", {}), dict) else {}

        row = {
            "run_id": run["run_id"],
            "n_folds": int(len(fold_df)),
            "n_folds_completed": int(manifest.get("n_folds_completed", len(fold_df))),
            "n_folds_skipped": int(manifest.get("n_folds_skipped", 0)),
            "overall_accuracy": overall_acc,
            "overall_accuracy_fold_mean": _safe_float(fold_df.get("overall_accuracy", pd.Series(dtype=float)).mean()),
            "overall_accuracy_fold_std": _safe_float(fold_df.get("overall_accuracy", pd.Series(dtype=float)).std(ddof=0)),
            "final_portfolio_pnl_pct": _safe_float(return_row.get("portfolio_cumulative_pnl_pct")),
            "final_portfolio_pnl_net_pct": _safe_float(return_row.get("portfolio_cumulative_pnl_net_pct")),
            "n_directional_trades": _safe_float(return_row.get("n_directional_trades")),
            "avg_profit_per_trade_pct": _safe_float(return_row.get("avg_profit_per_trade_pct")),
            "directional_accuracy": _safe_float(return_row.get("directional_accuracy")),
            "win_rate": _safe_float(return_row.get("win_rate")),
            "prepare.lookback": config_sig.get("prepare", {}).get("lookback", ""),
            "prepare.width_minutes": config_sig.get("prepare", {}).get("width_minutes", ""),
            "prepare.height_pct": config_sig.get("prepare", {}).get("height_pct", ""),
            "train.model": config_sig.get("train", {}).get("model", ""),
            "train.learning_rate": config_sig.get("train", {}).get("learning_rate", ""),
            "predict.top_k_per_timestamp": config_sig.get("predict", {}).get("top_k_per_timestamp", ""),
            "predict.ticker_cooldown_minutes": config_sig.get("predict", {}).get("ticker_cooldown_minutes", ""),
            "predict.max_concurrent_positions_per_ticker": config_sig.get("predict", {}).get(
                "max_concurrent_positions_per_ticker", ""
            ),
            "meta.enabled": config_sig.get("meta", {}).get("enabled", ""),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("final_portfolio_pnl_net_pct", ascending=False, na_position="last").reset_index(drop=True)
    return summary


def _save_plot_metric_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return

    df = summary.copy()
    x = range(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Walk-Forward Run Comparison", fontweight="bold")

    axes[0, 0].bar(x, df["final_portfolio_pnl_net_pct"], color="#4C72B0")
    axes[0, 0].set_title("Final Portfolio PnL Net (%)")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)

    axes[0, 1].bar(x, df["overall_accuracy"], color="#55A868")
    axes[0, 1].set_title("Overall Accuracy")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(x, df["directional_accuracy"], color="#DD8452")
    axes[1, 0].set_title("Directional Accuracy")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].bar(x, df["n_directional_trades"], color="#8172B3")
    axes[1, 1].set_title("Directional Trades")

    run_labels = df["run_id"].tolist()
    for ax in axes.ravel():
        ax.set_xticks(list(x))
        ax.set_xticklabels(run_labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "01_run_metric_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_plot_risk_return(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return

    df = summary.dropna(subset=["overall_accuracy_fold_std", "final_portfolio_pnl_net_pct"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["overall_accuracy_fold_std"], df["final_portfolio_pnl_net_pct"], c="#C44E52", s=70, alpha=0.9)

    for _, row in df.iterrows():
        ax.text(float(row["overall_accuracy_fold_std"]) + 0.001, float(row["final_portfolio_pnl_net_pct"]) + 0.01, str(row["run_id"]), fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fold Accuracy Std (lower is more stable)")
    ax.set_ylabel("Final Portfolio PnL Net (%)")
    ax.set_title("Stability vs Net Return", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "02_stability_vs_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_plot_fold_distributions(runs: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    parts = []
    for run in runs:
        df = run["fold_summary"].copy()
        if "overall_accuracy" not in df.columns:
            continue
        for value in df["overall_accuracy"].dropna().tolist():
            parts.append({"run_id": run["run_id"], "overall_accuracy": float(value)})

    if not parts:
        return

    acc_df = pd.DataFrame(parts)
    labels = sorted(acc_df["run_id"].unique().tolist())
    data = [acc_df.loc[acc_df["run_id"] == label, "overall_accuracy"].to_numpy() for label in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8CC0DE")

    ax.set_title("Fold Accuracy Distribution by Run", fontweight="bold")
    ax.set_ylabel("Overall Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_dir / "03_fold_accuracy_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare walk-forward runs across parameter sets.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    parser.add_argument("--run-ids", nargs="*", default=None, help="Optional explicit run ids to compare.")
    parser.add_argument("--max-runs", type=int, default=10, help="Max recent runs when --run-ids is not set.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory. Defaults to <prepared_root>/walkforward/comparisons/latest",
    )
    args = parser.parse_args()

    pipeline_cfg, cfg_path = load_pipeline_config(args.config)
    prepared_root = Path(pipeline_cfg["paths"].get("prepared_root", str(PREPARED_DATA_ROOT)))
    walk_root = prepared_root / "walkforward"
    if not walk_root.exists():
        raise SystemExit(f"Walk-forward root does not exist: {walk_root}")

    runs = _load_runs(walk_root, args.run_ids, args.max_runs)
    if len(runs) < 2:
        raise SystemExit("Need at least 2 completed walk-forward runs to compare.")

    out_dir = Path(args.out_dir) if args.out_dir else (walk_root / "comparisons" / "latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_run_summary(runs)
    summary_path = out_dir / "run_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    rows = []
    for run in runs:
        fold_df = run["fold_summary"].copy()
        fold_df.insert(0, "run_id", run["run_id"])
        rows.append(fold_df)
    fold_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fold_path = out_dir / "fold_comparison.csv"
    fold_all.to_csv(fold_path, index=False)

    _save_plot_metric_bars(summary, out_dir)
    _save_plot_risk_return(summary, out_dir)
    _save_plot_fold_distributions(runs, out_dir)

    print(f"Config          : {cfg_path}")
    print(f"Compared runs   : {len(runs)}")
    print(f"Output dir      : {out_dir.resolve()}")
    print(f"Saved summary   : {summary_path.name}")
    print("Saved figures   : 01_run_metric_bars.png, 02_stability_vs_return.png, 03_fold_accuracy_boxplot.png")


if __name__ == "__main__":
    main()
