from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def _daily_sample_counts_from_timestamps(ts: np.ndarray) -> pd.Series:
    """
    ts: np.ndarray of datetime64[ns] (as saved by prepare_experiment)
    Returns: Series indexed by day (Timestamp, midnight) with counts per day.
    """
    if ts is None or len(ts) == 0:
        return pd.Series(dtype=np.int64)

    dt = pd.to_datetime(ts)  # tz-naive OK
    days = pd.Series(dt).dt.floor("D")
    counts = days.value_counts().sort_index()
    counts.index.name = "day"
    counts.name = "samples"
    return counts

def _save_hist_png(values: np.ndarray, out_path: Path, title: str, bins: int = 50) -> None:
    """
    Uses matplotlib if available. If not, writes histogram bins to JSON instead.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        # nothing to plot
        (out_path.with_suffix(".json")).write_text(json.dumps({"title": title, "empty": True}, indent=2))
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        # fallback: save histogram data
        hist, edges = np.histogram(values, bins=bins)
        payload = {
            "title": title,
            "bins": int(bins),
            "edges": edges.tolist(),
            "counts": hist.tolist(),
        }
        (out_path.with_suffix(".json")).write_text(json.dumps(payload, indent=2))
        return

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel("samples per day")
    plt.ylabel("number of days")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def report_sampling_density(
    exp_dir: Path,
    *,
    tickers: Optional[List[str]] = None,
    bins: int = 50,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Reads prepared artifacts from exp_dir and produces:
      - per-ticker daily counts CSV
      - per-ticker histogram PNG of daily samples
      - global histogram PNG over all ticker-days
      - global histogram PNG over per-ticker mean samples/day

    Returns a DataFrame summary (also saved to exp_dir/sampling_report.csv).
    """
    tickers_root = exp_dir / "tickers"
    if tickers is None:
        tickers = sorted([p.name for p in tickers_root.iterdir() if p.is_dir()])

    out_report_dir = exp_dir / "reports"
    out_report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_daily_values = []
    per_ticker_means = []

    for t in tickers:
        tdir = tickers_root / t
        ts_path = tdir / "timestamps.npy"
        meta_path = tdir / "meta.json"
        if not ts_path.exists() or not meta_path.exists():
            continue

        ts = np.load(ts_path, mmap_mode="r")
        meta = _load_json(meta_path)

        n_raw = meta.get("n_rows_raw_full", None)  # requires you to include this in meta during prepare
        n_sampled = meta.get("n_rows_sampled_full", meta.get("n_rows_full", len(ts)))

        counts = _daily_sample_counts_from_timestamps(ts)
        n_days = int(counts.shape[0])
        mean_bpd = float(counts.mean()) if n_days else 0.0
        median_bpd = float(counts.median()) if n_days else 0.0

        all_daily_values.append(counts.to_numpy(dtype=float))
        per_ticker_means.append(mean_bpd)

        # Save per-day counts
        counts_csv = out_report_dir / f"{t}_samples_per_day.csv"
        counts.to_csv(counts_csv, header=True)

        # Per-ticker histogram
        _save_hist_png(
            counts.to_numpy(dtype=float),
            out_report_dir / f"{t}_samples_per_day_hist.png",
            title=f"{t}: samples/day distribution (n_days={n_days}, mean={mean_bpd:.2f})",
            bins=bins,
        )

        retention = None
        if isinstance(n_raw, (int, float)) and n_raw and n_sampled is not None:
            retention = float(n_sampled) / float(n_raw)

        rows.append({
            "ticker": t,
            "n_raw_full": n_raw,
            "n_sampled_full": int(n_sampled) if n_sampled is not None else None,
            "retention_ratio": retention,
            "n_days": n_days,
            "samples_per_day_mean": mean_bpd,
            "samples_per_day_median": median_bpd,
            "samples_per_day_min": float(counts.min()) if n_days else 0.0,
            "samples_per_day_max": float(counts.max()) if n_days else 0.0,
        })

    df = pd.DataFrame(rows).sort_values(["samples_per_day_mean", "ticker"], ascending=[False, True])
    df.to_csv(out_report_dir / "sampling_report.csv", index=False)

    # Global hist: all ticker-days combined
    if len(all_daily_values):
        all_vals = np.concatenate(all_daily_values, axis=0)
    else:
        all_vals = np.array([], dtype=float)

    _save_hist_png(
        all_vals,
        out_report_dir / "ALL_TICKERS_samples_per_day_hist.png",
        title=f"ALL TICKERS: samples/day distribution over ticker-days (n={len(all_vals)})",
        bins=bins,
    )

    # Global hist: per-ticker mean samples/day (one value per ticker)
    _save_hist_png(
        np.asarray(per_ticker_means, dtype=float),
        out_report_dir / "PER_TICKER_mean_samples_per_day_hist.png",
        title=f"PER TICKER: mean samples/day (n_tickers={len(per_ticker_means)})",
        bins=min(bins, max(10, len(per_ticker_means))),
    )

    if print_table and len(df):
        # Pretty minimal console table
        cols = ["ticker", "n_raw_full", "n_sampled_full", "retention_ratio", "n_days",
                "samples_per_day_mean", "samples_per_day_median"]
        print("\nSampling density report (top 50):")
        print(df[cols].head(50).to_string(index=False))

    return df