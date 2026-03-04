from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _run_label(r: dict[str, Any]) -> str:
    p = r["params"]
    return f"w={p['width_minutes']}, h={p['height']}"
def plot_per_ticker_split_counts(
    payload: dict[str, Any],
    *,
    split: str,
    out_path: Path | None = None,
):
    runs = payload["runs"]
    classes = payload["tb_classes"]

    tickers = runs[0]["stats"]["tickers_all"]
    x = np.arange(len(runs))

    nrows = len(tickers)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(12, max(3.0, 2.2 * nrows)),
        sharex=True
    )
    if nrows == 1:
        axes = [axes]

    # grouped bar geometry
    n_cls = len(classes)
    group_width = 0.8
    bar_w = group_width / n_cls
    offsets = (np.arange(n_cls) - (n_cls - 1) / 2.0) * bar_w

    for ax, t in zip(axes, tickers):
        for j, cls in enumerate(classes):
            vals = []
            for r in runs:
                cc = r["stats"]["per_ticker"][t][split]["class_counts"]
                vals.append( float(cc.get(cls, 0)))
            vals = np.asarray(vals, dtype=float)

            ax.bar(x + offsets[j], vals, width=bar_w, label=cls, edgecolor="black", linewidth=0.3)

        ax.set_title(f"{t} | split={split}")
        ax.set_ylabel("count")

        # Use symlog so 0 is displayable (log can't show 0)
        ax.set_yscale("symlog", linthresh=1.0)

        # reasonable y-lims
        all_vals = []
        for r in runs:
            for cls in classes:
                all_vals.append(r["stats"]["per_ticker"][t][split]["class_counts"].get(cls, 0))
        y_max = max(1, int(max(all_vals)))
        ax.set_ylim(0, y_max * 1.2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([_run_label(r) for r in runs], rotation=0)
    axes[0].legend(title="label", loc="upper right")
    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        print("Saved:", out_path)

def main():
    pkl_path = Path("prepared") / "sweep_tb_label_stats.pkl"
    payload = _load_pkl(pkl_path)

    for split in ("train", "val", "test"):
        plot_per_ticker_split_counts(
            payload,
            split=split,
            out_path=pkl_path.with_suffix(f".{split}.png"),
        )


if __name__ == "__main__":
    main()