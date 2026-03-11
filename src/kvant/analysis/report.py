"""
analysis.report — Pretty-print a comparison dict produced by compare_evaluations().

Public API
----------
print_comparison_report(comparison, top_n_tickers=5)

Tries ``rich`` first, then ``tabulate``, then falls back to plain pandas
``to_string()``.  All three paths produce the same information.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

# ---------------------------------------------------------------------------
# Optional pretty-printer backends
# ---------------------------------------------------------------------------
try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    _HAS_RICH = True
except ImportError:
    _RichConsole = None  # type: ignore[assignment,misc]
    _RichTable   = None  # type: ignore[assignment,misc]
    _HAS_RICH = False

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _tabulate = None  # type: ignore[assignment]
    _HAS_TABULATE = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def print_comparison_report(
    comparison: Dict[str, Any],
    top_n_tickers: int = 5,
) -> None:
    """
    Print a human-readable comparison report to stdout.

    Parameters
    ----------
    comparison    : dict returned by ``compare_evaluations()``.
    top_n_tickers : How many tickers to show in the Trade Stats section
                    (ranked by total profit, best first).
    """
    _section("RUN SUMMARY")
    _print_df(comparison.get("run_meta", pd.DataFrame()))

    _section("CLASSIFICATION METRICS")
    _print_df(comparison.get("classification_metrics", pd.DataFrame()))

    _section("RETURN STATS")
    _print_df(comparison.get("return_stats", pd.DataFrame()))

    _section("TRADE STATS  (top {} tickers by total profit)".format(top_n_tickers))
    ts_df = comparison.get("trade_stats", pd.DataFrame())
    if not ts_df.empty:
        # Sort each run individually and show the top-N tickers
        if "bruto_profit_pct/avg" in ts_df.columns:
            ts_top = (
                ts_df.sort_values("bruto_profit_pct/avg", ascending=False)
                     .groupby("run", sort=False)
                     .head(top_n_tickers)
                     .reset_index(drop=True)
            )
        else:
            ts_top = ts_df.head(top_n_tickers * max(1, ts_df["run"].nunique()))
        _print_df(ts_top)
    else:
        print("  (no trade stats available)")

    _section("EQUITY CURVE  (summary)")
    eq_df = comparison.get("equity_curves", pd.DataFrame())
    if not eq_df.empty:
        pnl_cols = [c for c in eq_df.columns if c.startswith("cumulative_pnl_pct")]
        summary_rows = []
        for col in pnl_cols:
            run_name = col.replace("cumulative_pnl_pct_", "")
            series   = eq_df[col].dropna()
            if series.empty:
                continue
            summary_rows.append({
                "run":      run_name,
                "n_trades": int(len(eq_df.dropna(subset=[col]))),
                "final_pnl_pct":  round(float(series.iloc[-1]),  4),
                "max_pnl_pct":    round(float(series.max()),      4),
                "min_pnl_pct":    round(float(series.min()),      4),
            })
        _print_df(pd.DataFrame(summary_rows))
    else:
        print("  (no equity curve data available)")

    _section("LABEL DISTRIBUTION")
    ld_df = comparison.get("label_distribution", pd.DataFrame())
    if not ld_df.empty:
        # Aggregate across tickers for a compact overview
        agg_cols = ["run", "label", "y_true_count", "y_pred_count"]
        agg_cols_present = [c for c in agg_cols if c in ld_df.columns]
        if agg_cols_present:
            agg = ld_df[agg_cols_present].groupby(
                [c for c in ["run", "label"] if c in agg_cols_present],
                as_index=False,
            ).sum()
            _print_df(agg)
    else:
        print("  (no label distribution data available)")

    _section("CONFUSION MATRICES")
    cm_dict = comparison.get("confusion_matrices", {})
    for run_label, cm_df in cm_dict.items():
        print(f"\n  [{run_label}]")
        _print_df(cm_df)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    bar = "─" * (len(title) + 4)
    print(f"\n┌{bar}┐")
    print(f"│  {title}  │")
    print(f"└{bar}┘")


def _print_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("  (empty)")
        return

    if _HAS_RICH:
        _rich_print(df)
    elif _HAS_TABULATE:
        print(_tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    else:
        print(df.to_string(index=False))


def _rich_print(df: pd.DataFrame) -> None:
    console = _RichConsole()
    table   = _RichTable(show_header=True, header_style="bold cyan")
    for col in df.columns:
        table.add_column(str(col), no_wrap=False)
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)
