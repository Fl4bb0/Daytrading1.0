from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from kvant.utils.time_utils import ensure_utc_sorted_index


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    fold_id: str
    mode: str
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    val_start: pd.Timestamp
    val_end_exclusive: pd.Timestamp
    test_start: pd.Timestamp
    test_end_exclusive: pd.Timestamp

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, pd.Timestamp):
                payload[key] = value.isoformat()
        return payload


def build_walk_forward_folds(
    ticker_dfs: Dict[str, pd.DataFrame],
    walk_cfg: dict,
) -> List[WalkForwardFold]:
    """Build month-based walk-forward folds from the available ticker history."""
    mode = str(walk_cfg.get("mode", "expanding")).strip().lower()
    train_span_months = int(walk_cfg.get("train_span_months", 6))
    val_span_months = int(walk_cfg.get("val_span_months", 1))
    test_span_months = int(walk_cfg.get("test_span_months", 1))
    step_span_months = int(walk_cfg.get("step_span_months", test_span_months))
    gap_days = int(walk_cfg.get("gap_days", 0))
    max_train_span_months = walk_cfg.get("max_train_span_months")
    max_train_span_months = None if max_train_span_months in (None, "", 0) else int(max_train_span_months)

    available_min, available_max = _infer_available_month_range(ticker_dfs)
    base_start = _month_start(walk_cfg.get("start_month")) if walk_cfg.get("start_month") else available_min
    horizon_end_exclusive = (
        _month_after(walk_cfg.get("end_month"))
        if walk_cfg.get("end_month")
        else available_max
    )

    folds: List[WalkForwardFold] = []
    fold_index = 0
    while True:
        if mode == "rolling":
            train_start = _add_months(base_start, fold_index * step_span_months)
            train_end_exclusive = _add_months(train_start, train_span_months)
        else:
            train_start = base_start
            train_end_exclusive = _add_months(base_start, train_span_months + fold_index * step_span_months)
            if max_train_span_months is not None:
                train_start = max(train_start, _add_months(train_end_exclusive, -max_train_span_months))

        val_start = train_end_exclusive + pd.Timedelta(days=gap_days)
        val_end_exclusive = val_start + pd.DateOffset(months=val_span_months)
        test_start = val_end_exclusive + pd.Timedelta(days=gap_days)
        test_end_exclusive = test_start + pd.DateOffset(months=test_span_months)

        if test_end_exclusive > horizon_end_exclusive:
            break

        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                fold_id=f"fold_{fold_index:03d}",
                mode=mode,
                train_start=train_start,
                train_end_exclusive=train_end_exclusive,
                val_start=val_start,
                val_end_exclusive=val_end_exclusive,
                test_start=test_start,
                test_end_exclusive=test_end_exclusive,
            )
        )
        fold_index += 1

    return folds


def split_ticker_dfs_for_fold(
    ticker_dfs: Dict[str, pd.DataFrame],
    fold: WalkForwardFold,
    *,
    min_train_rows_per_ticker: int = 1,
    min_val_rows_per_ticker: int = 1,
    min_test_rows_per_ticker: int = 1,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[dict]]:
    """Slice per-ticker train/val/test DataFrames for one walk-forward fold."""
    train_dfs: Dict[str, pd.DataFrame] = {}
    val_dfs: Dict[str, pd.DataFrame] = {}
    test_dfs: Dict[str, pd.DataFrame] = {}
    rows: List[dict] = []

    for sym, raw_df in ticker_dfs.items():
        df = ensure_utc_sorted_index(raw_df)
        train_df = _slice_df(df, fold.train_start, fold.train_end_exclusive)
        val_df = _slice_df(df, fold.val_start, fold.val_end_exclusive)
        test_df = _slice_df(df, fold.test_start, fold.test_end_exclusive)
        eligible = (
            len(train_df) >= int(min_train_rows_per_ticker)
            and len(val_df) >= int(min_val_rows_per_ticker)
            and len(test_df) >= int(min_test_rows_per_ticker)
        )
        if eligible:
            train_dfs[sym] = train_df
            val_dfs[sym] = val_df
            test_dfs[sym] = test_df

        rows.append(
            {
                "fold_id": fold.fold_id,
                "ticker": sym,
                "eligible": bool(eligible),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
            }
        )

    return train_dfs, val_dfs, test_dfs, rows


def describe_fold(fold: WalkForwardFold) -> dict:
    return fold.to_dict()


def _infer_available_month_range(ticker_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    for df in ticker_dfs.values():
        if df is None or len(df) == 0:
            continue
        norm = ensure_utc_sorted_index(df)
        starts.append(norm.index[0].to_period("M").to_timestamp())
        last = norm.index[-1].to_period("M").to_timestamp()
        ends.append(last + pd.DateOffset(months=1))

    if not starts or not ends:
        raise SystemExit("No timestamped data available to build walk-forward folds.")
    return min(starts), max(ends)


def _month_start(value: Optional[str]) -> pd.Timestamp:
    if value is None:
        raise ValueError("Expected month string, got None")
    ts = pd.Timestamp(f"{value}-01", tz="UTC").tz_localize(None)
    return ts.normalize()


def _month_after(value: str) -> pd.Timestamp:
    return _add_months(_month_start(value), 1)


def _add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=months)).normalize()


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    mask = (df.index >= start) & (df.index < end_exclusive)
    return df.loc[mask].copy()

