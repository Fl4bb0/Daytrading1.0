"""
experiment.prepare — Main orchestration entry-point.

prepare_experiment():
  1. Fits sampler on TRAIN only.
  2. Fits feature engineer + labeler on sampled TRAIN data.
  3. Runs the full pipeline (sample → features → labels) per ticker
     over the concatenated train+val+test history.
  4. Saves all artifacts to disk via experiment.artifacts.
  5. Returns a PreparedExperimentManifest.

prepare_from_yahoo() is the top-level entry-point called by scripts/run_prepare.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tqdm

from kvant.experiment.config import ExperimentConfig
from kvant.experiment.artifacts import save_ticker_artifacts, json_default
from kvant.sampling.base import BarSampler
from kvant.features.feature_engineering import FeatureEngineer
from kvant.labeling.base import Labeler
from kvant.utils.time_utils import ensure_utc_sorted_index, as_dt64_utc_naive
from kvant.utils.index_utils import valid_target_positions, in_split

# Root directory where prepared experiments are written
PREPARED_DATA_ROOT = Path(__file__).resolve().parents[3] / "prepared"


@dataclass
class PreparedExperimentManifest:
    exp_dir: Path
    tickers_all: List[str]
    tickers_train: List[str]
    tickers_val: List[str]
    tickers_test: List[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_ts_utc_dt64(df: pd.DataFrame) -> Optional[np.datetime64]:
    df = ensure_utc_sorted_index(df)
    return as_dt64_utc_naive(df.index[0])


def _concat_nonempty(parts: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
    valid = [ensure_utc_sorted_index(p) for p in parts if p is not None and len(p) > 0]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, axis=0).sort_index()


def _counts_by_split(
    ts: np.ndarray,
    val_start: Optional[np.datetime64],
    test_start: Optional[np.datetime64],
) -> Dict[str, int]:
    out: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for tt in ts:
        for split in ("train", "val", "test"):
            if in_split(tt, split, val_start, test_start):
                out[split] += 1
                break
    return out


def _label_counts(y: np.ndarray) -> Dict[str, int]:
    if len(y) == 0:
        return {}
    u, c = np.unique(y, return_counts=True)
    return {str(int(uu)): int(cc) for uu, cc in zip(u, c)}


def _bars_per_day(ts: np.ndarray) -> float:
    if len(ts) == 0:
        return 0.0
    days = pd.Series(pd.to_datetime(ts)).dt.normalize()
    n_days = int(days.nunique())
    return float(len(ts) / n_days) if n_days > 0 else 0.0


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------

def prepare_experiment(
    out_root: Path,
    cfg: ExperimentConfig,
    sampler: BarSampler,
    fe: FeatureEngineer,
    labeler: Labeler,
    ticker_dfs_train: Dict[str, pd.DataFrame],
    ticker_dfs_val: Dict[str, pd.DataFrame],
    ticker_dfs_test: Dict[str, pd.DataFrame],
    experiment_id: Optional[str] = None,
) -> PreparedExperimentManifest:
    """
    Prepare a full experiment and persist all artifacts to disk.

    Splits are provided explicitly as dicts of DataFrames. For each ticker
    the full history (train+val+test) is concatenated, sampled, and processed
    so val/test can use the causal training history without leakage.
    """
    # ------------------------------------------------------------------
    # Directories + config
    # ------------------------------------------------------------------
    exp_id  = cfg.stable_id() if experiment_id is None else experiment_id
    exp_dir = Path(out_root) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    tickers_train = sorted(ticker_dfs_train)
    tickers_val   = sorted(ticker_dfs_val)
    tickers_test  = sorted(ticker_dfs_test)
    tickers_all   = sorted(set(tickers_train) | set(tickers_val) | set(tickers_test))

    (exp_dir / "tickers_all.json").write_text(json.dumps(tickers_all, indent=2))
    (exp_dir / "tickers_train.json").write_text(json.dumps(tickers_train, indent=2))
    (exp_dir / "tickers_val.json").write_text(json.dumps(tickers_val, indent=2))
    (exp_dir / "tickers_test.json").write_text(json.dumps(tickers_test, indent=2))

    ticker_id    = {t: i for i, t in enumerate(tickers_all)}
    tickers_root = exp_dir / "tickers"
    tickers_root.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Infer per-ticker split boundaries from the provided DataFrames
    # ------------------------------------------------------------------
    boundaries: Dict[str, tuple] = {}
    for t in tickers_all:
        val_start = (
            _first_ts_utc_dt64(ticker_dfs_val[t])
            if t in ticker_dfs_val and len(ticker_dfs_val[t]) > 0
            else None
        )
        test_start = (
            _first_ts_utc_dt64(ticker_dfs_test[t])
            if t in ticker_dfs_test and len(ticker_dfs_test[t]) > 0
            else None
        )
        boundaries[t] = (val_start, test_start)

    # ------------------------------------------------------------------
    # 1. Fit sampler on TRAIN only
    # ------------------------------------------------------------------
    sampler.fit(ticker_dfs_train)
    (exp_dir / "sampler_global_meta.json").write_text(
        json.dumps(sampler.get_global_meta(), indent=2, default=json_default)
    )
    (exp_dir / "sampler_per_ticker_meta.json").write_text(
        json.dumps(
            {t: sampler.get_ticker_meta(t) for t in tickers_all},
            indent=2, default=json_default,
        )
    )

    # ------------------------------------------------------------------
    # 2. Build sampled TRAIN corpus, then fit FE + labeler on it
    # ------------------------------------------------------------------
    sampled_train_parts: List[pd.DataFrame] = []
    for t in tickers_train:
        df = ticker_dfs_train.get(t)
        if df is None or len(df) == 0:
            continue
        ds = sampler.transform(ensure_utc_sorted_index(df), ticker=t)
        if ds is not None and len(ds) > 0:
            sampled_train_parts.append(ensure_utc_sorted_index(ds))

    if not sampled_train_parts:
        raise RuntimeError(
            "No sampled training rows available. "
            "Check that ticker_dfs_train is non-empty and the sampler is not too sparse."
        )
    df_fit = pd.concat(sampled_train_parts, axis=0).sort_index()
    fe.fit(df_fit)
    labeler.fit(df_fit)

    # ------------------------------------------------------------------
    # 3. Process each ticker on its full history (train + val + test)
    # ------------------------------------------------------------------
    valid_pos_by_ticker: Dict[str, np.ndarray] = {}
    density_rows: List[dict] = []

    for t in tqdm.tqdm(tickers_all, desc="Preparing tickers"):
        df_full = _concat_nonempty([
            ticker_dfs_train.get(t),
            ticker_dfs_val.get(t),
            ticker_dfs_test.get(t),
        ])
        if len(df_full) == 0:
            raise RuntimeError(f"Ticker {t!r} has no rows across any split.")

        val_start, test_start = boundaries[t]
        ts_raw = df_full.index.to_numpy()

        # Sample → features → labels
        df_s          = ensure_utc_sorted_index(sampler.transform(df_full, ticker=t))
        X, feat_names = fe.transform(df_s)
        y, y_meta     = labeler.transform(df_s)

        if len(X) != len(y):
            raise RuntimeError(f"Length mismatch for {t!r}: X={len(X)}, y={len(y)}")

        ts        = df_s.index.to_numpy()
        valid_pos = valid_target_positions(y, cfg.lookback_L)
        valid_pos_by_ticker[t] = valid_pos

        # Diagnostics
        n_raw, n_sampled = int(len(df_full)), int(len(df_s))
        retention = float(n_sampled / n_raw) if n_raw > 0 else 0.0

        density_row = {
            "ticker":                  t,
            "n_raw":                   n_raw,
            "n_sampled":               n_sampled,
            "retention_ratio":         retention,
            "bars_per_day_raw":        _bars_per_day(ts_raw),
            "bars_per_day_sampled":    _bars_per_day(ts),
            "raw_counts_by_split":     _counts_by_split(ts_raw, val_start, test_start),
            "sampled_counts_by_split": _counts_by_split(ts, val_start, test_start),
            "label_counts":            _label_counts(y),
            "label_counts_valid":      _label_counts(y[valid_pos]) if len(valid_pos) else {},
        }
        density_rows.append(density_row)

        membership = (
            (["train"] if t in ticker_dfs_train else [])
            + (["val"]   if t in ticker_dfs_val   else [])
            + (["test"]  if t in ticker_dfs_test  else [])
        )
        meta = {
            "ticker":          t,
            "membership":      membership,
            "feature_names":   feat_names,
            "sampler_name":    sampler.name,
            "n_valid_targets": int(len(valid_pos)),
            "val_start_ts":    None if val_start  is None else str(pd.Timestamp(val_start,  tz="UTC")),
            "test_start_ts":   None if test_start is None else str(pd.Timestamp(test_start, tz="UTC")),
            **density_row,
        }
        save_ticker_artifacts(tickers_root / t, X, y, ts, meta, label_metadata=y_meta)

    (exp_dir / "density_summary.json").write_text(
        json.dumps(density_rows, indent=2, default=json_default)
    )

    # ------------------------------------------------------------------
    # 4. Build and save train/val/test index arrays  (tid, position)
    # ------------------------------------------------------------------
    def _build_index(tickers: List[str], split: str) -> np.ndarray:
        rows = []
        for t in tickers:
            ts_t      = np.load(tickers_root / t / "timestamps.npy", mmap_mode="r")
            vp        = valid_pos_by_ticker[t]
            tid       = ticker_id[t]
            val_s, test_s = boundaries[t]
            for p in vp:
                if in_split(ts_t[int(p)], split, val_s, test_s):
                    rows.append((tid, int(p)))
        return np.asarray(rows, dtype=np.int32)

    index_train = _build_index(tickers_train, "train")
    index_val   = _build_index(tickers_val,   "val")
    index_test  = _build_index(tickers_test,  "test")

    np.save(exp_dir / "index_train.npy", index_train)
    np.save(exp_dir / "index_val.npy",   index_val)
    np.save(exp_dir / "index_test.npy",  index_test)

    print(f"\nPrepared experiment → {exp_dir}")
    print(f"  train : {len(index_train)} samples")
    print(f"  val   : {len(index_val)} samples")
    print(f"  test  : {len(index_test)} samples")

    return PreparedExperimentManifest(
        exp_dir=exp_dir,
        tickers_all=tickers_all,
        tickers_train=tickers_train,
        tickers_val=tickers_val,
        tickers_test=tickers_test,
    )


# ---------------------------------------------------------------------------
# Default component wiring
# ---------------------------------------------------------------------------

def build_default_components(
    interval: str = "1d",
    volatility_scaled_barrier: bool = True,
    vol_scale_min: float = 0.5,
    vol_scale_max: float = 2.0,
):
    """
    Return (sampler, fe, labeler, cfg) wired with sensible defaults,
    scaled to the given bar interval.
    """
    from kvant.sampling.cusum import TunedCUSUMBarSampler
    from kvant.features.feature_engineering import IntradayTA10Features, StandardizedFeatures
    from kvant.labeling.triple_barrier import TripleBarrierLabeler

    _interval_minutes = {
        "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
        "60m": 60, "1h": 60, "90m": 90, "1d": 390, "1wk": 1950,
    }
    bar_minutes    = _interval_minutes.get(interval, 1)
    width_minutes  = bar_minutes * 20
    target_bars_pd = max(1, round(390 / bar_minutes * 0.5))
    L, height_pct  = 20, 0.5

    sampler = TunedCUSUMBarSampler(
        target_bars_per_day=target_bars_pd,
        aggregate_ohlcv=True,
    )
    fe = StandardizedFeatures(
        base=IntradayTA10Features(volume_output="log1p", include_time_features=True)
    )
    vol_mode = "ticker_std" if volatility_scaled_barrier else "none"

    labeler = TripleBarrierLabeler(
        name=f"tb_w{width_minutes}_h{height_pct}pct_{vol_mode}",
        width_minutes=width_minutes,
        height=height_pct / 100,
        drop_time_exit=False,
        volatility_scale_mode=vol_mode,
        vol_scale_min=float(vol_scale_min),
        vol_scale_max=float(vol_scale_max),
    )
    exp_suffix = f"_VS{vol_scale_min:g}-{vol_scale_max:g}" if volatility_scaled_barrier else ""
    cfg = ExperimentConfig(
        experiment_name=f"tb_L{L}_w{width_minutes}_h{height_pct}_TBPD{target_bars_pd}{exp_suffix}",
        sampler=asdict(sampler),
        feature_engineer=fe.get_meta(),
        labeler=asdict(labeler),
        lookback_L=L,
    )
    return sampler, fe, labeler, cfg


# ---------------------------------------------------------------------------
# Yahoo Finance entry-point
# ---------------------------------------------------------------------------

def fetch_yahoo_splits(
    symbols: List[str],
    period: str = "6mo",
    interval: str = "1d",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Download data from Yahoo Finance and split each ticker chronologically
    into train / val / test DataFrames.

    Returns (ticker_dfs_train, ticker_dfs_val, ticker_dfs_test).
    """
    from kvant.kdata.retriever import YahooRetriever

    print(f"Fetching {symbols}  |  period={period}  interval={interval}")
    all_data = YahooRetriever(interval=interval, period=period).get_ticker_data(
        symbols, interval=interval, period=period
    )

    train_dfs: Dict[str, pd.DataFrame] = {}
    val_dfs:   Dict[str, pd.DataFrame] = {}
    test_dfs:  Dict[str, pd.DataFrame] = {}

    for sym, df in all_data.items():
        n       = len(df)
        n_test  = max(1, int(n * test_frac))
        n_val   = max(1, int(n * val_frac))
        n_train = n - n_val - n_test
        if n_train <= 0:
            print(f"  Skipping {sym}: only {n} bars — not enough for the requested split fractions.")
            continue
        train_dfs[sym] = df.iloc[:n_train].copy()
        val_dfs[sym]   = df.iloc[n_train: n_train + n_val].copy()
        test_dfs[sym]  = df.iloc[n_train + n_val:].copy()
        print(f"  {sym}: {n_train} train / {n_val} val / {n_test} test bars")

    if not train_dfs:
        raise RuntimeError("No tickers had enough data. Try a longer period or fewer splits.")

    return train_dfs, val_dfs, test_dfs


def prepare_from_yahoo(
    symbols: Optional[List[str]] = None,
    period: str = "6mo",
    interval: str = "1d",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> PreparedExperimentManifest:
    """
    Fetch data from Yahoo Finance and run the full preparation pipeline.

    Uses default components (CUSUM sampler, IntradayTA10 features with
    standardisation, triple-barrier labeler) scaled to the bar interval.
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    train_dfs, val_dfs, test_dfs = fetch_yahoo_splits(
        symbols, period=period, interval=interval,
        val_frac=val_frac, test_frac=test_frac,
    )

    sampler, fe, labeler, cfg = build_default_components(interval=interval)

    PREPARED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = prepare_experiment(
        out_root=PREPARED_DATA_ROOT,
        cfg=cfg,
        sampler=sampler,
        fe=fe,
        labeler=labeler,
        ticker_dfs_train=train_dfs,
        ticker_dfs_val=val_dfs,
        ticker_dfs_test=test_dfs,
        experiment_id=cfg.experiment_name,
    )
    (PREPARED_DATA_ROOT / "last_experiment.txt").write_text(manifest.exp_dir.name)
    return manifest
