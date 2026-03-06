# prepare_experiment.py
from kvant.ml_prepare_data.features.feature_engineering import IntradayTA10Features, StandardizedFeatures, \
    FeatureEngineer
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm

from kvant.ml_prepare_data.labelling.tripple_bar import Labeler, TripleBarrierLabeler
from kvant.ml_prepare_data.samplers.sampling import BarSampler
from kvant.ml_prepare_data.reporting import report_sampling_density
from kvant.ml_prepare_data.samplers.sampler_cumsum import TunedCUSUMBarSampler
from typing import Dict, Optional, List  # add Any, List
from kvant.kdata.hf_minute_data import get_ticker_data, DownloadedDatasetSplit, get_huggingface_top_5_small_splits
from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index

# ============================================================
# 3) Experiment config + stable id
# ============================================================
@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    sampler: dict
    feature_engineer: dict
    labeler: dict
    lookback_L: int

    def stable_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


# ============================================================
# 4) Utilities
# ============================================================

def valid_target_positions(labels: np.ndarray, lookback_L: int) -> np.ndarray:
    pos = np.arange(len(labels))
    return pos[(labels != -1) & (pos >= lookback_L)]

def _json_default(x):
    # Fallback serializer for json.dumps
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.datetime64,)):
        return str(pd.Timestamp(x))
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return str(x)


def save_label_metadata_jsonl(tdir: Path, metadata: List[Optional[dict]]) -> None:
    """
    Writes one JSON value per row: either `null` or an object.
    Aligned by position with features/labels/timestamps.
    """
    path = tdir / "label_metadata.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, default=_json_default))
            f.write("\n")

def save_ticker_artifacts(
    tdir: Path,
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    meta: dict,
    label_metadata: Optional[list[Optional[dict]]] = None,
) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    np.save(tdir / "features.npy", X.astype(np.float32))
    np.save(tdir / "labels.npy", y.astype(np.int8))
    np.save(tdir / "timestamps.npy", ts.astype("datetime64[ns]"))
    (tdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if label_metadata is not None:
        if len(label_metadata) != len(y):
            raise RuntimeError(f"label_metadata length {len(label_metadata)} != labels length {len(y)}")
        save_label_metadata_jsonl(tdir, label_metadata)

def _as_dt64_utc_naive(x) -> np.datetime64:
    """
    Convert x (pd.Timestamp/np.datetime64/etc.) to UTC-naive np.datetime64[ns].
    """
    if x is None:
        return None
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[ns]")
    if isinstance(x, pd.Timestamp):
        # convert tz-aware -> UTC, then drop tz
        if x.tz is not None:
            x = x.tz_convert("UTC").tz_localize(None)
        return x.to_datetime64().astype("datetime64[ns]")
    # last resort
    return np.datetime64(pd.Timestamp(x, tz="UTC").tz_localize(None)).astype("datetime64[ns]")

def _first_ts_utc_dt64(df: pd.DataFrame) -> np.datetime64:
    df = ensure_utc_sorted_index(df)
    return _as_dt64_utc_naive(df.index[0])


def _concat_nonempty(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts2 = [p for p in parts if p is not None and len(p) > 0]
    if not parts2:
        return pd.DataFrame()
    out = pd.concat([ensure_utc_sorted_index(p) for p in parts2], axis=0)
    # timestamps assumed strictly increasing per ticker; still keep sorted for safety
    out = out.sort_index()
    return out

def _in_split(tt, split: str, val_start, test_start) -> bool:
    tt = _as_dt64_utc_naive(tt)
    val_start = _as_dt64_utc_naive(val_start) if val_start is not None else None
    test_start = _as_dt64_utc_naive(test_start) if test_start is not None else None

    if split == "train":
        cut = val_start if val_start is not None else test_start
        return True if cut is None else (tt < cut)

    if split == "val":
        if val_start is None:
            return False
        if test_start is None:
            return tt >= val_start
        return (tt >= val_start) and (tt < test_start)

    if split == "test":
        if test_start is None:
            return False
        return tt >= test_start

    raise ValueError(split)


# ============================================================
# 5) Preparation Orchestrator
# ============================================================
@dataclass
class PreparedExperimentManifest:
    exp_dir: Path
    tickers_all: list[str]
    tickers_train: list[str]
    tickers_val: list[str]
    tickers_test: list[str]

def prepare_experiment(
    out_root: Path,
    cfg: ExperimentConfig,
    sampler: BarSampler,
    fe: FeatureEngineer,
    labeler: Labeler,
    ticker_dfs_train: Dict[str, pd.DataFrame],
    ticker_dfs_val: Dict[str, pd.DataFrame],
    ticker_dfs_test: Dict[str, pd.DataFrame],
    experiment_id: str = None,  # Provide a stable id of the experiment.
) -> PreparedExperimentManifest:
    """
    Key behavior:
      1) Splits are manual and always provided: train/val/test dicts.
      2) For each ticker, concatenate (train + val + test) first.
      3) Apply sampler + feature engineer + labeler on the concatenated series
         so val/test can use training history causally (no leakage).
      4) Then build train/val/test indices using per-ticker boundaries inferred
         from the first timestamp in val/test.

    Additional behavior in this version:
      - sampler.fit(...) is called explicitly on TRAIN ONLY (per-ticker tuning allowed).
      - sampler provides explicit metadata (global + per-ticker) that is persisted.
      - density + label distribution diagnostics are saved per ticker and as a global summary.
    """

    # -----------------------------
    # small local helpers
    # -----------------------------

    def _counts_by_split_for_ts(
        ts: np.ndarray,
        val_start: Optional[np.datetime64],
        test_start: Optional[np.datetime64],
    ) -> dict:
        out = {"train": 0, "val": 0, "test": 0}
        for tt in ts:
            if _in_split(tt, "train", val_start, test_start):
                out["train"] += 1
            elif _in_split(tt, "val", val_start, test_start):
                out["val"] += 1
            elif _in_split(tt, "test", val_start, test_start):
                out["test"] += 1
        return out

    def _bars_per_day(ts: np.ndarray) -> float:
        if ts is None or len(ts) == 0:
            return 0.0
        s = pd.to_datetime(ts)
        if getattr(s, "tz", None) is not None:
            s = s.tz_convert("UTC").tz_localize(None)
        days = pd.Series(s).dt.normalize()
        n_days = int(days.nunique())
        if n_days <= 0:
            return 0.0
        return float(len(ts) / n_days)

    def _label_counts(y: np.ndarray) -> dict:
        if y is None or len(y) == 0:
            return {}
        u, c = np.unique(y, return_counts=True)
        return {str(int(uu)): int(cc) for uu, cc in zip(u, c)}

    def _label_counts_by_split(
        y: np.ndarray,
        ts: np.ndarray,
        val_start: Optional[np.datetime64],
        test_start: Optional[np.datetime64],
        *,
        only_valid_positions: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Returns:
          {
            "train": {"-1":..., "0":..., ...},
            "val":   {...},
            "test":  {...}
          }
        """
        out = {"train": {}, "val": {}, "test": {}}

        if only_valid_positions is None:
            positions = range(len(y))
        else:
            positions = (int(p) for p in only_valid_positions)

        for p in positions:
            lab = int(y[p])
            tt = ts[p]
            if _in_split(tt, "train", val_start, test_start):
                bucket = "train"
            elif _in_split(tt, "val", val_start, test_start):
                bucket = "val"
            elif _in_split(tt, "test", val_start, test_start):
                bucket = "test"
            else:
                continue
            k = str(lab)
            out[bucket][k] = out[bucket].get(k, 0) + 1

        return out

    # -----------------------------
    # experiment id + dirs
    # -----------------------------
    exp_id = cfg.stable_id() if experiment_id is None else experiment_id
    exp_dir = out_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    tickers_train = sorted(ticker_dfs_train.keys())
    tickers_val = sorted(ticker_dfs_val.keys())
    tickers_test = sorted(ticker_dfs_test.keys())
    tickers_all = sorted(set(tickers_train) | set(tickers_val) | set(tickers_test))

    (exp_dir / "tickers_all.json").write_text(json.dumps(tickers_all, indent=2))
    (exp_dir / "tickers_train.json").write_text(json.dumps(tickers_train, indent=2))
    (exp_dir / "tickers_val.json").write_text(json.dumps(tickers_val, indent=2))
    (exp_dir / "tickers_test.json").write_text(json.dumps(tickers_test, indent=2))

    ticker_id = {t: i for i, t in enumerate(tickers_all)}
    tickers_root = exp_dir / "tickers"
    tickers_root.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Infer per-ticker boundaries from provided split dicts
    # --------------------------------------------------------
    boundaries: dict[str, tuple[Optional[np.datetime64], Optional[np.datetime64]]] = {}
    for t in tickers_all:
        val_start = (
            _first_ts_utc_dt64(ticker_dfs_val[t])
            if t in ticker_dfs_val and len(ticker_dfs_val[t])
            else None
        )
        test_start = (
            _first_ts_utc_dt64(ticker_dfs_test[t])
            if t in ticker_dfs_test and len(ticker_dfs_test[t])
            else None
        )
        boundaries[t] = (val_start, test_start)
        # --------------------------------------------------------
        # Fit on TRAIN ONLY
        #   - sampler: explicit per-ticker tuning allowed (no-op in IdentitySampler)
        #   - feature engineer: fit on *SAMPLED TRAIN* (paper: indicators computed after sampling)
        #   - labeler: no-op fit (but keep consistent: fit on sampled train too)
        # --------------------------------------------------------

        # 1) Tune sampler on TRAIN ONLY (per-ticker tuning handled internally).
    sampler.fit(ticker_dfs_train)

    # 2) Persist sampler metadata
    sampler_global_meta = sampler.get_global_meta()
    sampler_per_ticker_meta = {t: sampler.get_ticker_meta(t) for t in tickers_all}
    (exp_dir / "sampler_global_meta.json").write_text(
        json.dumps(sampler_global_meta, indent=2, default=_json_default)
    )
    (exp_dir / "sampler_per_ticker_meta.json").write_text(
        json.dumps(sampler_per_ticker_meta, indent=2, default=_json_default)
    )

    # 3) Build df_fit_sampled by sampling each ticker's TRAIN split,
    #    then concatenating. This is what you want if fe.fit() estimates
    #    scaling statistics (mean/std) or other global parameters.
    sampled_parts: list[pd.DataFrame] = []
    for t in tickers_train:
        dft = ticker_dfs_train.get(t)
        if dft is None or len(dft) == 0:
            continue
        dft = ensure_utc_sorted_index(dft)
        dft_s = sampler.transform(dft, ticker=t)
        if dft_s is not None and len(dft_s) > 0:
            sampled_parts.append(ensure_utc_sorted_index(dft_s))

    df_fit_sampled = pd.concat(sampled_parts, axis=0) if sampled_parts else pd.DataFrame()

    if len(df_fit_sampled) == 0:
        raise RuntimeError(
            "No sampled training rows available to fit feature engineer. "
            "This usually means your sampler is too sparse or train data is empty."
        )

    # 4) Fit FE + labeler on sampled train
    fe.fit(df_fit_sampled)
    labeler.fit(df_fit_sampled)
    # --------------------------------------------------------
    # Process each ticker on full history (train+val+test)
    # --------------------------------------------------------
    valid_pos_by_ticker: Dict[str, np.ndarray] = {}

    # global diagnostics accumulator
    density_summary_rows: list[dict] = []

    for t in tqdm.tqdm(tickers_all, desc="Preparing tickers"):
        df_full_raw = _concat_nonempty(
            [
                ticker_dfs_train.get(t),
                ticker_dfs_val.get(t),
                ticker_dfs_test.get(t),
            ]
        )
        df_full_raw.sort_index(inplace=True)
        assert df_full_raw.index.is_monotonic_increasing == 1

        if len(df_full_raw) == 0:
            raise RuntimeError(f"Ticker {t} has no rows across train/val/test.")

        val_start, test_start = boundaries[t]

        # Raw density (before sampling)
        ts_raw = df_full_raw.index.to_numpy()
        raw_counts_by_split = _counts_by_split_for_ts(ts_raw, val_start, test_start)

        # Sampled
        df1 = sampler.transform(df_full_raw, ticker=t)
        df1 = ensure_utc_sorted_index(df1)

        X, feat_names = fe.transform(df1)
        y, y_meta = labeler.transform(df1)

        if len(X) != len(y):
            raise RuntimeError(f"Length mismatch for {t}: features={len(X)} labels={len(y)}")

        ts = df1.index.to_numpy()
        valid_pos = valid_target_positions(y, cfg.lookback_L)
        valid_pos_by_ticker[t] = valid_pos

        # Sampled density
        sampled_counts_by_split = _counts_by_split_for_ts(ts, val_start, test_start)

        # label distributions (overall + valid targets)
        y_counts_all = _label_counts(y)
        y_counts_valid = _label_counts(y[valid_pos] if len(valid_pos) else np.asarray([], dtype=y.dtype))

        y_counts_all_by_split = _label_counts_by_split(y, ts, val_start, test_start, only_valid_positions=None)
        y_counts_valid_by_split = _label_counts_by_split(y, ts, val_start, test_start, only_valid_positions=valid_pos)

        # simple per-split valid target counts (sanity/debug)
        n_valid_train = 0
        n_valid_val = 0
        n_valid_test = 0
        for p in valid_pos:
            tt = ts[int(p)]
            if _in_split(tt, "train", val_start, test_start):
                n_valid_train += 1
            elif _in_split(tt, "val", val_start, test_start):
                n_valid_val += 1
            elif _in_split(tt, "test", val_start, test_start):
                n_valid_test += 1

        # Membership of ticker in the provided split dicts (not time membership).
        membership = []
        if t in ticker_dfs_train:
            membership.append("train")
        if t in ticker_dfs_val:
            membership.append("val")
        if t in ticker_dfs_test:
            membership.append("test")

        n_raw_full = int(len(df_full_raw))
        n_sampled_full = int(len(df1))
        retention = float(n_sampled_full / n_raw_full) if n_raw_full > 0 else 0.0

        density_row = {
            "ticker": t,
            "n_raw_full": n_raw_full,
            "n_sampled_full": n_sampled_full,
            "retention_ratio": retention,
            "bars_per_day_raw": _bars_per_day(ts_raw),
            "bars_per_day_sampled": _bars_per_day(ts),
            "raw_counts_by_split": raw_counts_by_split,
            "sampled_counts_by_split": sampled_counts_by_split,
            "sampler_ticker_meta": sampler.get_ticker_meta(t),
        }
        density_summary_rows.append(density_row)

        meta = {
            "ticker": t,
            "membership": membership,
            "feature_names": feat_names,
            "sampler_name": sampler.name,
            "sampler_global_meta": sampler_global_meta,
            "sampler_ticker_meta": sampler.get_ticker_meta(t),
            # density diagnostics
            "n_rows_raw_full": n_raw_full,
            "n_rows_sampled_full": n_sampled_full,
            "retention_ratio": retention,
            "bars_per_day_raw": density_row["bars_per_day_raw"],
            "bars_per_day_sampled": density_row["bars_per_day_sampled"],
            "raw_counts_by_split": raw_counts_by_split,
            "sampled_counts_by_split": sampled_counts_by_split,
            # labeling diagnostics
            "label_counts_all": y_counts_all,
            "label_counts_valid_targets": y_counts_valid,
            "label_counts_all_by_split": y_counts_all_by_split,
            "label_counts_valid_targets_by_split": y_counts_valid_by_split,
            # existing info
            "n_valid_targets_full": int(len(valid_pos)),
            "val_start_ts": None if val_start is None else str(pd.Timestamp(val_start, tz="UTC")),
            "test_start_ts": None if test_start is None else str(pd.Timestamp(test_start, tz="UTC")),
            "n_valid_train": int(n_valid_train),
            "n_valid_val": int(n_valid_val),
            "n_valid_test": int(n_valid_test),
        }

        save_ticker_artifacts(tickers_root / t, X, y, ts, meta, label_metadata=y_meta)

    # Persist global density summary
    (exp_dir / "density_summary.json").write_text(
        json.dumps(density_summary_rows, indent=2, default=_json_default)
    )

    # --------------------------------------------------------
    # Build indices for train/val/test using inferred boundaries
    # --------------------------------------------------------
    def build_index_for_tickers(tickers: list[str], split: str) -> np.ndarray:
        out = []
        for t in tickers:
            ts = np.load(tickers_root / t / "timestamps.npy", mmap_mode="r")
            valid_pos = valid_pos_by_ticker[t]
            tid = ticker_id[t]
            val_start, test_start = boundaries[t]

            for p in valid_pos:
                p = int(p)
                tt = ts[p]
                if _in_split(tt, split, val_start, test_start):
                    out.append((tid, p))

        return np.asarray(out, dtype=np.int32)

    index_train = build_index_for_tickers(tickers_train, "train")
    index_val = build_index_for_tickers(tickers_val, "val")
    index_test = build_index_for_tickers(tickers_test, "test")

    np.save(exp_dir / "index_train.npy", index_train)
    np.save(exp_dir / "index_val.npy", index_val)
    np.save(exp_dir / "index_test.npy", index_test)

    print("Prepared indices:")
    print("  train:", len(index_train))
    print("  val:", len(index_val))
    print("  test:", len(index_test))

    return PreparedExperimentManifest(
        exp_dir=exp_dir,
        tickers_all=tickers_all,
        tickers_train=tickers_train,
        tickers_val=tickers_val,
        tickers_test=tickers_test,
    )



def prepare_single_dataset(dataset_split : DownloadedDatasetSplit, sampler, feature_engineer, labeler, L=64):
    ticker_data_train, ticker_data_val, ticker_data_test = get_ticker_data(dataset_split)

    # sampler = IdentitySampler(subsample_every=1)
    # fe = OHLCVFeatures(cols=("open", "high", "low", "close", "volume"), log1p_volume=True)
    # labeler = TripleBarrierLabeler(name="tb_w60_h2pct", width_minutes=60, height=0.02, drop_time_exit_label=False)
    # L = 200
    cfg = ExperimentConfig(
        experiment_name="exp_minimal_sep_components",
        sampler=asdict(sampler),
        feature_engineer=asdict(feature_engineer),
        labeler=asdict(labeler),
        lookback_L=L,
    )
    out_root = Path("../ml_framework/prepared")
    prepared = prepare_experiment(
        out_root=out_root,
        cfg=cfg,
        sampler=sampler,
        fe=feature_engineer,
        labeler=labeler,
        ticker_dfs_train=ticker_data_train,
        ticker_dfs_val=ticker_data_val,
        ticker_dfs_test=ticker_data_test,
    )
    print("Experiment prepared at:", prepared.exp_dir)
    return prepared

# ============================================================
# 6) Minimal runnable main (plug in your data loader)
# ============================================================
def main():
    downloaded_splits = get_huggingface_top_5_small_splits()
    ticker_data_train, ticker_data_val, ticker_data_test = get_ticker_data(downloaded_splits[-1])

    TBPD = 50
    sampler = TunedCUSUMBarSampler(target_bars_per_day=TBPD, aggregate_ohlcv=True)
    # fe = OHLCVFeatures(cols=("open", "high", "low", "close", "volume"), log1p_volume=True)

    base_fe = IntradayTA10Features(
        volume_output="log1p",
        include_time_features=True,
        typical_bar_minutes=None,  # periods in bars (paper style)
        fillna_value=0.0,
    )
    fe = StandardizedFeatures(base=base_fe)

    # fe
    L, width, height_pct = 20, 60, 1.5
    label = f"sb_L_{L}_w{width}_h{height_pct}_TBPD{TBPD}"
    print(f"Writing to {label=}")
    labeler = TripleBarrierLabeler(
        name=label,
        width_minutes=width,
        height=height_pct / 100,
        drop_time_exit_label=False
    )

    cfg = ExperimentConfig(
        experiment_name="exp_minimal_sep_components",
        sampler=asdict(sampler),
        feature_engineer=asdict(fe),
        labeler=asdict(labeler),
        lookback_L=L,
    )

    from kvant.ml_prepare_data import prepared_data_root

    prepared = prepare_experiment(
        out_root=prepared_data_root,
        cfg=cfg,
        sampler=sampler,
        fe=fe,
        labeler=labeler,
        ticker_dfs_train=ticker_data_train,
        ticker_dfs_val=ticker_data_val,
        ticker_dfs_test=ticker_data_test,
        experiment_id=label
    )

    # NEW: post-step density report + histograms
    report_sampling_density(prepared.exp_dir, bins=60, print_table=True)
    print("Experiment prepared at:", prepared.exp_dir)
    import os
    os.path.basename(prepared.exp_dir)
    from kvant.ml_prepare_data import prepared_data_root
    with open( prepared_data_root / "last_experiment.txt", 'w') as f:
        f.write(name := os.path.basename( prepared.exp_dir) )
        print("Wrote name to", name)


if __name__ == "__main__":
    main()