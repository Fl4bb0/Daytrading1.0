"""
scripts/run_walk_forward.py — Prepare, train, and evaluate walk-forward folds.

Usage
-----
  python scripts/run_walk_forward.py
  python scripts/run_walk_forward.py --config pipeline.toml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from kvant.experiment.prepare import PREPARED_DATA_ROOT, build_default_components, prepare_experiment
from kvant.pipeline_runtime import predict_experiment, train_experiment, train_meta_experiment
from kvant.utils.pipeline_config import load_pipeline_config
from kvant.utils.prepare_pipeline import apply_prepare_filters, load_pipeline_ticker_dfs
from kvant.utils.walk_forward import build_walk_forward_folds, describe_fold, split_ticker_dfs_for_fold
from kvant.walk_forward_reporting import write_walk_forward_aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the walk-forward train/test pipeline.")
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional walk-forward run id. Defaults to the configured value or a stable hash.",
    )
    args = parser.parse_args()

    pipeline_cfg, cfg_path = load_pipeline_config(args.config)
    walk_cfg = pipeline_cfg.get("walk_forward", {})
    if not bool(walk_cfg.get("enabled", False)):
        raise SystemExit("walk_forward.enabled must be true to run scripts/run_walk_forward.py")

    ticker_dfs = load_pipeline_ticker_dfs(pipeline_cfg)
    ticker_dfs = apply_prepare_filters(ticker_dfs, pipeline_cfg)
    folds = build_walk_forward_folds(ticker_dfs, walk_cfg)
    if not folds:
        raise SystemExit("No walk-forward folds fit within the configured data range.")

    prepared_root = Path(pipeline_cfg["paths"].get("prepared_root", str(PREPARED_DATA_ROOT)))
    run_id = args.run_id.strip() or str(walk_cfg.get("run_id", "")).strip() or _auto_run_id(pipeline_cfg)
    run_root = prepared_root / "walkforward" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Config    : {cfg_path}")
    print(f"Run root  : {run_root}")
    print(f"Folds     : {len(folds)}")

    manifest_path = run_root / "walk_forward_manifest.json"
    eligibility_rows: list[dict] = []
    completed_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for fold in folds:
        print(
            f"\n[{fold.fold_id}] "
            f"train={fold.train_start.date()}..{(fold.train_end_exclusive - pd.Timedelta(days=1)).date()} "
            f"val={fold.val_start.date()}..{(fold.val_end_exclusive - pd.Timedelta(days=1)).date()} "
            f"test={fold.test_start.date()}..{(fold.test_end_exclusive - pd.Timedelta(days=1)).date()}"
        )

        train_dfs, val_dfs, test_dfs, fold_eligibility = split_ticker_dfs_for_fold(
            ticker_dfs,
            fold,
            min_train_rows_per_ticker=int(walk_cfg.get("min_train_rows_per_ticker", 1)),
            min_val_rows_per_ticker=int(walk_cfg.get("min_val_rows_per_ticker", 1)),
            min_test_rows_per_ticker=int(walk_cfg.get("min_test_rows_per_ticker", 1)),
        )
        eligibility_rows.extend(fold_eligibility)
        eligible_tickers = sorted(train_dfs)
        print(f"  Eligible tickers: {len(eligible_tickers)}")
        if not train_dfs or not val_dfs or not test_dfs:
            skipped_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "reason": "empty_fold_after_ticker_filter",
                    "eligible_tickers": len(eligible_tickers),
                    **describe_fold(fold),
                }
            )
            print("  Skipping fold: empty train/val/test universe after eligibility filtering.")
            _write_manifest(manifest_path, cfg_path, run_id, folds, completed_rows, skipped_rows)
            continue

        sampler, fe, labeler, exp_cfg = build_default_components(
            interval=pipeline_cfg["data"].get("interval", "1m"),
            volatility_scaled_barrier=bool(pipeline_cfg["prepare"].get("volatility_scaled_barrier", True)),
            vol_scale_min=float(pipeline_cfg["prepare"].get("vol_scale_min", 0.5)),
            vol_scale_max=float(pipeline_cfg["prepare"].get("vol_scale_max", 2.0)),
            lookback_L=int(pipeline_cfg["prepare"].get("lookback", 20)),
            width_minutes=int(pipeline_cfg["prepare"].get("width_minutes", 20)),
            height_pct=float(pipeline_cfg["prepare"].get("height_pct", 0.5)),
            target_bars_per_day=int(pipeline_cfg["prepare"].get("target_bars_per_day", 195)),
            brokerage_fee=float(pipeline_cfg.get("trading", {}).get("brokerage_fee", 0.0008)),
        )

        prepared = prepare_experiment(
            out_root=run_root,
            cfg=exp_cfg,
            sampler=sampler,
            fe=fe,
            labeler=labeler,
            ticker_dfs_train=train_dfs,
            ticker_dfs_val=val_dfs,
            ticker_dfs_test=test_dfs,
            experiment_id=fold.fold_id,
        )
        fold_dir = prepared.exp_dir
        (fold_dir / "walk_forward_fold.json").write_text(
            json.dumps(
                {
                    **describe_fold(fold),
                    "eligible_tickers": eligible_tickers,
                },
                indent=2,
            )
        )

        train_artifacts = train_experiment(fold_dir, pipeline_cfg)
        if bool(pipeline_cfg.get("meta", {}).get("enabled", False)):
            train_meta_experiment(fold_dir, pipeline_cfg)
        eval_dir = predict_experiment(fold_dir, pipeline_cfg)

        return_stats_path = Path(eval_dir) / "return_stats.csv"
        class_metrics_path = Path(eval_dir) / "classification_metrics.csv"
        return_stats = pd.read_csv(return_stats_path).iloc[0].to_dict() if return_stats_path.exists() else {}
        overall_accuracy = ""
        if class_metrics_path.exists():
            class_df = pd.read_csv(class_metrics_path)
            overall_row = class_df[class_df["class"] == "overall"]
            if not overall_row.empty:
                overall_accuracy = float(overall_row.iloc[0]["accuracy"])

        completed_rows.append(
            {
                "fold_id": fold.fold_id,
                "fold_dir": str(fold_dir),
                "eval_dir": str(eval_dir),
                "eligible_tickers": len(eligible_tickers),
                "n_tickers_train": len(prepared.tickers_train),
                "n_tickers_val": len(prepared.tickers_val),
                "n_tickers_test": len(prepared.tickers_test),
                "overall_accuracy": overall_accuracy,
                "best_val_accuracy_max": max(a.best_val_accuracy for a in train_artifacts),
                "model_names": ",".join(a.model_name for a in train_artifacts),
                **describe_fold(fold),
                **return_stats,
            }
        )
        _write_manifest(manifest_path, cfg_path, run_id, folds, completed_rows, skipped_rows)

    pd.DataFrame(eligibility_rows).to_csv(run_root / "fold_ticker_eligibility.csv", index=False)

    if not completed_rows:
        raise SystemExit("No walk-forward folds completed successfully.")

    predict_cfg = pipeline_cfg["predict"]
    aggregate_dir = run_root / "aggregate" / f"{Path(completed_rows[0]['eval_dir']).name}"
    aggregate_path = write_walk_forward_aggregate(
        aggregate_dir=aggregate_dir,
        fold_rows=completed_rows,
        fee=float(pipeline_cfg.get("trading", {}).get("brokerage_fee", 0.0008)),
        execution_priority=str(predict_cfg.get("execution_priority", "model_confidence")),
        top_k_per_timestamp=None
        if predict_cfg.get("top_k_per_timestamp") in (None, "", 0)
        else int(predict_cfg.get("top_k_per_timestamp")),
        ticker_cooldown_minutes=int(predict_cfg.get("ticker_cooldown_minutes", 0)),
    )

    _write_manifest(
        manifest_path,
        cfg_path,
        run_id,
        folds,
        completed_rows,
        skipped_rows,
        aggregate_dir=aggregate_path,
    )
    (prepared_root / "last_walk_forward.txt").write_text(str(run_root))

    print(f"\nAggregate outputs: {aggregate_path}")
    print(f"Completed folds  : {len(completed_rows)}")
    print(f"Skipped folds    : {len(skipped_rows)}")


def _auto_run_id(pipeline_cfg: dict) -> str:
    payload = {
        "data": pipeline_cfg.get("data", {}),
        "prepare": pipeline_cfg.get("prepare", {}),
        "train": pipeline_cfg.get("train", {}),
        "predict": pipeline_cfg.get("predict", {}),
        "meta": pipeline_cfg.get("meta", {}),
        "ensemble": pipeline_cfg.get("ensemble", {}),
        "walk_forward": pipeline_cfg.get("walk_forward", {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"wf_{digest}"


def _write_manifest(
    manifest_path: Path,
    cfg_path: Path,
    run_id: str,
    folds,
    completed_rows: list[dict],
    skipped_rows: list[dict],
    *,
    aggregate_dir: Path | None = None,
) -> None:
    payload = {
        "config_path": str(cfg_path),
        "run_id": run_id,
        "n_folds_planned": int(len(folds)),
        "n_folds_completed": int(len(completed_rows)),
        "n_folds_skipped": int(len(skipped_rows)),
        "aggregate_dir": "" if aggregate_dir is None else str(aggregate_dir),
        "folds": [describe_fold(fold) for fold in folds],
        "completed": completed_rows,
        "skipped": skipped_rows,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
