from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from kvant.evaluation import build_prediction_frame, evaluate_experiment
from kvant.models.base import KvantModel
from kvant.training.trainer import TrainConfig
from kvant.utils.device import resolve_torch_device
from kvant.utils.ensemble import ensemble_slug, normalize_model_names
from kvant.utils.pipeline_config import list_from_config


@dataclass
class BenchmarkResult:
    benchmark_dir: Path
    summary_csv: Path
    equity_comparison_csv: Path
    figures_dir: Path
    run_dirs: dict[str, Path]


@dataclass
class _EvalRun:
    label: str
    eval_dir: Path
    strategy: str
    seed: Optional[int] = None


class RandomTradingModel(KvantModel):
    """Random BUY/SHORT/HOLD signal generator with deterministic probabilities."""

    def __init__(
        self,
        *,
        trade_probability: float,
        seed: int,
        name: Optional[str] = None,
        n_classes: int = 3,
    ) -> None:
        if trade_probability < 0.0 or trade_probability > 1.0:
            raise ValueError("trade_probability must be between 0 and 1")
        self.trade_probability = float(trade_probability)
        self.seed = int(seed)
        self.n_classes = int(n_classes)
        self._name = name or f"random_trading_seed_{self.seed}"
        self._cached_n: Optional[int] = None
        self._cached_pred: Optional[np.ndarray] = None
        self._cached_proba: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):  # pragma: no cover - inference-only
        raise NotImplementedError("RandomTradingModel is inference-only")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_cache(len(X))
        assert self._cached_pred is not None
        return self._cached_pred.copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_cache(len(X))
        assert self._cached_proba is not None
        return self._cached_proba.copy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "random_trading_config.json").write_text(
            json.dumps(
                {
                    "type": "RandomTradingModel",
                    "name": self.name,
                    "trade_probability": self.trade_probability,
                    "seed": self.seed,
                    "n_classes": self.n_classes,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "RandomTradingModel":
        payload = json.loads((Path(path) / "random_trading_config.json").read_text())
        return cls(
            trade_probability=float(payload["trade_probability"]),
            seed=int(payload["seed"]),
            name=str(payload.get("name") or ""),
            n_classes=int(payload.get("n_classes", 3)),
        )

    def _ensure_cache(self, n: int) -> None:
        if self._cached_n == n and self._cached_pred is not None and self._cached_proba is not None:
            return

        rng = np.random.default_rng(self.seed)
        pred = np.full(n, 1, dtype=np.int64)
        directional = rng.random(n) < self.trade_probability
        sides = rng.choice(np.asarray([0, 2], dtype=np.int64), size=int(directional.sum()))
        pred[directional] = sides

        proba = np.zeros((n, self.n_classes), dtype=np.float32)
        proba[:, 1] = 1.0
        directional_idx = np.flatnonzero(directional)
        if len(directional_idx) > 0:
            side_conf = rng.uniform(0.8, 1.0, size=len(directional_idx)).astype(np.float32)
            for row_idx, conf in zip(directional_idx, side_conf):
                label = int(pred[row_idx])
                proba[row_idx, :] = (1.0 - conf) / max(self.n_classes - 1, 1)
                proba[row_idx, label] = conf

        self._cached_n = n
        self._cached_pred = pred
        self._cached_proba = proba


def run_benchmark(
    cfg: dict[str, Any],
    *,
    benchmark_id: Optional[str] = None,
    random_seeds: Optional[int] = None,
    train_shallow: bool = True,
    make_plots: bool = True,
) -> BenchmarkResult:
    paths_cfg = cfg["paths"]
    predict_cfg = cfg["predict"]
    train_cfg = cfg["train"]
    meta_cfg = cfg.get("meta", {})
    benchmark_cfg = cfg.get("benchmark", {})
    brokerage_fee = float(cfg.get("trading", {}).get("brokerage_fee", 0.0008))

    prepared_root = Path(paths_cfg.get("prepared_root", "prepared"))
    checkpoints_root = Path(paths_cfg.get("checkpoints_root", "checkpoints"))
    exp_id = _resolve_experiment_id(prepared_root, str(predict_cfg.get("experiment_id", "last")))
    exp_dir = prepared_root / exp_id
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    split = str(predict_cfg.get("split", "test"))
    requested_tickers = list_from_config(predict_cfg.get("tickers")) or None
    required_buy_probability = float(predict_cfg.get("required_buy_probability", 0.0))
    required_sell_probability = float(predict_cfg.get("required_sell_probability", 0.0))
    allow_short = bool(predict_cfg.get("allow_short", True))
    top_k_raw = predict_cfg.get("top_k_per_timestamp")
    top_k_per_timestamp = None if top_k_raw in (None, "", 0) else int(top_k_raw)
    ticker_cooldown_minutes = int(predict_cfg.get("ticker_cooldown_minutes", 0))
    execution_priority = str(predict_cfg.get("execution_priority", "model_confidence"))

    meta_train_split = str(meta_cfg.get("train_split", "val"))
    meta_shrinkage_k = float(meta_cfg.get("shrinkage_k", 10.0))
    meta_alpha = float(meta_cfg.get("alpha", 1.0))
    meta_min_score_buy = _optional_float(meta_cfg.get("min_score_buy"))
    meta_min_score_short = _optional_float(meta_cfg.get("min_score_short"))

    random_count = int(
        random_seeds
        if random_seeds is not None
        else benchmark_cfg.get("random_seeds", 50)
    )
    if random_count <= 0:
        raise SystemExit("random_seeds must be > 0")
    random_seed_start = int(benchmark_cfg.get("random_seed_start", 0))
    random_fallback_trade_probability = float(
        benchmark_cfg.get("random_fallback_trade_probability", 0.03)
    )

    single_model_name = str(benchmark_cfg.get("single_model", predict_cfg.get("model", "resnls")))
    shallow_model_name = "shallow_cnn"
    shallow_epochs = int(benchmark_cfg.get("shallow_epochs", min(int(train_cfg.get("epochs", 50)), 20)))
    shallow_patience = int(benchmark_cfg.get("shallow_patience", min(int(train_cfg.get("patience", 10)), 5)))
    try:
        device = resolve_torch_device(str(train_cfg.get("device", "auto")))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    ensemble_model_names = normalize_model_names(cfg.get("ensemble", {}).get("models"))
    if not ensemble_model_names:
        raise SystemExit("benchmark requires ensemble.models to define the council members")

    run_id = benchmark_id or benchmark_cfg.get("benchmark_id") or f"benchmark_{split}"
    benchmark_dir = exp_dir / "benchmark" / str(run_id)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = benchmark_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    from kvant.models import MODEL_REGISTRY

    for model_name in [single_model_name, shallow_model_name, *ensemble_model_names]:
        if model_name not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown benchmark model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    runs: list[_EvalRun] = []

    # 1. Council + meta. This runs first because random defaults to matching
    # the council's post-meta directional signal rate.
    council_name = ensemble_slug(ensemble_model_names)
    council_model, council_path, council_cls = _load_ensemble(
        model_names=ensemble_model_names,
        checkpoints_root=checkpoints_root,
        exp_id=exp_id,
    )
    council_meta, council_meta_path, council_history = _load_or_train_meta_model(
        exp_dir=exp_dir,
        checkpoints_root=checkpoints_root,
        active_model_name=council_name,
        model=council_model,
        model_path=council_path,
        model_cls=council_cls,
        meta_train_split=meta_train_split,
        current_split=split,
        tickers=requested_tickers,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        alpha=meta_alpha,
        shrinkage_k=meta_shrinkage_k,
    )
    council_dir = benchmark_dir / "council_meta"
    evaluate_experiment(
        exp_dir=exp_dir,
        model_path=council_path,
        model_cls=council_cls,
        out_dir=council_dir,
        split=split,
        tickers=requested_tickers,
        fee=brokerage_fee,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        allow_short=allow_short,
        execution_priority=execution_priority,
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
        model=council_model,
        meta_model=council_meta,
        meta_model_path=council_meta_path,
        meta_history_pred_df=council_history,
        meta_shrinkage_k=meta_shrinkage_k,
        meta_train_split=meta_train_split,
        meta_min_score_buy=meta_min_score_buy,
        meta_min_score_short=meta_min_score_short,
    )
    runs.append(_EvalRun("council_meta", council_dir, "council_meta"))

    council_pred = pd.read_csv(council_dir / "predictions.csv")
    default_trade_probability = float(council_pred["y_pred"].isin([0, 2]).mean())
    random_trade_probability = _optional_float(benchmark_cfg.get("random_trade_probability"))
    if random_trade_probability is None:
        random_trade_probability = default_trade_probability
        if random_trade_probability <= 0.0:
            random_trade_probability = random_fallback_trade_probability

    # 2. Single model + meta.
    single_model, single_path, single_cls = _load_checkpoint_model(
        model_name=single_model_name,
        checkpoints_root=checkpoints_root,
        exp_id=exp_id,
    )
    single_meta, single_meta_path, single_history = _load_or_train_meta_model(
        exp_dir=exp_dir,
        checkpoints_root=checkpoints_root,
        active_model_name=single_model_name,
        model=single_model,
        model_path=single_path,
        model_cls=single_cls,
        meta_train_split=meta_train_split,
        current_split=split,
        tickers=requested_tickers,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        alpha=meta_alpha,
        shrinkage_k=meta_shrinkage_k,
    )
    single_dir = benchmark_dir / f"single_{single_model_name}_meta"
    evaluate_experiment(
        exp_dir=exp_dir,
        model_path=single_path,
        model_cls=single_cls,
        out_dir=single_dir,
        split=split,
        tickers=requested_tickers,
        fee=brokerage_fee,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        allow_short=allow_short,
        execution_priority=execution_priority,
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
        model=single_model,
        meta_model=single_meta,
        meta_model_path=single_meta_path,
        meta_history_pred_df=single_history,
        meta_shrinkage_k=meta_shrinkage_k,
        meta_train_split=meta_train_split,
        meta_min_score_buy=meta_min_score_buy,
        meta_min_score_short=meta_min_score_short,
    )
    runs.append(_EvalRun(f"single_{single_model_name}_meta", single_dir, "single_meta"))

    # 3. One-layer CNN weak learned baseline, without meta.
    shallow_path = checkpoints_root / exp_id / shallow_model_name
    if not (shallow_path / "weights.pt").exists():
        if not train_shallow:
            raise SystemExit(
                f"No shallow CNN checkpoint found at {shallow_path}/weights.pt. "
                "Run benchmark with train_shallow enabled or train the model first."
            )
        _train_model_checkpoint(
            exp_dir=exp_dir,
            checkpoint_dir=shallow_path,
            model_name=shallow_model_name,
            device=device,
            epochs=shallow_epochs,
            batch_size=int(train_cfg.get("batch_size", 256)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            patience=shallow_patience,
        )
    shallow_model, _, shallow_cls = _load_checkpoint_model(
        model_name=shallow_model_name,
        checkpoints_root=checkpoints_root,
        exp_id=exp_id,
    )
    shallow_dir = benchmark_dir / "one_layer_cnn"
    evaluate_experiment(
        exp_dir=exp_dir,
        model_path=shallow_path,
        model_cls=shallow_cls,
        out_dir=shallow_dir,
        split=split,
        tickers=requested_tickers,
        fee=brokerage_fee,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        allow_short=allow_short,
        execution_priority=_non_meta_execution_priority(execution_priority),
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
        model=shallow_model,
    )
    runs.append(_EvalRun("one_layer_cnn", shallow_dir, "one_layer_cnn"))

    # 4. Random trading baseline, repeated across seeds.
    random_runs_root = benchmark_dir / ".random_runs"
    random_runs_root.mkdir(exist_ok=True)
    random_model_root = benchmark_dir / ".random_models"
    random_model_root.mkdir(exist_ok=True)
    for i in range(random_count):
        seed = random_seed_start + i
        random_model = RandomTradingModel(
            trade_probability=random_trade_probability,
            seed=seed,
            name=f"random_trading_seed_{i:03d}",
        )
        random_path = random_model_root / f"seed_{i:03d}"
        random_model.save(random_path)
        random_dir = random_runs_root / f"random_seed_{i:03d}"
        evaluate_experiment(
            exp_dir=exp_dir,
            model_path=random_path,
            model_cls=RandomTradingModel,
            out_dir=random_dir,
            split=split,
            tickers=requested_tickers,
            fee=brokerage_fee,
            required_buy_probability=required_buy_probability,
            required_sell_probability=required_sell_probability,
            allow_short=allow_short,
            execution_priority="model_confidence",
            top_k_per_timestamp=top_k_per_timestamp,
            ticker_cooldown_minutes=ticker_cooldown_minutes,
            model=random_model,
        )
        runs.append(_EvalRun(f"random_seed_{i:03d}", random_dir, "random", seed=seed))

    seed_summary = _build_seed_summary(runs)
    seed_summary.to_csv(benchmark_dir / "random_runs.csv", index=False)
    summary = _build_strategy_summary(seed_summary)
    summary.to_csv(benchmark_dir / "summary.csv", index=False)

    equity = _build_equity_comparison(
        runs=runs,
        store_dir=Path(paths_cfg.get("store", "data/1m")),
    )
    equity.to_csv(benchmark_dir / "equity_comparison.csv", index=False)

    _write_benchmark_config(
        benchmark_dir / "benchmark_config.json",
        cfg=cfg,
        exp_id=exp_id,
        split=split,
        benchmark_id=str(run_id),
        random_trade_probability=random_trade_probability,
        default_trade_probability=default_trade_probability,
        random_seeds=random_count,
    )

    if make_plots:
        _plot_benchmark_outputs(
            summary=summary,
            seed_summary=seed_summary,
            equity=equity,
            figures_dir=figures_dir,
        )

    return BenchmarkResult(
        benchmark_dir=benchmark_dir.resolve(),
        summary_csv=(benchmark_dir / "summary.csv").resolve(),
        equity_comparison_csv=(benchmark_dir / "equity_comparison.csv").resolve(),
        figures_dir=figures_dir.resolve(),
        run_dirs={run.label: run.eval_dir.resolve() for run in runs},
    )


def _resolve_experiment_id(prepared_root: Path, exp_id: str) -> str:
    if exp_id != "last":
        return exp_id
    last_file = prepared_root / "last_experiment.txt"
    if last_file.exists():
        return last_file.read_text().strip()

    # Fallback: use the latest completed walk-forward fold when available.
    wf_last_file = prepared_root / "last_walk_forward.txt"
    if wf_last_file.exists():
        run_root = Path(wf_last_file.read_text().strip())
        manifest_path = run_root / "walk_forward_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}
            completed = manifest.get("completed", [])
            if isinstance(completed, list) and completed:
                def _sort_key(row: dict) -> tuple[int, str]:
                    try:
                        idx = int(row.get("fold_index", -1))
                    except Exception:
                        idx = -1
                    return idx, str(row.get("fold_id", ""))

                latest = sorted(completed, key=_sort_key)[-1]
                fold_dir = str(latest.get("fold_dir", "")).strip()
                if fold_dir:
                    fold_path = Path(fold_dir)
                    try:
                        rel = fold_path.resolve().relative_to(prepared_root.resolve())
                        print(
                            "[benchmark] last_experiment.txt missing; "
                            f"falling back to latest walk-forward fold: {rel}"
                        )
                        return str(rel)
                    except ValueError:
                        pass

    raise SystemExit(
        f"No last_experiment.txt found in {prepared_root}, and no usable walk-forward fallback was found."
    )


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _non_meta_execution_priority(execution_priority: str) -> str:
    return "model_confidence" if execution_priority == "meta_score" else execution_priority


def _load_checkpoint_model(
    *,
    model_name: str,
    checkpoints_root: Path,
    exp_id: str,
) -> tuple[KvantModel, Path, type[KvantModel]]:
    from kvant.models import MODEL_REGISTRY

    model_cls = MODEL_REGISTRY[model_name]
    checkpoint = checkpoints_root / exp_id / model_name
    if not (checkpoint / "weights.pt").exists():
        raise SystemExit(
            f"No checkpoint found at {checkpoint}/weights.pt. "
            f"Train {model_name!r} first or let the benchmark train supported baselines."
        )
    return model_cls.load(checkpoint), checkpoint, model_cls


def _load_ensemble(
    *,
    model_names: list[str],
    checkpoints_root: Path,
    exp_id: str,
) -> tuple[KvantModel, Path, type[KvantModel]]:
    from kvant.models import MODEL_REGISTRY
    from kvant.models.ensemble import AveragingEnsembleModel

    if "conv3d" in model_names:
        raise SystemExit("conv3d cannot be used in the current council benchmark")

    members: list[KvantModel] = []
    member_paths: list[Path] = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown council member {model_name!r}")
        checkpoint = checkpoints_root / exp_id / model_name
        if not (checkpoint / "weights.pt").exists():
            raise SystemExit(f"No checkpoint found for council member {model_name}: {checkpoint}/weights.pt")
        members.append(MODEL_REGISTRY[model_name].load(checkpoint))
        member_paths.append(checkpoint)

    name = ensemble_slug(model_names)
    model = AveragingEnsembleModel(
        members,
        member_names=model_names,
        member_paths=member_paths,
        name=name,
    )
    model_path = checkpoints_root / exp_id / name
    model.save(model_path)
    return model, model_path, type(members[0])


def _load_or_train_meta_model(
    *,
    exp_dir: Path,
    checkpoints_root: Path,
    active_model_name: str,
    model: KvantModel,
    model_path: Path,
    model_cls: type[KvantModel],
    meta_train_split: str,
    current_split: str,
    tickers: Optional[list[str]],
    required_buy_probability: float,
    required_sell_probability: float,
    alpha: float,
    shrinkage_k: float,
) -> tuple[Any, Path, Optional[pd.DataFrame]]:
    from kvant.meta import RidgeMetaModel, build_meta_training_frame

    meta_path = checkpoints_root / exp_dir.name / active_model_name / "meta"
    if (meta_path / "model.pkl").exists():
        meta_model = RidgeMetaModel.load(meta_path)
    else:
        pred_df = build_prediction_frame(
            exp_dir=exp_dir,
            model_path=model_path,
            model_cls=model_cls,
            split=meta_train_split,
            tickers=tickers,
            required_buy_probability=required_buy_probability,
            required_sell_probability=required_sell_probability,
            model=model,
        )
        train_df = build_meta_training_frame(pred_df, shrinkage_k=shrinkage_k)
        if train_df.empty:
            raise SystemExit(
                f"No valid directional rows were available to train meta model for {active_model_name}."
            )
        meta_model = RidgeMetaModel(alpha=alpha)
        metrics = meta_model.fit(train_df)
        meta_model.save(meta_path)
        (meta_path / "training_metrics.json").write_text(
            json.dumps(
                {
                    "base_model_name": active_model_name,
                    "train_split": meta_train_split,
                    "n_rows_total": int(len(pred_df)),
                    "n_rows_trainable": int(len(train_df)),
                    "metrics": metrics,
                },
                indent=2,
            )
        )

    history = None
    if current_split != meta_train_split:
        history = build_prediction_frame(
            exp_dir=exp_dir,
            model_path=model_path,
            model_cls=model_cls,
            split=meta_train_split,
            tickers=tickers,
            required_buy_probability=required_buy_probability,
            required_sell_probability=required_sell_probability,
            model=model,
        )
    return meta_model, meta_path, history


def _train_model_checkpoint(
    *,
    exp_dir: Path,
    checkpoint_dir: Path,
    model_name: str,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> None:
    import inspect

    from kvant.models import MODEL_REGISTRY
    from kvant.training.pytorch_trainer import PytorchTrainer
    from kvant.utils.split_loader import load_split_from_index

    cfg_data = json.loads((exp_dir / "config.json").read_text())
    lookback_L = int(cfg_data["lookback_L"])
    index_train = np.load(exp_dir / "index_train.npy")
    index_val = np.load(exp_dir / "index_val.npy")

    loaded_train = load_split_from_index(
        exp_dir=exp_dir,
        index=index_train,
        lookback_L=lookback_L,
        include_timestamps=False,
        include_metadata=False,
    )
    loaded_val = load_split_from_index(
        exp_dir=exp_dir,
        index=index_val,
        lookback_L=lookback_L,
        include_timestamps=False,
        include_metadata=False,
    )

    model_cls = MODEL_REGISTRY[model_name]
    kwargs = {
        "n_features": loaded_train.X.shape[1],
        "n_classes": int(loaded_train.y.max()) + 1,
        "device": device,
        "seq_len": lookback_L,
    }
    accepted = inspect.signature(model_cls.__init__).parameters
    model = model_cls(**{key: value for key, value in kwargs.items() if key in accepted})
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=patience,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = PytorchTrainer(model, train_cfg)
    history = trainer.fit(loaded_train.X, loaded_train.y, loaded_val.X, loaded_val.y)
    model.save(checkpoint_dir)
    (checkpoint_dir / "training_history.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "patience": patience,
                "device": device,
                "best_epoch": history.get("best_epoch"),
                "best_val_accuracy": history.get("best_val_accuracy"),
            },
            indent=2,
        )
    )


def _build_seed_summary(runs: list[_EvalRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(_summarize_eval_run(run))
    return pd.DataFrame(rows)


def _summarize_eval_run(run: _EvalRun) -> dict[str, Any]:
    ret = pd.read_csv(run.eval_dir / "return_stats.csv")
    eq = pd.read_csv(run.eval_dir / "equity_curve.csv")
    pred = pd.read_csv(run.eval_dir / "predictions.csv")
    meta = pd.read_csv(run.eval_dir / "run_meta.csv")

    row: dict[str, Any] = {
        "run": run.label,
        "strategy": run.strategy,
        "seed": "" if run.seed is None else int(run.seed),
        "eval_dir": str(run.eval_dir),
        "n_samples": int(len(pred)),
        "n_pred_directional": int(pred["y_pred"].isin([0, 2]).sum()),
        "pred_directional_rate": float(pred["y_pred"].isin([0, 2]).mean()) if len(pred) else 0.0,
    }

    if len(ret) > 0:
        for key in [
            "accuracy_call_put/avg",
            "bruto_profit_pct/avg",
            "directional_accuracy",
            "directional_opposite_rate",
        ]:
            if key in ret.columns:
                row[key] = float(ret[key].iloc[0])

    if len(meta) > 0:
        for key in [
            "model_name",
            "execution_priority",
            "top_k_per_timestamp",
            "ticker_cooldown_minutes",
            "n_thresholded_to_hold",
            "n_meta_thresholded_to_hold",
            "allow_short",
            "n_short_blocked_by_policy",
            "meta_enabled",
        ]:
            if key in meta.columns:
                row[key] = meta[key].iloc[0]

    if eq.empty:
        row.update(
            {
                "candidate_trades": 0,
                "executed_trades": 0,
                "skipped_trades": 0,
                "final_trade_pnl_pct": 0.0,
                "final_portfolio_gross_pct": 0.0,
                "final_portfolio_net_pct": 0.0,
                "max_drawdown_net_pct": 0.0,
            }
        )
        return row

    skipped = eq["skipped"].astype(bool) if "skipped" in eq.columns else pd.Series(False, index=eq.index)
    row["candidate_trades"] = int(len(eq))
    row["executed_trades"] = int((~skipped).sum())
    row["skipped_trades"] = int(skipped.sum())
    row["final_trade_pnl_pct"] = _last_float(eq, "cumulative_pnl_pct")
    row["final_portfolio_gross_pct"] = _last_float(eq, "cumulative_portfolio_pnl_pct")
    row["final_portfolio_net_pct"] = _last_float(eq, "cumulative_portfolio_pnl_net_pct")
    row["max_drawdown_net_pct"] = _max_drawdown(eq.get("cumulative_portfolio_pnl_net_pct"))
    return row


def _last_float(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df[column].dropna().empty:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").dropna().iloc[-1])


def _max_drawdown(series: Optional[pd.Series]) -> float:
    if series is None:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    drawdown = values - peak
    return float(drawdown.min())


def _build_strategy_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    non_random = seed_summary[seed_summary["strategy"] != "random"].copy()
    random = seed_summary[seed_summary["strategy"] == "random"].copy()
    rows: list[dict[str, Any]] = []

    for _, row in non_random.iterrows():
        out = row.to_dict()
        out["n_runs"] = 1
        out["final_portfolio_net_pct_p05"] = out.get("final_portfolio_net_pct")
        out["final_portfolio_net_pct_p95"] = out.get("final_portfolio_net_pct")
        rows.append(out)

    if not random.empty:
        numeric_cols = [
            "n_samples",
            "n_pred_directional",
            "pred_directional_rate",
            "accuracy_call_put/avg",
            "bruto_profit_pct/avg",
            "directional_accuracy",
            "directional_opposite_rate",
            "candidate_trades",
            "executed_trades",
            "skipped_trades",
            "final_trade_pnl_pct",
            "final_portfolio_gross_pct",
            "final_portfolio_net_pct",
            "max_drawdown_net_pct",
        ]
        out = {
            "run": "random_trading_mean",
            "strategy": "random",
            "seed": "",
            "eval_dir": "",
            "n_runs": int(len(random)),
        }
        for col in numeric_cols:
            if col in random.columns:
                out[col] = float(pd.to_numeric(random[col], errors="coerce").mean())
        final_net = pd.to_numeric(random["final_portfolio_net_pct"], errors="coerce").dropna()
        if not final_net.empty:
            out["final_portfolio_net_pct_p05"] = float(final_net.quantile(0.05))
            out["final_portfolio_net_pct_p95"] = float(final_net.quantile(0.95))
        rows.append(out)

    preferred = [
        "council_meta",
        "single_meta",
        "one_layer_cnn",
        "random",
    ]
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["_order"] = summary["strategy"].map({v: i for i, v in enumerate(preferred)}).fillna(99)
    return summary.sort_values(["_order", "run"]).drop(columns=["_order"]).reset_index(drop=True)


def _build_equity_comparison(
    *,
    runs: list[_EvalRun],
    store_dir: Path,
) -> pd.DataFrame:
    curves: dict[str, pd.Series] = {}
    empty_curve_names: list[str] = []
    random_curves: list[pd.Series] = []
    saw_random = False
    all_prediction_times: list[pd.Timestamp] = []
    tickers: set[str] = set()

    for run in runs:
        pred = pd.read_csv(run.eval_dir / "predictions.csv")
        if "timestamp" in pred.columns:
            ts = pd.to_datetime(pred["timestamp"], errors="coerce", utc=True).dropna()
            all_prediction_times.extend(ts.tolist())
        if "ticker" in pred.columns:
            tickers.update(str(v) for v in pred["ticker"].dropna().unique())

        eq = pd.read_csv(run.eval_dir / "equity_curve.csv")
        if eq.empty or "timestamp" not in eq.columns or "cumulative_portfolio_pnl_net_pct" not in eq.columns:
            if run.strategy == "random":
                saw_random = True
            else:
                empty_curve_names.append(f"{run.label}_net_pct")
            continue
        series = _equity_series(eq, "cumulative_portfolio_pnl_net_pct")
        if run.strategy == "random":
            saw_random = True
            random_curves.append(series)
        else:
            curves[f"{run.label}_net_pct"] = series

    indexes = [s.index for s in curves.values()]
    indexes.extend(s.index for s in random_curves)
    if all_prediction_times:
        indexes.append(pd.DatetimeIndex(all_prediction_times).unique().sort_values())
    bnh = None
    if all_prediction_times and tickers:
        bnh = _buy_and_hold_series(
            store_dir=store_dir,
            tickers=sorted(tickers),
            t_min=min(all_prediction_times),
            t_max=max(all_prediction_times),
        )
        if bnh is not None:
            indexes.append(bnh.index)

    if not indexes:
        return pd.DataFrame()

    index = indexes[0]
    for idx in indexes[1:]:
        index = index.union(idx)
    index = index.sort_values()
    out = pd.DataFrame({"timestamp": index})

    for name, series in curves.items():
        out[name] = _align_curve(series, index)

    for name in empty_curve_names:
        if name not in out.columns:
            out[name] = 0.0

    if random_curves:
        aligned = np.vstack([_align_curve(s, index) for s in random_curves])
        out["random_mean_net_pct"] = np.nanmean(aligned, axis=0)
        out["random_p05_net_pct"] = np.nanpercentile(aligned, 5, axis=0)
        out["random_p95_net_pct"] = np.nanpercentile(aligned, 95, axis=0)
    elif saw_random:
        out["random_mean_net_pct"] = 0.0
        out["random_p05_net_pct"] = 0.0
        out["random_p95_net_pct"] = 0.0

    if bnh is not None:
        out["buy_and_hold_pct"] = _align_curve(bnh, index)

    return out


def _equity_series(eq: pd.DataFrame, column: str) -> pd.Series:
    work = eq.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    values = pd.to_numeric(work[column], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=work["timestamp"])
    series = series.dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def _align_curve(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    aligned = series.reindex(index).ffill().fillna(0.0)
    return aligned.to_numpy(dtype=float)


def _buy_and_hold_series(
    *,
    store_dir: Path,
    tickers: list[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> Optional[pd.Series]:
    if not store_dir.exists():
        return None

    returns = []
    for ticker in tickers:
        raw = _load_price_history(store_dir, ticker)
        if raw is None:
            continue
        if "close" not in raw.columns:
            continue
        close = raw.loc[t_min:t_max, "close"].dropna()
        if len(close) < 2:
            continue
        returns.append((close / float(close.iloc[0]) - 1.0) * 100.0)

    if not returns:
        return None
    combined = pd.concat(returns, axis=1, sort=True).sort_index().ffill()
    return combined.mean(axis=1).dropna()


def _load_price_history(store_dir: Path, ticker: str) -> Optional[pd.DataFrame]:
    """Load flat and month-partitioned OHLCV files for one ticker."""
    paths: list[Path] = []
    flat_path = store_dir / f"{ticker}.csv"
    if flat_path.exists():
        paths.append(flat_path)
    paths.extend(sorted(store_dir.glob(f"????-??/{ticker}.csv")))

    frames: list[pd.DataFrame] = []
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


def _write_benchmark_config(
    path: Path,
    *,
    cfg: dict[str, Any],
    exp_id: str,
    split: str,
    benchmark_id: str,
    random_trade_probability: float,
    default_trade_probability: float,
    random_seeds: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "experiment_id": exp_id,
                "split": split,
                "timestamp_run": datetime.now(tz=timezone.utc).isoformat(),
                "random_trade_probability": random_trade_probability,
                "council_directional_rate_used_as_default": default_trade_probability,
                "random_seeds": random_seeds,
                "random_fallback_trade_probability": cfg.get("benchmark", {}).get(
                    "random_fallback_trade_probability"
                ),
                "predict": cfg.get("predict", {}),
                "meta": cfg.get("meta", {}),
                "ensemble": cfg.get("ensemble", {}),
                "benchmark": cfg.get("benchmark", {}),
            },
            indent=2,
            default=str,
        )
    )


def _plot_benchmark_outputs(
    *,
    summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    equity: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if equity.empty:
        return

    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    x = pd.to_datetime(equity["timestamp"], errors="coerce")

    fig, ax = plt.subplots(figsize=(14, 6))
    _plot_curve_if_present(ax, x, equity, "council_meta_net_pct", "Council + meta", "#1f77b4", linewidth=1.8)
    single_cols = [c for c in equity.columns if c.startswith("single_") and c.endswith("_net_pct")]
    for col in single_cols:
        _plot_curve_if_present(ax, x, equity, col, col.replace("_net_pct", ""), "#2ca02c", linewidth=1.4)
    _plot_curve_if_present(ax, x, equity, "one_layer_cnn_net_pct", "One-layer CNN", "#9467bd", linewidth=1.2)
    if "random_mean_net_pct" in equity.columns:
        ax.plot(x, equity["random_mean_net_pct"], color="#7f7f7f", linewidth=1.2, label="Random mean")
        if {"random_p05_net_pct", "random_p95_net_pct"}.issubset(equity.columns):
            ax.fill_between(
                x,
                equity["random_p05_net_pct"].astype(float),
                equity["random_p95_net_pct"].astype(float),
                color="#7f7f7f",
                alpha=0.15,
                label="Random 5-95%",
            )
    _plot_curve_if_present(ax, x, equity, "buy_and_hold_pct", "Buy & hold", "#111111", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Benchmark Equity Comparison", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative net PnL (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    _save_figure(fig, figures_dir / "01_equity_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_df = summary.copy()
    if not bar_df.empty and "final_portfolio_net_pct" in bar_df.columns:
        labels = bar_df["run"].astype(str)
        values = pd.to_numeric(bar_df["final_portfolio_net_pct"], errors="coerce").fillna(0.0)
        colors = ["#1f77b4" if s == "council_meta" else "#7f7f7f" for s in bar_df["strategy"]]
        ax.bar(labels, values, color=colors)
        if "final_portfolio_net_pct_p05" in bar_df.columns and "final_portfolio_net_pct_p95" in bar_df.columns:
            for i, row in bar_df.iterrows():
                if row.get("strategy") != "random":
                    continue
                y = float(row["final_portfolio_net_pct"])
                low = float(row["final_portfolio_net_pct_p05"])
                high = float(row["final_portfolio_net_pct_p95"])
                ax.errorbar(i, y, yerr=[[y - low], [high - y]], fmt="none", ecolor="black", capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Final Net Portfolio PnL", fontweight="bold")
        ax.set_ylabel("Final net PnL (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir / "02_final_net_pnl_bar.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    if not seed_summary.empty:
        random = seed_summary[seed_summary["strategy"] == "random"]
        other = seed_summary[seed_summary["strategy"] != "random"]
        if not random.empty:
            ax.scatter(
                random["executed_trades"],
                random["final_portfolio_net_pct"],
                color="#7f7f7f",
                alpha=0.35,
                label="Random seeds",
            )
        if not other.empty:
            ax.scatter(
                other["executed_trades"],
                other["final_portfolio_net_pct"],
                color="#1f77b4",
                s=70,
                label="Models",
            )
            for _, row in other.iterrows():
                ax.annotate(str(row["run"]), (row["executed_trades"], row["final_portfolio_net_pct"]))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Executed Trades vs Final Net PnL", fontweight="bold")
        ax.set_xlabel("Executed trades")
        ax.set_ylabel("Final net PnL (%)")
        ax.legend()
        ax.grid(alpha=0.3)
    _save_figure(fig, figures_dir / "03_trades_vs_pnl.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    if not seed_summary.empty:
        random = seed_summary[seed_summary["strategy"] == "random"]
        if not random.empty:
            values = pd.to_numeric(random["final_portfolio_net_pct"], errors="coerce").dropna()
            ax.hist(values, bins=min(30, max(5, int(math.sqrt(len(values))))), color="#7f7f7f", alpha=0.75)
            for _, row in seed_summary[seed_summary["strategy"] != "random"].iterrows():
                ax.axvline(float(row["final_portfolio_net_pct"]), linewidth=1.4, label=str(row["run"]))
            ax.set_title("Random Baseline Distribution", fontweight="bold")
            ax.set_xlabel("Final net PnL (%)")
            ax.set_ylabel("Random seed count")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, figures_dir / "04_random_baseline_distribution.png")


def _plot_curve_if_present(
    ax,
    x: pd.Series,
    df: pd.DataFrame,
    column: str,
    label: str,
    color: str,
    *,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    if column in df.columns:
        ax.plot(x, df[column].astype(float), color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def _save_figure(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
