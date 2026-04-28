from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from kvant.utils.device import resolve_torch_device
from kvant.utils.ensemble import ensemble_slug, normalize_model_names
from kvant.utils.pipeline_config import list_from_config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PREPARED_ROOT = _PROJECT_ROOT / "prepared"
_CHECKPOINTS_ROOT = _PROJECT_ROOT / "checkpoints"


@dataclass
class TrainedModelArtifact:
    model_name: str
    checkpoint_dir: Path
    best_val_accuracy: float
    best_epoch: int
    test_metrics: dict[str, float]


@dataclass
class LoadedModelRuntime:
    model_names: list[str]
    active_model_name: str
    model: object
    model_path: Path
    model_cls: type
    use_ensemble: bool


def prepared_root_from_config(pipeline_cfg: dict) -> Path:
    return Path(pipeline_cfg["paths"].get("prepared_root", str(_PREPARED_ROOT)))


def checkpoints_root_from_config(pipeline_cfg: dict) -> Path:
    return Path(pipeline_cfg["paths"].get("checkpoints_root", str(_CHECKPOINTS_ROOT)))


def resolve_experiment_dir(exp_id: str, pipeline_cfg: dict) -> Path:
    prepared_root = prepared_root_from_config(pipeline_cfg)
    if exp_id == "last":
        last_file = prepared_root / "last_experiment.txt"
        if not last_file.exists():
            raise SystemExit(f"No last_experiment.txt found in {prepared_root}")
        exp_id = last_file.read_text().strip()

    exp_dir = prepared_root / exp_id
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")
    return exp_dir


def train_experiment(exp_dir: Path, pipeline_cfg: dict) -> list[TrainedModelArtifact]:
    exp_dir = Path(exp_dir)
    train_cfg = pipeline_cfg["train"]
    ensemble_cfg = pipeline_cfg.get("ensemble", {})

    model_names = normalize_model_names(ensemble_cfg.get("models")) or [str(train_cfg.get("model", "conv1d"))]
    requested_device = str(train_cfg.get("device", "auto"))
    try:
        device = resolve_torch_device(requested_device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if len(model_names) > 1 and "conv3d" in model_names:
        raise SystemExit(
            "conv3d cannot be used in ensemble mode in the current pipeline. "
            "Use non-conv3d models together, or run conv3d as a standalone model"
        )

    cfg_data = json.loads((exp_dir / "config.json").read_text())
    lookback_L = int(cfg_data["lookback_L"])

    index_train = np.load(exp_dir / "index_train.npy")
    index_val = np.load(exp_dir / "index_val.npy")
    index_test = np.load(exp_dir / "index_test.npy")

    X_train, y_train = _load_split(exp_dir, index_train, lookback_L)
    X_val, y_val = _load_split(exp_dir, index_val, lookback_L)
    X_test, y_test = _load_split(exp_dir, index_test, lookback_L)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise SystemExit(
            f"Experiment {exp_dir} has an empty split after preparation. "
            "Walk-forward folds must retain non-empty train/val/test indices."
        )

    n_features = X_train.shape[1]
    n_classes = int(y_train.max()) + 1

    from kvant.models import MODEL_REGISTRY
    from kvant.training.pytorch_trainer import PytorchTrainer
    from kvant.training.trainer import TrainConfig

    artifacts: list[TrainedModelArtifact] = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

        checkpoint_dir = checkpoint_dir_for_experiment(
            exp_dir=exp_dir,
            checkpoints_root=checkpoints_root_from_config(pipeline_cfg),
            prepared_root=prepared_root_from_config(pipeline_cfg),
            model_name=model_name,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model = _build_model(
            MODEL_REGISTRY[model_name],
            n_features=n_features,
            n_classes=n_classes,
            device=device,
            seq_len=lookback_L,
        )
        cfg = TrainConfig(
            epochs=int(train_cfg.get("epochs", 50)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            early_stopping_patience=int(train_cfg.get("patience", 10)),
            checkpoint_dir=checkpoint_dir,
        )
        trainer = PytorchTrainer(model, cfg)
        history = trainer.fit(X_train, y_train, X_val, y_val)
        test_metrics = trainer.evaluate(X_test, y_test)
        model.save(checkpoint_dir)
        artifacts.append(
            TrainedModelArtifact(
                model_name=model_name,
                checkpoint_dir=checkpoint_dir,
                best_val_accuracy=float(history["best_val_accuracy"]),
                best_epoch=int(history["best_epoch"]),
                test_metrics={key: float(value) for key, value in test_metrics.items()},
            )
        )

    return artifacts


def load_runtime_model(exp_dir: Path, pipeline_cfg: dict) -> LoadedModelRuntime:
    exp_dir = Path(exp_dir)
    predict_cfg = pipeline_cfg["predict"]
    ensemble_cfg = pipeline_cfg.get("ensemble", {})
    checkpoints_root = checkpoints_root_from_config(pipeline_cfg)
    prepared_root = prepared_root_from_config(pipeline_cfg)

    model_names = normalize_model_names(ensemble_cfg.get("models"))
    use_ensemble = bool(model_names)
    if use_ensemble and "conv3d" in model_names:
        raise SystemExit(
            "conv3d cannot be used in ensemble mode in the current pipeline. "
            "Use non-conv3d models together, or run conv3d as a standalone model"
        )

    if use_ensemble:
        active_model_name = ensemble_slug(model_names)
    else:
        active_model_name = str(predict_cfg.get("model", "conv1d"))
        model_names = [active_model_name]

    from kvant.models import MODEL_REGISTRY
    from kvant.models.ensemble import AveragingEnsembleModel

    member_models = []
    member_paths = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            raise SystemExit(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        checkpoint = checkpoint_dir_for_experiment(
            exp_dir=exp_dir,
            checkpoints_root=checkpoints_root,
            prepared_root=prepared_root,
            model_name=model_name,
        )
        if not (checkpoint / "weights.pt").exists():
            raise SystemExit(
                f"No checkpoint found at {checkpoint}/weights.pt. "
                "Train the experiment before prediction."
            )
        member_models.append(MODEL_REGISTRY[model_name].load(checkpoint))
        member_paths.append(checkpoint)

    if len(member_models) == 1 and not use_ensemble:
        model = member_models[0]
        model_path = member_paths[0]
        model_cls = type(model)
    else:
        model = AveragingEnsembleModel(
            member_models,
            member_names=model_names,
            member_paths=member_paths,
            name=active_model_name,
        )
        model_path = checkpoint_dir_for_experiment(
            exp_dir=exp_dir,
            checkpoints_root=checkpoints_root,
            prepared_root=prepared_root,
            model_name=active_model_name,
        )
        model.save(model_path)
        model_cls = type(member_models[0])

    return LoadedModelRuntime(
        model_names=model_names,
        active_model_name=active_model_name,
        model=model,
        model_path=model_path,
        model_cls=model_cls,
        use_ensemble=use_ensemble,
    )


def train_meta_experiment(exp_dir: Path, pipeline_cfg: dict) -> Path:
    exp_dir = Path(exp_dir)
    predict_cfg = pipeline_cfg["predict"]
    meta_cfg = pipeline_cfg.get("meta", {})
    trading_cfg = pipeline_cfg.get("trading", {})
    requested_tickers = list_from_config(predict_cfg.get("tickers")) or None
    required_buy_probability = float(predict_cfg.get("required_buy_probability", 0.0))
    required_sell_probability = float(predict_cfg.get("required_sell_probability", 0.0))
    meta_train_split = str(meta_cfg.get("train_split", "val"))
    meta_alpha = float(meta_cfg.get("alpha", 1.0))
    meta_shrinkage_k = float(meta_cfg.get("shrinkage_k", 10.0))
    brokerage_fee = float(trading_cfg.get("brokerage_fee", 0.0008))

    runtime = load_runtime_model(exp_dir, pipeline_cfg)

    from kvant.evaluation import build_prediction_frame
    from kvant.meta import META_FEATURE_COLUMNS, RidgeMetaModel, build_meta_training_frame

    pred_df = build_prediction_frame(
        exp_dir=exp_dir,
        model_path=runtime.model_path,
        model_cls=runtime.model_cls,
        split=meta_train_split,
        tickers=requested_tickers,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        model=runtime.model,
    )
    meta_train_df = build_meta_training_frame(
        pred_df,
        fee=brokerage_fee,
        shrinkage_k=meta_shrinkage_k,
    )
    if meta_train_df.empty:
        raise SystemExit(
            "No valid directional rows were available to train the meta model. "
            "Check that the base model produces BUY/SHORT predictions with probabilities."
        )

    meta_model = RidgeMetaModel(alpha=meta_alpha)
    metrics = meta_model.fit(meta_train_df)
    meta_dir = runtime.model_path / "meta"
    meta_model.save(meta_dir)
    (meta_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "base_model_name": runtime.active_model_name,
                "base_member_models": runtime.model_names,
                "train_split": meta_train_split,
                "n_rows_total": int(len(pred_df)),
                "n_rows_trainable": int(len(meta_train_df)),
                "feature_columns": META_FEATURE_COLUMNS,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return meta_dir


def predict_experiment(exp_dir: Path, pipeline_cfg: dict) -> Path:
    exp_dir = Path(exp_dir)
    predict_cfg = pipeline_cfg["predict"]
    meta_cfg = pipeline_cfg.get("meta", {})
    trading_cfg = pipeline_cfg.get("trading", {})

    required_buy_probability = float(predict_cfg.get("required_buy_probability", 0.0))
    required_sell_probability = float(predict_cfg.get("required_sell_probability", 0.0))
    allow_short = bool(predict_cfg.get("allow_short", True))
    execution_priority = str(predict_cfg.get("execution_priority", "model_confidence"))
    top_k_raw = predict_cfg.get("top_k_per_timestamp")
    top_k_per_timestamp = None if top_k_raw in (None, "", 0) else int(top_k_raw)
    ticker_cooldown_minutes = int(predict_cfg.get("ticker_cooldown_minutes", 0))
    meta_enabled = bool(meta_cfg.get("enabled", False))
    meta_train_split = str(meta_cfg.get("train_split", "val"))
    meta_shrinkage_k = float(meta_cfg.get("shrinkage_k", 10.0))
    meta_min_score_buy_raw = meta_cfg.get("min_score_buy")
    meta_min_score_short_raw = meta_cfg.get("min_score_short")
    meta_min_score_buy = None if meta_min_score_buy_raw in (None, "") else float(meta_min_score_buy_raw)
    meta_min_score_short = None if meta_min_score_short_raw in (None, "") else float(meta_min_score_short_raw)
    requested_tickers = list_from_config(predict_cfg.get("tickers")) or None
    brokerage_fee = float(trading_cfg.get("brokerage_fee", 0.0008))

    runtime = load_runtime_model(exp_dir, pipeline_cfg)
    out_dir = exp_dir / "eval" / f"{runtime.model.name}_{str(predict_cfg.get('split', 'test'))}"

    meta_model = None
    meta_model_path = None
    meta_history_pred_df = None
    if meta_enabled or execution_priority == "meta_score":
        from kvant.evaluation import build_prediction_frame
        from kvant.meta import RidgeMetaModel

        meta_model_path = runtime.model_path / "meta"
        if not (meta_model_path / "model.pkl").exists():
            raise SystemExit(
                f"No meta model found at {meta_model_path}/model.pkl. "
                "Train it before prediction."
            )
        meta_model = RidgeMetaModel.load(meta_model_path)

        current_split = str(predict_cfg.get("split", "test"))
        if current_split != meta_train_split:
            meta_history_pred_df = build_prediction_frame(
                exp_dir=exp_dir,
                model_path=runtime.model_path,
                model_cls=runtime.model_cls,
                split=meta_train_split,
                tickers=requested_tickers,
                required_buy_probability=required_buy_probability,
                required_sell_probability=required_sell_probability,
                model=runtime.model,
            )

    from kvant.evaluation import evaluate_experiment

    return evaluate_experiment(
        exp_dir=exp_dir,
        model_path=runtime.model_path,
        model_cls=runtime.model_cls,
        out_dir=out_dir,
        split=str(predict_cfg.get("split", "test")),
        tickers=requested_tickers,
        fee=brokerage_fee,
        required_buy_probability=required_buy_probability,
        required_sell_probability=required_sell_probability,
        allow_short=allow_short,
        execution_priority=execution_priority,
        top_k_per_timestamp=top_k_per_timestamp,
        ticker_cooldown_minutes=ticker_cooldown_minutes,
        model=runtime.model,
        meta_model=meta_model,
        meta_model_path=meta_model_path,
        meta_history_pred_df=meta_history_pred_df,
        meta_shrinkage_k=meta_shrinkage_k,
        meta_train_split=meta_train_split,
        meta_min_score_buy=meta_min_score_buy,
        meta_min_score_short=meta_min_score_short,
    )


def checkpoint_dir_for_experiment(
    *,
    exp_dir: Path,
    checkpoints_root: Path,
    prepared_root: Path,
    model_name: str,
) -> Path:
    exp_dir = Path(exp_dir).resolve()
    prepared_root = Path(prepared_root).resolve()
    checkpoints_root = Path(checkpoints_root).resolve()

    try:
        exp_rel = exp_dir.relative_to(prepared_root)
    except ValueError:
        exp_rel = Path(exp_dir.name)
    return checkpoints_root / exp_rel / model_name


def _load_split(exp_dir: Path, index: np.ndarray, lookback_L: int) -> tuple[np.ndarray, np.ndarray]:
    from kvant.utils.split_loader import load_split_from_index

    loaded = load_split_from_index(
        exp_dir=exp_dir,
        index=index,
        lookback_L=lookback_L,
        include_timestamps=False,
        include_metadata=False,
    )
    return loaded.X, loaded.y


def _build_model(model_cls, *, n_features: int, n_classes: int, device: str, seq_len: int):
    kwargs = {
        "n_features": n_features,
        "n_classes": n_classes,
        "device": device,
        "seq_len": seq_len,
    }
    accepted = inspect.signature(model_cls.__init__).parameters
    return model_cls(**{key: value for key, value in kwargs.items() if key in accepted})

