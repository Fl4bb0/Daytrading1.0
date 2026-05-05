from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from kvant.models.base import KvantModel
from kvant.training.metrics import classification_metrics
from kvant.training.trainer import TrainConfig, Trainer


class SklearnTrainer(Trainer):
    """Trainer wrapper for sklearn-backed KvantModel implementations."""

    def __init__(self, model: KvantModel, cfg: TrainConfig, logger: Optional[Any] = None) -> None:
        super().__init__(model, cfg)
        self.logger = logger

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        model = self.model
        X_fit = model._reshape_X(X_train)
        model.scaler.fit(X_fit)
        X_fit_scaled = model.scaler.transform(X_fit)
        model.model.fit(X_fit_scaled, y_train)
        model._is_fitted = True

        train_acc = float((model.model.predict(X_fit_scaled) == y_train).mean())
        val_acc = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_pred = model.predict(X_val)
            val_acc = float((val_pred == y_val).mean())

        if self.cfg.checkpoint_dir is not None:
            model.save(Path(self.cfg.checkpoint_dir))

        return {
            "train_loss": [],
            "val_accuracy": [] if val_acc is None else [val_acc],
            "best_val_accuracy": val_acc,
            "best_epoch": 1,
            "train_accuracy": train_acc,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.model.predict(X)
        metrics = classification_metrics(y, y_pred)
        return {k: float(v) for k, v in metrics.items()}
