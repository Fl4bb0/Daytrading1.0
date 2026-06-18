from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kvant.models.base import KvantModel


class LogisticRegressionModel(KvantModel):
    """Multiclass logistic-regression baseline on flattened sequence features."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        seq_len: int = 20,
        device: str = "cpu",
        max_iter: int = 1000,
        C: float = 1.0,
    ) -> None:
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.seq_len = int(seq_len)
        self.device = str(device)  # Kept for constructor parity with torch models.
        self.max_iter = int(max_iter)
        self.C = float(C)

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "logreg"

    def _reshape_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n, features, seq_len), got {X.shape}")
        n, f, L = X.shape
        if f != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {f}")
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")
        return X.reshape(n, f * L)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from kvant.training.sklearn_trainer import SklearnTrainer
        from kvant.training.trainer import TrainConfig

        cfg = kwargs.pop("cfg", None)
        if cfg is None:
            cfg = TrainConfig(**kwargs)
        return SklearnTrainer(self, cfg).fit(X_train, y_train, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._reshape_X(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._reshape_X(X)
        X_scaled = self.scaler.transform(X_flat)
        proba = self.model.predict_proba(X_scaled)
        return np.asarray(proba, dtype=np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "seq_len": self.seq_len,
            "device": self.device,
            "max_iter": self.max_iter,
            "C": self.C,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self._is_fitted,
        }
        with (path / "model.pkl").open("wb") as f:
            pickle.dump(payload, f)

        (path / "cfg.json").write_text(
            json.dumps(
                {
                    "n_features": self.n_features,
                    "n_classes": self.n_classes,
                    "seq_len": self.seq_len,
                    "max_iter": self.max_iter,
                    "C": self.C,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        path = Path(path)
        with (path / "model.pkl").open("rb") as f:
            payload = pickle.load(f)

        model = cls(
            n_features=int(payload["n_features"]),
            n_classes=int(payload["n_classes"]),
            seq_len=int(payload.get("seq_len", 20)),
            device=str(payload.get("device", "cpu")),
            max_iter=int(payload.get("max_iter", 1000)),
            C=float(payload.get("C", 1.0)),
        )
        model.scaler = payload["scaler"]
        model.model = payload["model"]
        model._is_fitted = bool(payload.get("is_fitted", True))
        return model
