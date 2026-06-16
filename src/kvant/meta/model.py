"""
meta.model — Minimal regression model for ranking directional trade candidates.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kvant.meta.features import META_FEATURE_COLUMNS, META_TARGET_COLUMN


class RidgeMetaModel:
    """Small tabular regressor trained on top of base-model prediction rows."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.feature_columns = list(feature_columns or META_FEATURE_COLUMNS)
        self.target_column = META_TARGET_COLUMN
        self._pipeline: Pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )

    @property
    def name(self) -> str:
        return "ridge_meta"

    def _feature_matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = frame.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        X_np = X.to_numpy(dtype=float)
        mask = np.isfinite(X_np).all(axis=1)
        return X_np, mask

    def fit(self, frame: pd.DataFrame) -> dict[str, float | int]:
        X_np, feature_mask = self._feature_matrix(frame)
        y = pd.to_numeric(frame.get(self.target_column), errors="coerce").to_numpy(dtype=float)
        keep_mask = feature_mask & np.isfinite(y)
        if not bool(keep_mask.any()):
            raise ValueError("No valid rows available to train the meta model.")

        X_train = X_np[keep_mask]
        y_train = y[keep_mask]
        self._pipeline.fit(X_train, y_train)

        pred = self._pipeline.predict(X_train)
        return {
            "n_samples": int(len(y_train)),
            "mae": float(mean_absolute_error(y_train, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, pred))),
            "r2": float(r2_score(y_train, pred)),
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X_np, feature_mask = self._feature_matrix(frame)
        out = np.full(len(frame), np.nan, dtype=float)
        if bool(feature_mask.any()):
            out[feature_mask] = self._pipeline.predict(X_np[feature_mask])
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as f:
            pickle.dump(
                {
                    "alpha": self.alpha,
                    "feature_columns": self.feature_columns,
                    "pipeline": self._pipeline,
                },
                f,
            )
        (path / "meta_config.json").write_text(
            json.dumps(
                {
                    "model_name": self.name,
                    "alpha": self.alpha,
                    "feature_columns": self.feature_columns,
                    "target_column": self.target_column,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeMetaModel":
        path = Path(path)
        with (path / "model.pkl").open("rb") as f:
            payload = pickle.load(f)
        model = cls(
            alpha=float(payload["alpha"]),
            feature_columns=list(payload["feature_columns"]),
        )
        model._pipeline = payload["pipeline"]
        return model


class BinaryMetaClassifier:
    """
    Meta-labeler (López de Prado, AFML ch. 4): binary classifier trained on top of the
    primary model's predictions to filter trades by predicted profitability.

    Input features come from ``add_meta_features()`` (side confidence, ticker prior, etc.).
    Target: 1 if the trade's realised net P&L > 0, else 0.
    At inference, ``predict_proba()`` returns P(profitable) — use as a confidence gate.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.C = float(C)
        self.feature_columns = list(feature_columns or META_FEATURE_COLUMNS)
        self.target_column = META_TARGET_COLUMN
        self._pipeline: Pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=self.C,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                )),
            ]
        )

    @property
    def name(self) -> str:
        return "binary_meta"

    def _feature_matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = frame.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        X_np = X.to_numpy(dtype=float)
        mask = np.isfinite(X_np).all(axis=1)
        return X_np, mask

    def fit(self, frame: pd.DataFrame) -> dict[str, float | int]:
        """Train on directional rows. Returns train accuracy and sample count."""
        X_np, feature_mask = self._feature_matrix(frame)
        raw_target = pd.to_numeric(frame.get(self.target_column), errors="coerce").to_numpy(dtype=float)
        y = (raw_target > 0.0).astype(int)
        keep_mask = feature_mask & np.isfinite(raw_target)
        if not bool(keep_mask.any()):
            raise ValueError("No valid rows available to train the meta classifier.")

        self._pipeline.fit(X_np[keep_mask], y[keep_mask])
        pred = self._pipeline.predict(X_np[keep_mask])
        return {
            "n_samples": int(keep_mask.sum()),
            "train_accuracy": float((pred == y[keep_mask]).mean()),
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Returns 1 (trade expected profitable) or 0 (not), -1 for missing features."""
        X_np, feature_mask = self._feature_matrix(frame)
        out = np.full(len(frame), -1, dtype=int)
        if bool(feature_mask.any()):
            out[feature_mask] = self._pipeline.predict(X_np[feature_mask])
        return out

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Returns P(profitable) in [0, 1] for each row; NaN where features are missing."""
        X_np, feature_mask = self._feature_matrix(frame)
        out = np.full(len(frame), np.nan, dtype=float)
        if bool(feature_mask.any()):
            probs = self._pipeline.predict_proba(X_np[feature_mask])
            out[feature_mask] = probs[:, 1]
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as f:
            pickle.dump(
                {
                    "C": self.C,
                    "feature_columns": self.feature_columns,
                    "pipeline": self._pipeline,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BinaryMetaClassifier":
        path = Path(path)
        with (path / "model.pkl").open("rb") as f:
            payload = pickle.load(f)
        obj = cls(C=float(payload["C"]), feature_columns=list(payload["feature_columns"]))
        obj._pipeline = payload["pipeline"]
        return obj
