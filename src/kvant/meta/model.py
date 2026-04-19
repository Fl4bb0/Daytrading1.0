"""
meta.model — Minimal regression model for ranking directional trade candidates.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
