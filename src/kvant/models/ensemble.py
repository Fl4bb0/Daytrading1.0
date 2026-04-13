from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from kvant.models.base import KvantModel
from kvant.utils.ensemble import ensemble_slug


class AveragingEnsembleModel(KvantModel):
    def __init__(
        self,
        members: Sequence[KvantModel],
        *,
        member_names: Optional[Sequence[str]] = None,
        member_paths: Optional[Sequence[Path]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.members = list(members)
        if not self.members:
            raise ValueError("AveragingEnsembleModel requires at least one member")

        self.member_names = [str(v) for v in (member_names or [m.name for m in self.members])]
        self.member_paths = [Path(p) for p in member_paths] if member_paths is not None else None
        self._name = name or ensemble_slug(self.member_names)

    @property
    def name(self) -> str:
        return self._name

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError("AveragingEnsembleModel is inference-only")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [member.predict_proba(X) for member in self.members]
        if not probs:
            raise RuntimeError("AveragingEnsembleModel has no members")
        first_shape = probs[0].shape
        for idx, proba in enumerate(probs[1:], start=1):
            if proba.shape != first_shape:
                raise ValueError(
                    f"Ensemble member {idx} returned shape {proba.shape}, expected {first_shape}"
                )
        stacked = np.stack(probs, axis=0)
        return stacked.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "AveragingEnsembleModel",
            "name": self.name,
            "strategy": "mean",
            "members": [
                {
                    "name": name,
                    "path": str(member_path) if member_path is not None else "",
                }
                for name, member_path in zip(self.member_names, self.member_paths or [None] * len(self.member_names))
            ],
        }
        (path / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path: Path) -> "AveragingEnsembleModel":
        path = Path(path)
        manifest = json.loads((path / "ensemble_manifest.json").read_text())
        members: list[KvantModel] = []
        member_names: list[str] = []
        member_paths: list[Path] = []

        from kvant.models import MODEL_REGISTRY

        for item in manifest.get("members", []):
            name = str(item["name"])
            member_path = Path(item["path"])
            if name not in MODEL_REGISTRY:
                raise ValueError(f"Unknown ensemble member model: {name}")
            members.append(MODEL_REGISTRY[name].load(member_path))
            member_names.append(name)
            member_paths.append(member_path)

        return cls(members, member_names=member_names, member_paths=member_paths, name=manifest.get("name"))

