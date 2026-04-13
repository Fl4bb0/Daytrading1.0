from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kvant.models.base import KvantModel
from kvant.models.ensemble import AveragingEnsembleModel
from kvant.utils.ensemble import ensemble_slug


class _DummyModel(KvantModel):
    def __init__(self, name: str, proba: list[list[float]]) -> None:
        self._name = name
        self._proba = np.asarray(proba, dtype=np.float32)

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):  # pragma: no cover - unused in tests
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba[: len(X)]

    def save(self, path: Path) -> None:  # pragma: no cover - unused in tests
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path):  # pragma: no cover - unused in tests
        raise NotImplementedError


class AveragingEnsembleModelTests(unittest.TestCase):
    def test_predict_proba_averages_member_probabilities(self) -> None:
        member_a = _DummyModel(
            "resnls",
            [[0.80, 0.10, 0.10], [0.10, 0.20, 0.70]],
        )
        member_b = _DummyModel(
            "conv1d",
            [[0.40, 0.20, 0.40], [0.10, 0.10, 0.80]],
        )
        ensemble = AveragingEnsembleModel([member_a, member_b])

        proba = ensemble.predict_proba(np.zeros((2, 4), dtype=np.float32))

        np.testing.assert_allclose(
            proba,
            np.asarray([[0.60, 0.15, 0.25], [0.10, 0.15, 0.75]], dtype=np.float32),
        )

    def test_predict_uses_averaged_probabilities(self) -> None:
        member_a = _DummyModel(
            "resnls",
            [[0.80, 0.10, 0.10], [0.10, 0.20, 0.70]],
        )
        member_b = _DummyModel(
            "conv1d",
            [[0.40, 0.20, 0.40], [0.10, 0.10, 0.80]],
        )
        ensemble = AveragingEnsembleModel([member_a, member_b])

        preds = ensemble.predict(np.zeros((2, 4), dtype=np.float32))

        np.testing.assert_array_equal(preds, np.asarray([0, 2]))

    def test_ensemble_slug_is_deterministic(self) -> None:
        self.assertEqual(ensemble_slug(["resnls", "conv1d"]), "ensemble_resnls-conv1d")

    def test_save_writes_manifest(self) -> None:
        member_a = _DummyModel("resnls", [[0.80, 0.10, 0.10]])
        member_b = _DummyModel("conv1d", [[0.40, 0.20, 0.40]])
        ensemble = AveragingEnsembleModel([member_a, member_b], member_paths=[Path("a"), Path("b")])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "ensemble"
            ensemble.save(out_dir)
            self.assertTrue((out_dir / "ensemble_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()

