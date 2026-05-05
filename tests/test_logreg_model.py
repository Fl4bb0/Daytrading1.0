from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kvant.models import MODEL_REGISTRY
from kvant.models.logreg import LogisticRegressionModel


class LogisticRegressionModelTests(unittest.TestCase):
    def test_registered_in_model_registry(self) -> None:
        self.assertIn("logreg", MODEL_REGISTRY)

    def test_fit_predict_and_save_load(self) -> None:
        rng = np.random.default_rng(42)
        n = 120
        n_features = 4
        seq_len = 6
        X = rng.normal(size=(n, n_features, seq_len)).astype(np.float32)

        score = X[:, 0, :].mean(axis=1)
        y = np.where(score > 0.2, 2, np.where(score < -0.2, 0, 1)).astype(np.int64)

        model = LogisticRegressionModel(n_features=n_features, n_classes=3, seq_len=seq_len, max_iter=500)
        history = model.fit(X, y)

        self.assertIn("best_epoch", history)
        self.assertEqual(int(history["best_epoch"]), 1)

        pred = model.predict(X)
        proba = model.predict_proba(X)

        self.assertEqual(pred.shape, (n,))
        self.assertEqual(proba.shape, (n, 3))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(n), rtol=1e-5, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "logreg_ckpt"
            model.save(out)
            loaded = LogisticRegressionModel.load(out)

            pred_loaded = loaded.predict(X)
            proba_loaded = loaded.predict_proba(X)

        np.testing.assert_array_equal(pred, pred_loaded)
        np.testing.assert_allclose(proba, proba_loaded, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
