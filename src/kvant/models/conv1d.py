"""
models.conv1d — 1-D convolutional classifier wrapped in the KvantModel interface.

Architecture
------------
Two Conv1d blocks (32 → 64 filters, kernel=5) with BatchNorm, ReLU and Dropout,
followed by AdaptiveAvgPool1d and a linear classification head.

Expected input shape: (batch, n_features, seq_len)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from kvant.models.base import KvantModel


# ---------------------------------------------------------------------------
# Raw nn.Module (architecture only — no training logic)
# ---------------------------------------------------------------------------

class _Conv1DNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# KvantModel wrapper
# ---------------------------------------------------------------------------

class Conv1DModel(KvantModel):
    """
    Conv1DClassifier wrapped in the KvantModel interface.

    Parameters
    ----------
    n_features : int   — number of input features per timestep.
    n_classes  : int   — number of output classes (default 3).
    device     : str   — torch device string, e.g. ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        device: str = "cpu",
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = torch.device(device)
        self.net = _Conv1DNet(n_features, n_classes).to(self.device)

    @property
    def name(self) -> str:
        return "conv1d"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from kvant.training.pytorch_trainer import PytorchTrainer
        from kvant.training.trainer import TrainConfig
        cfg = kwargs.pop("cfg", None)
        if cfg is None:
            cfg = TrainConfig(**kwargs)
        return PytorchTrainer(self, cfg).fit(X_train, y_train, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.net(t)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            return torch.softmax(self.net(t), dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        torch.save({"n_features": self.n_features, "n_classes": self.n_classes}, path / "cfg.pt")

    @classmethod
    def load(cls, path: Path) -> "Conv1DModel":
        path = Path(path)
        cfg = torch.load(path / "cfg.pt", weights_only=True)
        model = cls(n_features=cfg["n_features"], n_classes=cfg["n_classes"])
        model.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return model
