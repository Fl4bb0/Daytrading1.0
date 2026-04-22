"""
models.shallow_cnn — one-layer CNN baseline wrapped in the KvantModel interface.

This model is intentionally small. It is meant to be a weak learned baseline for
benchmarks, not a production architecture.

Expected input shape: (batch, n_features, seq_len)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from kvant.models.base import KvantModel


class _ShallowCNNNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        n_channels: int = 16,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_features, n_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShallowCNNModel(KvantModel):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        device: str = "cpu",
        n_channels: int = 16,
        kernel_size: int = 3,
        seq_len: int | None = None,
    ) -> None:
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.n_channels = int(n_channels)
        self.kernel_size = int(kernel_size)
        self.seq_len = None if seq_len is None else int(seq_len)
        self.device = torch.device(device)
        self.net = _ShallowCNNNet(
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            kernel_size=self.kernel_size,
        ).to(self.device)

    @property
    def name(self) -> str:
        return "shallow_cnn"

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

        cfg = kwargs.pop("cfg", TrainConfig(**kwargs))
        return PytorchTrainer(self, cfg).fit(X_train, y_train, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net(t).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            return torch.softmax(self.net(t), dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        torch.save(
            {
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "n_channels": self.n_channels,
                "kernel_size": self.kernel_size,
                "seq_len": self.seq_len,
            },
            path / "cfg.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "ShallowCNNModel":
        path = Path(path)
        cfg = torch.load(path / "cfg.pt", weights_only=True)
        model = cls(**cfg)
        model.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return model
