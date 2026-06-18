"""
models.conv3d — 3-D convolutional classifier wrapped in the KvantModel interface.

Architecture
------------
Three Conv3d blocks (16 → 32 → 64 filters) operating across timeframes, features
and time simultaneously, followed by AdaptiveAvgPool3d and an MLP head.
Designed for multi-timeframe input.

Expected input shape: (batch, n_timeframes, n_features, seq_len)
    - n_timeframes : e.g. 4 for [1 m, 5 m, 15 m, 60 m]
    - n_features   : e.g. 5 for OHLCV
    - seq_len      : number of candles per timeframe
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from kvant.models.base import KvantModel


# ---------------------------------------------------------------------------
# Raw nn.Module (architecture only)
# ---------------------------------------------------------------------------

class _Conv3DNet(nn.Module):
    """3D CNN for multi-timeframe financial classification."""

    def __init__(self, n_classes: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1 — local patterns across features and time
            nn.Conv3d(1, hidden_dim // 4, kernel_size=(1, 3, 5), padding=(0, 1, 2)),
            nn.BatchNorm3d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            # Layer 2 — merge across timeframes
            nn.Conv3d(hidden_dim // 4, hidden_dim // 2, kernel_size=(2, 3, 5), padding=(0, 1, 2)),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            # Layer 3 — high-level cross-timeframe representation
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=(1, 3, 5), padding=(0, 1, 2)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_timeframes, n_features, seq_len) → add channel dim
        return self.net(x.unsqueeze(1))


# ---------------------------------------------------------------------------
# KvantModel wrapper
# ---------------------------------------------------------------------------

class Conv3DModel(KvantModel):
    """
    Conv3DClassifier wrapped in the KvantModel interface.

    Parameters
    ----------
    n_classes  : int  — number of output classes (default 3).
    hidden_dim : int  — base channel width; layers use hidden_dim//4, //2, and hidden_dim.
    device     : str  — torch device string.
    """

    def __init__(
        self,
        n_classes: int = 3,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.net = _Conv3DNet(n_classes, hidden_dim).to(self.device)

    @property
    def name(self) -> str:
        return "conv3d"

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
        torch.save({"n_classes": self.n_classes, "hidden_dim": self.hidden_dim}, path / "cfg.pt")

    @classmethod
    def load(cls, path: Path) -> "Conv3DModel":
        path = Path(path)
        cfg = torch.load(path / "cfg.pt", weights_only=True)
        model = cls(n_classes=cfg["n_classes"], hidden_dim=cfg["hidden_dim"])
        model.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return model
