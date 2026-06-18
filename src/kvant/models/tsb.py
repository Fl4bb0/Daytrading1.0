"""
models.tsb — LSTM-based time-series classifier wrapped in the KvantModel interface.

Architecture
------------
A stacked LSTM processes the input sequence timestep-by-timestep; the final
layer's hidden state is passed through a small MLP head for classification.

Expected input shape: (batch, n_features, seq_len)
Output shape:         (batch, n_classes)
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

class _TSBNet(nn.Module):
    """LSTM-based time-series classifier."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, seq_len) → (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden_dim)
        return self.head(h_n[-1])    # use last layer's hidden state


# ---------------------------------------------------------------------------
# KvantModel wrapper
# ---------------------------------------------------------------------------

class TSBModel(KvantModel):
    """
    TSBClassifier (LSTM) wrapped in the KvantModel interface.

    Parameters
    ----------
    n_features : int   — number of input features per timestep.
    n_classes  : int   — number of output classes (default 3).
    hidden_dim : int   — LSTM hidden state size (default 64).
    num_layers : int   — number of stacked LSTM layers (default 2).
    dropout    : float — dropout rate (default 0.3).
    device     : str   — torch device string.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device(device)
        self.net = _TSBNet(n_features, n_classes, hidden_dim, num_layers, dropout).to(self.device)

    @property
    def name(self) -> str:
        return "tsb"

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
        torch.save(
            {
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            path / "cfg.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "TSBModel":
        path = Path(path)
        cfg = torch.load(path / "cfg.pt", weights_only=True)
        model = cls(**cfg)
        model.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return model
