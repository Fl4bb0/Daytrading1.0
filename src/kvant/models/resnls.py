"""
models.resnls — ResNet + LSTM hybrid wrapped in the KvantModel interface.

Architecture
------------
A residual convolutional block (two Conv1d layers with a skip connection)
extracts local dependency features, then an LSTM processes the enriched
sequence and the final hidden state is fed into a linear head.

Reference: Dubey & Dixit, Procedia Computer Science 260 (2025) 752–760
           "Stock Price Prediction using ResNLS Technique"

Expected input shape: (batch, n_features, seq_len)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kvant.models.base import KvantModel


# ---------------------------------------------------------------------------
# Raw nn.Module (architecture only)
# ---------------------------------------------------------------------------

class _ResNLSNet(nn.Module):
    """ResNet + LSTM hybrid for stock price prediction."""

    def __init__(
        self,
        n_features: int = 1,
        seq_len: int = 5,
        n_filters: int = 64,
        lstm_hidden: int = 32,
        n_classes: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # Residual block
        self.conv1 = nn.Conv1d(n_features, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.dropout = nn.Dropout(0.2)

        # Project input channels to n_filters for the residual connection
        self.residual_proj = nn.Conv1d(n_features, n_filters, kernel_size=1)

        # LSTM on residual-augmented sequence
        self.lstm = nn.LSTM(input_size=n_filters, hidden_size=lstm_hidden, batch_first=True)

        # Output head
        self.head = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, seq_len)
        residual = self.residual_proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        combined = out + residual                       # (batch, n_filters, seq_len)

        lstm_in = combined.permute(0, 2, 1)             # (batch, seq_len, n_filters)
        _, (h_n, _) = self.lstm(lstm_in)                # h_n: (1, batch, lstm_hidden)
        return self.head(h_n.squeeze(0))                # (batch, n_classes)


# ---------------------------------------------------------------------------
# KvantModel wrapper
# ---------------------------------------------------------------------------

class ResNLSModel(KvantModel):
    """
    ResNLS wrapped in the KvantModel interface.

    Parameters
    ----------
    n_features  : int — number of input features per timestep.
    seq_len     : int — lookback window length (number of bars).
    n_filters   : int — number of Conv1d filters in the residual block.
    lstm_hidden : int — LSTM hidden state size.
    n_classes   : int — number of output classes (default 3).
    device      : str — torch device string.
    """

    def __init__(
        self,
        n_features: int = 1,
        seq_len: int = 5,
        n_filters: int = 64,
        lstm_hidden: int = 32,
        n_classes: int = 3,
        device: str = "cpu",
    ) -> None:
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_filters = n_filters
        self.lstm_hidden = lstm_hidden
        self.n_classes = n_classes
        self.device = torch.device(device)
        self.net = _ResNLSNet(n_features, seq_len, n_filters, lstm_hidden, n_classes).to(self.device)

    @property
    def name(self) -> str:
        return "resnls"

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
                "seq_len": self.seq_len,
                "n_filters": self.n_filters,
                "lstm_hidden": self.lstm_hidden,
                "n_classes": self.n_classes,
            },
            path / "cfg.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "ResNLSModel":
        path = Path(path)
        cfg = torch.load(path / "cfg.pt", weights_only=True)
        model = cls(**cfg)
        model.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return model
