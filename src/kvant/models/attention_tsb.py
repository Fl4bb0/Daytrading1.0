"""
models.attention_tsb — LSTM + multi-head self-attention classifier.

Architecture
------------
1. LSTM processes the input sequence to produce a contextual embedding per timestep.
2. Multi-head self-attention re-weights the LSTM output sequence, letting the model
   focus on the most relevant timesteps rather than relying solely on the final hidden state.
3. A residual + LayerNorm connection stabilises training.
4. The last (most recent) attended timestep feeds a linear classification head.

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
# Raw nn.Module
# ---------------------------------------------------------------------------

class _AttentionTSBNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_heads: int = 4,
    ):
        super().__init__()
        # n_heads must divide hidden_dim
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, seq_len) → (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, hidden_dim)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attended = self.norm(attn_out + lstm_out)  # residual connection
        return self.head(attended[:, -1, :])    # use final (most recent) timestep


# ---------------------------------------------------------------------------
# KvantModel wrapper
# ---------------------------------------------------------------------------

class AttentionTSBModel(KvantModel):
    """
    LSTM + multi-head self-attention model in the KvantModel interface.

    Parameters
    ----------
    n_features : int   — number of input features per timestep.
    n_classes  : int   — number of output classes (default 3).
    hidden_dim : int   — LSTM/attention embedding size (default 64; must be divisible by n_heads).
    num_layers : int   — stacked LSTM layers (default 2).
    dropout    : float — dropout rate (default 0.3).
    n_heads    : int   — number of attention heads (default 4).
    device     : str   — torch device string.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_heads: int = 4,
        device: str = "cpu",
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.device = torch.device(device)
        self.net = _AttentionTSBNet(
            n_features, n_classes, hidden_dim, num_layers, dropout, n_heads
        ).to(self.device)

    @property
    def name(self) -> str:
        return "attention_tsb"

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
                "n_heads": self.n_heads,
            },
            path / "cfg.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "AttentionTSBModel":
        path = Path(path)
        cfg = torch.load(path / "cfg.pt", weights_only=True)
        model = cls(**cfg)
        model.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return model