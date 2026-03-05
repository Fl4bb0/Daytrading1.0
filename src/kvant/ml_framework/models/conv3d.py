from __future__ import annotations

import torch
import torch.nn as nn


class Conv3DClassifier(nn.Module):
    """3D CNN for multi-timeframe financial classification.

    Expected input shape: (batch, n_timeframes, n_features, seq_len)
        - n_timeframes: e.g. 4 for [1m, 5m, 15m, 60m]
        - n_features:   e.g. 5 for OHLCV
        - seq_len:      number of candles per timeframe
    """

    def __init__(self, n_classes: int = 3, hidden_dim: int = 64):
        # hidden_dim controls channel width: layers use hidden_dim//4, //2, and hidden_dim
        # e.g. hidden_dim=64 → 16, 32, 64 channels; hidden_dim=128 → 32, 64, 128
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