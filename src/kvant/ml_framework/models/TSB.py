from __future__ import annotations

import torch.nn as nn


class TSBClassifier(nn.Module):
    """LSTM-based time-series classifier.

    Drop-in replacement for Conv1DClassifier. Processes each timestep as a
    feature vector and classifies using the final hidden state.

    Expected input shape: (batch, n_features, seq_len)
    Output shape:         (batch, n_classes)
    """

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

    def forward(self, x):
        # x: (batch, n_features, seq_len) → (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden_dim)
        return self.head(h_n[-1])    # use last layer's hidden state