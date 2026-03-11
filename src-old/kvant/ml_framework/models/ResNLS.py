# Architecture inspired by the paper "Stock Price Prediction using ResNLS Technique "
# Link: file:///Users/oskarkarlsson/Desktop/DTU/4.%20Semester/Project%20Work/Papers/resnet_trading.pdf

# Note: The crypto paper uses 3 layers

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNLS(nn.Module):
    """ResNet + LSTM hybrid for stock price prediction.

    Residual block extracts local dependency features between neighboring
    timesteps via convolution, then adds them element-wise to the projected
    input before passing through LSTM.

    Expected input shape: (batch, n_features, seq_len)
        - n_features: number of input features per timestep
        - seq_len: number of lookback bars

    Reference: Dubey & Dixit, Procedia Computer Science 260 (2025) 752-760
    """

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

        # Residual block: extract dependency features between neighboring timesteps
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
        residual = self.residual_proj(x)                    # (batch, n_filters, seq_len)
        out = F.relu(self.bn1(self.conv1(x)))               # (batch, n_filters, seq_len)
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))             # (batch, n_filters, seq_len)
        combined = out + residual                            # (batch, n_filters, seq_len)

        lstm_in = combined.permute(0, 2, 1)                 # (batch, seq_len, n_filters)
        _, (h_n, _) = self.lstm(lstm_in)                    # h_n: (1, batch, lstm_hidden)
        return self.head(h_n.squeeze(0))                    # (batch, n_classes)