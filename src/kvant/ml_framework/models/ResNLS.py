# Architecture inspired by the paper "Stock Price Prediction using ResNLS Technique "
# Link: file:///Users/oskarkarlsson/Desktop/DTU/4.%20Semester/Project%20Work/Papers/resnet_trading.pdf

# Note: The crypto paper uses 3 layers

import torch
import torch.nn as nn


class ResNLS(nn.Module):
    """ResNet + LSTM hybrid for stock price prediction.

    Residual block extracts local dependency features between neighboring
    stock prices via convolution, then adds them element-wise to the original
    input before passing through LSTM.

    Expected input shape: (batch, seq_len)
        - seq_len: number of previous trading days (paper recommends 5)

    Reference: Dubey & Dixit, Procedia Computer Science 260 (2025) 752-760
    """

    def __init__(self, seq_len: int = 5, n_filters: int = 64, lstm_hidden: int = 32):
        super().__init__()
        self.seq_len = seq_len

        # Residual block: extract dependency features between neighboring prices
        self.residual_block = nn.Sequential(
            # Conv 1
            nn.Conv1d(1, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Conv 2
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),

            # Project back to input dims
            nn.Flatten(),
            nn.Linear(n_filters * seq_len, seq_len),
        )

        # LSTM on residual-augmented sequence
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)

        # Output
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        x_conv = x.unsqueeze(1)                        # (batch, 1, seq_len)
        residual = self.residual_block(x_conv)          # (batch, seq_len)
        combined = (residual + x).unsqueeze(-1)         # (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(combined)               # h_n: (1, batch, hidden)
        return self.head(h_n.squeeze(0)).squeeze(-1)    # (batch,)
