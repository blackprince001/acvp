"""
LSTM estimator (T-063).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import BaseEstimator


class LSTMEstimator(BaseEstimator):
    """Multi-layer LSTM with a linear FC head.

    Config keys (in addition to base):
        hidden_size (int): LSTM hidden dimension. Default 64.
        num_layers (int): Number of stacked LSTM layers. Default 2.
        dropout (float): Dropout between layers. Default 0.1.
        bidirectional (bool): Use bidirectional LSTM. Default False.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        hidden = config.get("hidden_size", 64)
        layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)
        bidir = config.get("bidirectional", False)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidir,
            batch_first=True,
        )
        fc_in = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(fc_in, self.horizon * self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)   # h: (num_layers * D, B, hidden)
        if self.lstm.bidirectional:
            last = torch.cat([h[-2], h[-1]], dim=1)  # (B, hidden * 2)
        else:
            last = h[-1]                              # (B, hidden)
        return self.fc(last)                          # (B, horizon * output_size)
