"""
TCN estimator (T-065).

Temporal Convolutional Network with dilated causal convolutions and
residual connections.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _TemporalBlock(nn.Module):
    """Single TCN residual block: two dilated causal conv layers + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def _chomp(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        return x[:, :, :-pad].contiguous() if pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.conv1.weight.shape[2]  # kernel_size
        dilation = self.conv1.dilation[0]
        causal_pad = (pad - 1) * dilation

        out = self._chomp(self.conv1(x), causal_pad)
        out = self.relu(out)
        out = self.dropout(out)
        out = self._chomp(self.conv2(out), causal_pad)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEstimator(nn.Module):
    """TCN-based traffic density estimator.

    Config keys (in addition to base):
        num_channels (list[int]): Output channels per temporal block.
            Default [64, 64, 64].
        kernel_size (int): Convolution kernel size. Default 3.
        dropout (float): Dropout rate. Default 0.1.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.input_size: int = config["input_size"]
        self.output_size: int = config["output_size"]
        self.horizon: int = config["horizon"]

        channels: list[int] = config.get("num_channels", [64, 64, 64])
        kernel_size: int = config.get("kernel_size", 3)
        dropout: float = config.get("dropout", 0.1)

        layers = []
        in_ch = self.input_size
        for i, out_ch in enumerate(channels):
            layers.append(
                _TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], self.horizon * self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T) for Conv1d
        out = self.network(x.permute(0, 2, 1))  # (B, C, T)
        last = out[:, :, -1]                     # (B, C) — last timestep
        return self.fc(last)                     # (B, horizon * output_size)

    def predict(self, x) -> "np.ndarray":
        import numpy as np
        self.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.forward(x)
        return out.reshape(-1, self.horizon, self.output_size).cpu().numpy()
