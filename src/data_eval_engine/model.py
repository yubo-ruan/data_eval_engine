"""Model definitions used by the starter project."""

from __future__ import annotations

import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Two-layer MLP for MNIST-sized inputs (1x28x28)."""

    def __init__(self, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs):  # type: ignore[override]
        return self.net(inputs)
