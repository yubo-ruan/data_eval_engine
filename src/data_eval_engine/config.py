"""Configuration helpers for the training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class TrainingConfig(BaseModel):
    """All parameters needed to run the demo training loop."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    batch_size: PositiveInt = 64
    num_epochs: PositiveInt = 5
    learning_rate: float = Field(default=1e-3, gt=0)
    num_workers: int = 2
    log_every: PositiveInt = 100
    hidden_dim: PositiveInt = 256
    device: str = "cpu"


def load_config(path: str | Path | None = None) -> TrainingConfig:
    """Load a YAML config file or return defaults when no path is provided."""

    if path is None:
        return TrainingConfig()

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return TrainingConfig(**raw)
