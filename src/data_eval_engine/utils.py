"""Utility helpers for reproducibility and device selection."""

from __future__ import annotations

import random
from typing import Final

import numpy as np
import torch


AVAILABLE_ACCELERATORS: Final[tuple[str, ...]] = ("cuda", "mps", "cpu")


def resolve_device(preferred: str) -> torch.device:
    """Return the first available device, honoring the preference when possible."""

    preferred = preferred.lower()
    if preferred in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
