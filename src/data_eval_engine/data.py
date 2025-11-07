"""Data loading utilities built around torchvision datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainingConfig


def _mnist_dataset(train: bool, root: Path) -> datasets.MNIST:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return datasets.MNIST(root=str(root), train=train, download=True, transform=transform)


def create_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders for MNIST."""

    data_root = Path("data")
    train_set = _mnist_dataset(train=True, root=data_root)
    val_set = _mnist_dataset(train=False, root=data_root)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
