"""End-to-end training loop for the starter project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

from .config import TrainingConfig, load_config
from .data import create_dataloaders
from .model import SimpleClassifier
from .utils import resolve_device, set_seed


def _iterate_batches(loader: Iterable, *, device: torch.device):
    for inputs, targets in loader:
        yield inputs.to(device), targets.to(device)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
) -> float:
    model.train()
    running_loss = 0.0
    for step, (inputs, targets) in enumerate(_iterate_batches(loader, device=device), start=1):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % log_every == 0:
            avg_loss = running_loss / log_every
            tqdm.write(f"step {step:04d} | loss={avg_loss:.4f}")
            running_loss = 0.0
    return running_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in _iterate_batches(loader, device=device):
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    return correct / total if total else 0.0


def run_training(config_path: str | Path | None = None) -> None:
    config = load_config(config_path)
    set_seed(config.seed)
    device = resolve_device(config.device)

    train_loader, val_loader = create_dataloaders(config)
    model = SimpleClassifier(config.hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.num_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.num_epochs}")
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            log_every=config.log_every,
        )
        val_acc = evaluate(model, val_loader, device)
        tqdm.write(f"Validation accuracy: {val_acc * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the demo PyTorch training loop")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML config file",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
