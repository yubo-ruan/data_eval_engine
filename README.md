# data_eval_engine

Starter PyTorch project with a minimal-yet-structured training pipeline. It ships with
configuration management, dataset/dataloader helpers, a simple classifier, and a training
script that can be extended for your own experiments.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install --upgrade pip
pip install -e .[dev]
```

Download MNIST and kick off training (defaults to `configs/default.yaml`):

```bash
python -m data_eval_engine.train --config configs/default.yaml
```

## Project structure

```
├── configs/            # YAML configs for experiments
├── src/data_eval_engine
│   ├── config.py       # Pydantic config model + loader helper
│   ├── data.py         # TorchVision-based dataloaders
│   ├── model.py        # Simple MLP classifier
│   ├── train.py        # Training loop + CLI entry point
│   └── utils.py        # Seed/device helpers
└── tests/ (add your tests here)
```

## Next steps

- Swap out `SimpleClassifier` with your own architecture in `model.py`.
- Point the dataloader to a different dataset or custom `torch.utils.data.Dataset` subclass.
- Extend `TrainingConfig` with hyperparameters such as weight decay, schedulers, or logging
  toggles, then wire them into `train.py`.
