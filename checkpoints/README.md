# Checkpoints

Pre-trained model weights are not included in this repository due to file size.
Download from Zenodo and place them in this directory.

## Download

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19326039.svg)](https://doi.org/10.5281/zenodo.19326039)

Download URL: https://doi.org/10.5281/zenodo.19326039

The record contains a single archive, `checkpoints.zip`. Unpack it in the repository
root; it expands into this `checkpoints/` directory:

```bash
unzip checkpoints.zip
```

The result should look like:

```
checkpoints/
├── benchmark/
│   ├── seed42.pth
│   ├── seed123.pth
│   ├── seed369.pth
│   ├── seed777.pth
│   └── seed2024.pth
├── application.pth
└── README.md
```

## Model Descriptions

| Checkpoint | Description | Used in |
|------------|-------------|---------|
| `benchmark/seed*.pth` | Best validation RMSE checkpoint for each of the five data splits (seeds 42, 123, 369, 777, 2024) | Performance evaluation (benchmarks) |
| `application.pth` | Best validation RMSE checkpoint (epoch 76) of the primary data split (seed 369) used throughout model development. Identical weights to `benchmark/seed369.pth`, provided separately for convenience. | Drug repositioning, attention analysis, off-target screening (application analyses) |

Benchmark results report mean ± std over the 5 data splits. All checkpoints were selected on the validation set; test sets were not used for model selection.

## Checkpoint Format

Each checkpoint file (`.pth`) contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Training epoch
- `best_loss`: Best validation RMSE

## Usage

```python
from omnibind.predict import load_model
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/default.yaml")

# For performance evaluation
model = load_model(cfg, "checkpoints/benchmark/seed42.pth")

# For application analyses
model = load_model(cfg, "checkpoints/application.pth")
```
