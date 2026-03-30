# Checkpoints

Pre-trained model weights are not included in this repository due to file size.
Download from Zenodo and place them in this directory.

## Download

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19326040.svg)](https://doi.org/10.5281/zenodo.19326040)

Download URL: https://doi.org/10.5281/zenodo.19326040

After downloading, your directory should look like:

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
| `benchmark/seed*.pth` | Best test-set RMSE models for each random seed | Performance evaluation (Tables) |
| `application.pth` | Selected for generalization to novel compound-target pairs | Drug repositioning, off-target screening (Figures) |

Paper results report mean ± std over the 5 benchmark seeds.

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
