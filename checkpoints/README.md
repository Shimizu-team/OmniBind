# Checkpoints

Pre-trained model weights are not included in this repository due to file size.

## Download

Pre-trained weights will be available at: [TBD - link to be added upon publication]

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
model = load_model(cfg, "checkpoints/best_model.pth")
```
