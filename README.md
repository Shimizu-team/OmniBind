# OmniBind

**Pan–Pharmacological Drug–Target Interaction Prediction with 3D–Informed Protein Encoding at Scale**

![OmniBind Overview](assets/OmniBind_GA.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8-3.11](https://img.shields.io/badge/python-3.8--3.11-blue.svg)](https://www.python.org/downloads/)

OmniBind is a unified framework for predicting compound-protein interactions (CPIs) that simultaneously predicts four binding affinity metrics (Ki, Kd, IC50, EC50) using both amino acid sequences and 3Di structural alphabet sequences via adaptive gated fusion.

## System Requirements

### Software Dependencies

- Python 3.8 to 3.11 (the paper experiments used Python 3.8.12)
- PyTorch >= 2.0.0 (CUDA-enabled recommended)
- RDKit >= 2022.9
- hydra-core >= 1.3.0
- See `requirements.txt` for the full, version-pinned list

### Tested Environment

- Red Hat Enterprise Linux 8.6, Python 3.8.12, CUDA 12.1 / PyTorch 2.1.0 (training
  and all results in the paper)
- macOS 26.6 (Apple silicon), Python 3.8.20 and 3.11.15, PyTorch 2.1.0 on the CPU
  (installation, the demo below and the notebooks)

### Hardware Requirements

- **Minimum**: 1 NVIDIA GPU with >= 12 GB VRAM (e.g., RTX 3080)
- **Recommended**: NVIDIA A100 (40 GB) or equivalent for training (the paper used 8 × A100 40 GB; 100 epochs take approximately 60 hours)
- A GPU is required for training only. Inference, including every notebook in
  `notebooks/`, runs on the CPU in seconds
- Optional: Multiple GPUs with Horovod for distributed training

## Installation Guide

### 1. Create an environment

```bash
conda create -n omnibind python=3.11 -y
conda activate omnibind
```

Any environment manager works; `python -m venv` is equivalent.

Python 3.8 to 3.11 are supported; the pinned dependencies have no wheels for Python
3.12 or newer. We recommend 3.11, which we verified gives predictions identical to the
Python 3.8.12 used for the paper. Python 3.8 reached end of life in October 2024, so
use it only to match the original environment exactly.

### 2. Install OmniBind and its dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

This installs the CPU build of PyTorch, which is all that inference needs. RDKit is
included in `requirements.txt`, so it does not have to be installed separately.

### 3. (Optional) Install the CUDA build of PyTorch for training

```bash
# For CUDA 12.1 (adjust for your CUDA version)
pip install --force-reinstall torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Run this **after** step 2: installing `requirements.txt` afterwards would replace the
CUDA build with the CPU one.

### 4. (Optional) Install Horovod for distributed training

```bash
HOROVOD_WITH_PYTORCH=1 pip install horovod==0.28.1
```

Typical installation time: **5-15 minutes** on a standard desktop computer.

## Demo

### Quick inference with sample data

`data/sample/sample_proteins.csv` ships 102 human proteins, each with the two
sequences the model needs: the amino acid sequence and the matching 3Di structural
alphabet sequence. The example below predicts the four affinity endpoints for
imatinib against ABL1, its clinical target.

Download the weights first (see `checkpoints/README.md`), then run:

```python
import pandas as pd
from omegaconf import OmegaConf

from omnibind.predict import load_model, predict_single

cfg = OmegaConf.load("configs/default.yaml")
cfg.model.type = "aa3di_gmf"
cfg.training.device = "cpu"      # or "cuda"

model = load_model(cfg, "checkpoints/application.pth")

proteins = pd.read_csv("data/sample/sample_proteins.csv").set_index("gene_name")
target = proteins.loc["ABL1"]

result = predict_single(
    smiles="CN1CCN(Cc2ccc(cc2)C(=O)Nc2ccc(C)c(Nc3nccc(n3)-c3cccnc3)c2)CC1",  # Imatinib
    aa_sequence=target["aa_sequence"],
    sa_sequence=target["3di_sequence"],
    model=model,
    cfg=cfg,
)
print(f"Predicted pKi:   {result['predicted_ki']:.4f}")
print(f"Predicted pKd:   {result['predicted_kd']:.4f}")
print(f"Predicted pIC50: {result['predicted_ic50']:.4f}")
print(f"Predicted pEC50: {result['predicted_ec50']:.4f}")
```

**Expected output**:

```
Predicted pKi:   5.80
Predicted pKd:   5.17
Predicted pIC50: 5.67
Predicted pEC50: 5.43
```

Values are on the p-scale (`-log10(M)`) and typically fall between 3 and 12. The last
decimals may differ slightly between platforms.

**Demo run time**: well under a second per compound-protein pair on a laptop CPU.

### Tutorials

The notebooks in `notebooks/` reproduce the three application analyses of the paper
on the sample data. Each runs end to end on a CPU in a few seconds.

| Notebook | Analysis |
|----------|----------|
| `notebooks/01_drug_repositioning.ipynb` | Rank a compound library against one target |
| `notebooks/02_attention_map.ipynb` | Residue-level attention for a compound-protein pair |
| `notebooks/03_offtarget_screening.ipynb` | Rank a protein set for one compound |

```bash
pip install notebook
jupyter notebook notebooks/
```

## Usage

### Training

```bash
# Distributed training with Horovod (8 GPUs, as in the paper)
horovodrun -np 8 python scripts/train.py \
    dataset.data_dir=./data/processed/seed42

# Single GPU training
python scripts/train.py \
    dataset.data_dir=./data/processed/seed42
```

The default configuration (`configs/default.yaml`) uses the training settings of the paper: RAdam optimizer (learning rate 1e-4, weight decay 1e-4) with a constant learning rate, 100 epochs, batch size 2 per GPU with 64 gradient accumulation steps, and mixed precision (AMP).

- With Horovod, the learning rate is scaled linearly by the number of GPUs (8e-4 on 8 GPUs), and the effective batch size is 2 × 64 × 8 = 1,024. On a single GPU, the effective batch size is 128 and the learning rate is not scaled.
- After each epoch, the checkpoint with the lowest validation RMSE averaged over the four endpoints is saved as `best_rmse_model_epoch*-.pth` in `out_dir`.

### Testing

```bash
python scripts/test.py \
    dataset.data_dir=./data/processed/seed42 \
    test.checkpoint_path=./checkpoints/benchmark/seed42.pth
```

`dataset.data_dir` must point at the split the checkpoint was trained on. The
distributed `benchmark/seed*.pth` files belong to the original splits, which cannot be
regenerated; see [A note on the data splits](#a-note-on-the-data-splits).

### Applications

#### Drug Repositioning (batch compound screening)

```bash
python scripts/drug_repositioning.py \
    test.checkpoint_path=./checkpoints/application.pth \
    test.smiles_file=compounds.csv \
    test.aa="<target_aa_sequence>" \
    test.sa="<target_3di_sequence>"
```

#### Off-Target Screening (protein screening)

```bash
python scripts/offtarget_screening.py \
    test.checkpoint_path=./checkpoints/application.pth \
    test.fixed_smiles="<compound_smiles>" \
    test.seq_pkl_path=proteins_seq.pkl \
    test.ss_pkl_path=proteins_ss.pkl
```

#### Attention Map Extraction

```bash
python scripts/attention_map.py \
    test.checkpoint_path=./checkpoints/application.pth \
    test.smiles="<compound_smiles>" \
    test.aa="<aa_sequence>" \
    test.sa="<3di_sequence>" \
    test.name="analysis_name"
```

### Input Formats

- **SMILES**: Standard SMILES notation for compounds
- **AA sequence**: Single-letter amino acid sequence (uppercase)
- **3Di sequence**: Foldseek 3Di structural alphabet sequence (20-letter alphabet), predicted from the amino acid sequence with ProstT5 (see `data/README.md`)

## Reproduction

To reproduce the paper experiments:

1. Download and preprocess BindingDB data (see `data/README.md`)
2. Train one model on each of the five data splits:

```bash
for seed in 42 123 369 777 2024; do
    horovodrun -np 8 python scripts/train.py \
        dataset.data_dir=./data/processed/seed${seed} \
        out_dir=./outputs/seed${seed}
done
```

The seed selects the data split (see `data/README.md`); model weights are initialized with the same fixed seed (`training.seed=0`) in all runs, as in the paper. OmniBind trains all four affinity metrics (Ki, Kd, IC50, EC50) simultaneously via multi-task learning.

3. Evaluate each trained model on the test set of its own split. Training writes a
   `best_rmse_model_epoch*-.pth` file whenever the validation RMSE improves, so the
   last one is the best checkpoint:

```bash
for seed in 42 123 369 777 2024; do
    ckpt=$(ls -v ./outputs/seed${seed}/best_rmse_model_epoch*-.pth | tail -1)
    python scripts/test.py \
        dataset.data_dir=./data/processed/seed${seed} \
        test.checkpoint_path=${ckpt} \
        out_dir=./outputs/seed${seed}
done
```

Results in the paper are reported as mean ± standard deviation over the five splits.

### A note on the data splits

The checkpoints distributed on Zenodo were trained on the original splits, and those
splits cannot be regenerated bit-for-bit. `split_data.py` now sorts the ligands before
sampling, so that a given seed always yields the same split; the original runs did not
sort, and the selection therefore depended on Python's per-process string hashing,
which is randomized unless `PYTHONHASHSEED` is fixed. Regenerated splits are
deterministic, but they are not the splits of the paper.

Consequently, **do not evaluate the distributed checkpoints on a regenerated split**:
compounds held out in the regenerated test set may have been in the original training
set, which inflates the scores. Retrain on the regenerated splits as in step 2 and
evaluate those models, as the commands above do; expect numbers close to, but not
identical to, the published ones. The distributed checkpoints are intended for
inference and for the application analyses, where no held-out split is involved.

## Directory Structure

```
OmniBind/
├── omnibind/                   # Core library
│   ├── __init__.py
│   ├── model.py                # Model architectures (4 variants)
│   ├── data_utils.py           # Dataset and collation
│   ├── featurization.py        # Molecular and protein featurization
│   ├── train.py                # Training loop (single/distributed)
│   ├── evaluate.py             # Test evaluation
│   ├── predict.py              # Inference interface
│   └── utils.py                # Utilities
├── configs/                    # Hydra configuration files
│   ├── default.yaml
│   ├── dataset/
│   └── optimizer/
├── scripts/                    # Entry point scripts
│   ├── train.py
│   ├── test.py
│   ├── attention_map.py
│   ├── drug_repositioning.py
│   └── offtarget_screening.py
├── notebooks/                  # Tutorials for the application analyses
│   ├── 01_drug_repositioning.ipynb
│   ├── 02_attention_map.ipynb
│   └── 03_offtarget_screening.ipynb
├── data/
│   ├── README.md               # Data download and preprocessing instructions
│   ├── sample/                 # Sample data for the demo and notebooks
│   │   ├── sample_proteins.csv     # 102 human proteins (AA + 3Di sequences)
│   │   ├── sample_compounds.csv    # 40 FDA-approved compounds
│   │   └── sample_cpi.csv          # Example compound-protein pairs
│   └── preprocessing/          # Data preprocessing scripts
├── checkpoints/
│   └── README.md               # Download instructions for pre-trained weights
├── assets/                     # Figures for README
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Model Variants

| Model | Description | Protein Input |
|-------|-------------|---------------|
| `aa` | AA-only | Amino acid sequence |
| `3di` | 3Di-only | 3Di structural alphabet |
| `aa3di` | Simple fusion | AA + 3Di (addition) |
| `aa3di_gmf` | **Gated fusion (default)** | AA + 3Di (adaptive gating) |

## Citation

```bibtex
@article{omnibind2025,
  title={Pan--Pharmacological Drug--Target Interaction Prediction with 3D--Informed Protein Encoding at Scale},
  author={Kawaharada, Akinori and Ito, Takumi and Shimizu, Hideyuki},
  journal={Preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
