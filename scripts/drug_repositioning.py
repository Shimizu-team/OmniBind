"""Drug repositioning: batch prediction of multiple compounds against a single target.

Usage:
    python scripts/drug_repositioning.py \
        test.checkpoint_path=./checkpoints/best_model.pth \
        test.smiles_file=compounds.csv \
        test.aa="MSHHWGY..." \
        test.sa="DAFCDPP..."
"""

import os
import sys
import warnings

import pandas as pd
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omnibind.predict import load_model, batch_predict_compounds

warnings.filterwarnings('ignore')


def main(cfg: DictConfig):
    smiles_df = pd.read_csv(cfg.test.smiles_file)
    aa = cfg.test.aa
    sa = cfg.test.sa

    print(f'Processing {len(smiles_df)} compounds')
    print(f'Target protein AA sequence length: {len(aa)}')
    print(f'Target protein 3Di sequence length: {len(sa)}')

    model = load_model(cfg, cfg.test.checkpoint_path)

    output_dir = os.path.join(cfg.out_dir, 'batch_predictions')
    df = batch_predict_compounds(smiles_df, aa, sa, model, cfg, output_dir)

    valid_df = df.dropna(subset=['predicted_ki'])

    print(f"\nTotal compounds processed: {len(smiles_df)}")
    print(f"Successful predictions: {len(valid_df)}")
    print(f"Failed predictions: {len(smiles_df) - len(valid_df)}")

    if len(valid_df) > 0:
        for col in ['predicted_ki', 'predicted_kd', 'predicted_ic50', 'predicted_ec50']:
            print(f"\n{col} - Mean: {valid_df[col].mean():.6f}, Std: {valid_df[col].std():.6f}")


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def entry(cfg: DictConfig) -> None:
    main(cfg=cfg)


if __name__ == "__main__":
    entry()
