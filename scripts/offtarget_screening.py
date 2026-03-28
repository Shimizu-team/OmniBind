"""Off-target screening: predict binding of one compound against multiple proteins.

Usage:
    python scripts/offtarget_screening.py \
        test.checkpoint_path=./checkpoints/best_model.pth \
        test.fixed_smiles="CN1CCN(...)CC1" \
        test.seq_pkl_path=proteins_seq.pkl \
        test.ss_pkl_path=proteins_ss.pkl
"""

import os
import pickle
import sys
import warnings

import pandas as pd
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omnibind.predict import load_model, batch_predict_proteins

warnings.filterwarnings('ignore')


def load_protein_data(seq_pkl_path: str, ss_pkl_path: str) -> dict:
    """Load protein sequences and 3Di structures from pickle files.

    Args:
        seq_pkl_path: Path to pickle file with {protein_id: aa_sequence}.
        ss_pkl_path: Path to pickle file with {protein_id: sa_sequence}.

    Returns:
        Dict of {protein_id: (aa_sequence, sa_sequence)}.
    """
    with open(seq_pkl_path, 'rb') as f:
        seq_dict = pickle.load(f)
    with open(ss_pkl_path, 'rb') as f:
        ss_dict = pickle.load(f)

    common_keys = set(seq_dict.keys()) & set(ss_dict.keys())
    protein_data = {uid: (seq_dict[uid], ss_dict[uid]) for uid in common_keys}
    print(f"Loaded {len(protein_data)} proteins with both sequence and structure data")
    return protein_data


def main(cfg: DictConfig):
    fixed_smiles = cfg.test.fixed_smiles
    print(f'Fixed compound SMILES: {fixed_smiles}')

    protein_data = load_protein_data(cfg.test.seq_pkl_path, cfg.test.ss_pkl_path)
    print(f'Screening {len(protein_data)} proteins against fixed compound')

    model = load_model(cfg, cfg.test.checkpoint_path)

    output_dir = os.path.join(cfg.out_dir, 'protein_screening')
    df = batch_predict_proteins(fixed_smiles, protein_data, model, cfg, output_dir)

    valid_df = df.dropna(subset=['predicted_ki'])

    print(f"\nTotal proteins processed: {len(protein_data)}")
    print(f"Successful predictions: {len(valid_df)}")
    print(f"Failed predictions: {len(protein_data) - len(valid_df)}")

    if len(valid_df) > 0:
        for col in ['predicted_ki', 'predicted_kd', 'predicted_ic50', 'predicted_ec50']:
            print(f"\n{col} - Mean: {valid_df[col].mean():.6f}, Std: {valid_df[col].std():.6f}")

        # Save top proteins (ranked by Ki)
        top_n = getattr(cfg.test, 'top_n_proteins', 200)
        df_sorted = valid_df.sort_values('predicted_ki', ascending=False)
        top_ids = df_sorted.head(top_n)['protein_id'].tolist()

        top_file = os.path.join(output_dir, f'top_{top_n}_proteins.txt')
        with open(top_file, 'w') as f:
            f.write(','.join(top_ids))
        print(f"\nTop {top_n} protein IDs saved to {top_file}")


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def entry(cfg: DictConfig) -> None:
    main(cfg=cfg)


if __name__ == "__main__":
    entry()
