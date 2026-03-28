"""Convert split pickle data to numpy arrays for training.

Produces flat npy files in the data directory with separate files for each
affinity label (ki, kd, ic50, ec50). Missing labels are filled with -1.

Usage:
    # Convert train split for seed42
    python data/preprocessing/convert_to_npy.py \
        --data_dir ./data/processed/seed42 \
        --source_dir ./data/processed \
        --output_suffix train

    # Convert valid and test splits
    python data/preprocessing/convert_to_npy.py \
        --data_dir ./data/processed/seed42 \
        --source_dir ./data/processed \
        --output_suffix valid

    python data/preprocessing/convert_to_npy.py \
        --data_dir ./data/processed/seed42 \
        --source_dir ./data/processed \
        --output_suffix test
"""

import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tape import TAPETokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from omnibind.featurization import mol_features_from_mol, encode_3di_sequence

import warnings
warnings.filterwarnings('ignore')


def seq_cat(prot: str, tokenizer: TAPETokenizer) -> list:
    return tokenizer.encode(prot)


def convert_to_numpy(data_dir: str, source_dir: str, output_suffix: str = 'train',
                     smiles_max_len: int = 500, seq_max_len: int = 1500):
    """Convert data to numpy format with flat directory output.

    Produces 8 npy files per split: compounds, adjancies, aas, sas,
    ki, kd, ic50, ec50. Missing labels are -1.

    Args:
        data_dir: Directory containing split pickle files (e.g., data/processed/seed42).
        source_dir: Directory containing id_2_seq.pkl and id_2_ss.pkl.
        output_suffix: File suffix (train/valid/test).
        smiles_max_len: Max SMILES length filter.
        seq_max_len: Max sequence length filter.
    """
    data_path = os.path.join(data_dir, f'BindingDB_all_affinities_{output_suffix}.pkl')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f'Loading data from {data_path}')
    start = time.time()
    df = pd.read_pickle(data_path)
    print(f'Loaded {len(df)} rows in {time.time() - start:.0f}s')

    df = df.fillna(-1)

    # Load protein ID mappings
    with open(os.path.join(source_dir, 'id_2_seq.pkl'), 'rb') as f:
        id2seq = pickle.load(f)
    with open(os.path.join(source_dir, 'id_2_ss.pkl'), 'rb') as f:
        id2ss = pickle.load(f)

    # Filter long sequences
    original_len = len(df)
    df = df[df['smiles'].str.len() <= smiles_max_len]
    df = df[df['sequences'].str.len() <= seq_max_len]
    filtered_out = original_len - len(df)
    print(f"Filtered {filtered_out} long samples, remaining: {len(df)}")

    df = df.reset_index(drop=True)

    # ========== Featurize all data ==========

    # Compounds
    print('Featurizing compounds...')
    start = time.time()
    compounds, adjancies = [], []
    for mol in tqdm(df.mol):
        atom_feature, adj = mol_features_from_mol(mol)
        compounds.append(atom_feature)
        adjancies.append(adj)
    print(f'  Done in {time.time() - start:.0f}s')

    # Amino acid sequences
    print('Featurizing AA sequences...')
    start = time.time()
    tokenizer = TAPETokenizer(vocab='iupac')
    aas = []
    for seq_id in tqdm(df.protein_id):
        prot = id2seq[seq_id]
        sequence = seq_cat(prot, tokenizer)
        with torch.no_grad():
            protein_embedding = torch.tensor([sequence], dtype=torch.int64)
        aas.append(protein_embedding.squeeze(0).numpy())
    print(f'  Done in {time.time() - start:.0f}s')

    # 3Di sequences
    print('Featurizing 3Di sequences...')
    start = time.time()
    sas = []
    for seq_id in tqdm(df.protein_id):
        sa = id2ss[seq_id]
        ss = encode_3di_sequence(sa)
        sas.append(ss)
    print(f'  Done in {time.time() - start:.0f}s')

    # Per-label values (each shape N×1, -1 for missing)
    label_names = ['Ki', 'Kd', 'IC50', 'EC50']
    label_arrays = {}
    for label in label_names:
        if label in df.columns:
            label_arrays[label] = np.array([np.array([float(v)]) for v in df[label]])
            valid_count = (df[label] != -1).sum()
            print(f'  {label}: {valid_count} valid samples')

    # Convert to numpy arrays
    compounds = np.array(compounds, dtype=object)
    adjancies = np.array(adjancies, dtype=object)
    aas = np.array(aas, dtype=object)
    sas = np.array(sas, dtype=object)

    del df
    gc.collect()

    # ========== Save flat npy files ==========

    print(f'Saving npy files to {data_dir}...')

    np.save(os.path.join(data_dir, f'compounds_{output_suffix}'), compounds)
    np.save(os.path.join(data_dir, f'adjancies_{output_suffix}'), adjancies)
    np.save(os.path.join(data_dir, f'aas_{output_suffix}'), aas)
    np.save(os.path.join(data_dir, f'sas_{output_suffix}'), sas)

    # Save per-label npy files (lowercase filenames)
    for label in label_names:
        if label in label_arrays:
            filename = f'{label.lower()}_{output_suffix}'
            np.save(os.path.join(data_dir, filename), label_arrays[label])

    gc.collect()
    print('Done!')


def main():
    parser = argparse.ArgumentParser(description='Convert data to numpy format')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing split pkl files (e.g., data/processed/seed42)')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Directory containing id_2_seq.pkl and id_2_ss.pkl')
    parser.add_argument('--output_suffix', type=str, default='train',
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--smiles_max_len', type=int, default=500)
    parser.add_argument('--seq_max_len', type=int, default=1500)
    args = parser.parse_args()

    convert_to_numpy(args.data_dir, args.source_dir, args.output_suffix,
                     args.smiles_max_len, args.seq_max_len)


if __name__ == '__main__':
    main()
