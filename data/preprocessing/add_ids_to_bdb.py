"""Add protein_id column to BindingDB pickle files.

Maps amino acid sequences to integer IDs using seq_2_id.pkl,
then overwrites the pickle files with the added protein_id column.

Usage:
    python data/preprocessing/add_ids_to_bdb.py \
        --data_dir ./data/processed
"""

import argparse
import os
import pickle

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Add protein_id column to BindingDB pickle files')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed pickle files and seq_2_id.pkl')
    args = parser.parse_args()

    seq2id_path = os.path.join(args.data_dir, 'seq_2_id.pkl')
    with open(seq2id_path, 'rb') as f:
        seq2id = pickle.load(f)

    pkl_names = [
        'BindingDB_all_affinities.pkl',
        'BindingDB_Ki.pkl',
        'BindingDB_Kd.pkl',
        'BindingDB_IC50.pkl',
        'BindingDB_EC50.pkl',
    ]

    for name in pkl_names:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            print(f'Skipping {name} (not found)')
            continue

        df = pd.read_pickle(path)
        df['protein_id'] = df['sequences'].map(lambda x: seq2id[x])
        df.to_pickle(path)
        print(f'Added protein_id to {name} ({len(df)} rows)')

    # Reorder columns for all_affinities
    all_path = os.path.join(args.data_dir, 'BindingDB_all_affinities.pkl')
    if os.path.exists(all_path):
        df = pd.read_pickle(all_path)
        cols = ['smiles', 'mol', 'sequences', 'protein_id', 'binds', 'Ki', 'Kd', 'IC50', 'EC50']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        df.to_pickle(all_path)
        print(f'Reordered columns in BindingDB_all_affinities.pkl')

    print('Done.')


if __name__ == '__main__':
    main()
