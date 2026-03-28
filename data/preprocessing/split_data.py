"""Split BindingDB data into train/validation/test sets with multiple random seeds.

Uses ligand-based splitting to prevent data leakage: compounds that appear
in both positive and negative sets are split into test sets for evaluating
label-reversal robustness.

Usage:
    python data/preprocessing/split_data.py \
        --data_dir ./data/processed \
        --seeds 42 123 369 777 2024
"""

import argparse
import gc
import os
import random

import pandas as pd


def split_bdb_data(data_dir: str, seed: int, output_dir: str = None):
    """Split BindingDB data by ligand for a given random seed.

    Args:
        data_dir: Directory containing BindingDB_all_affinities.pkl.
        seed: Random seed for reproducibility.
        output_dir: Output directory (defaults to data_dir/seed{seed}).
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, f'seed{seed}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing with seed: {seed}")
    print(f"Output directory: {output_dir}")

    bdb_all = pd.read_pickle(os.path.join(data_dir, 'BindingDB_all_affinities.pkl'))

    bdb_pos = bdb_all.loc[bdb_all.binds >= 6.5]
    bdb_neg = bdb_all.loc[bdb_all.binds < 6.5]
    print(f'Positive: {len(bdb_pos)}, Negative: {len(bdb_neg)}')

    pos_lgds = set(bdb_pos.smiles)
    neg_lgds = set(bdb_neg.smiles)
    both_lgds = pos_lgds & neg_lgds
    print(f'Ligands in both: {len(both_lgds)}')

    # Split test set from label-reversal ligands
    random.seed(seed)
    shuffle_lgds = random.sample(list(both_lgds), len(both_lgds))
    tst_pos_lgds = set(shuffle_lgds[:50000])
    tst_neg_lgds = set(shuffle_lgds[50000:100000])

    tst_df_pos = bdb_pos.loc[bdb_pos['smiles'].apply(lambda x: x in tst_pos_lgds)]
    tst_df_neg = bdb_neg.loc[bdb_neg['smiles'].apply(lambda x: x in tst_neg_lgds)]
    tst_df = pd.concat([tst_df_neg, tst_df_pos], axis=0)
    print(f'Test: {len(tst_df)}')

    tr_df = bdb_all.drop(index=tst_df.index)
    tst_df = tst_df.reset_index(drop=True)
    tst_df.to_pickle(os.path.join(output_dir, 'BindingDB_all_affinities_test.pkl'))
    del tst_df
    gc.collect()

    # Split train/validation from remaining ligands
    lgds = list(set(tr_df.smiles))
    random.seed(seed)
    random.shuffle(lgds)
    va_lgds = set(lgds[:150000])
    tr_lgds = set(lgds[150000:])

    train = tr_df.loc[tr_df['smiles'].apply(lambda x: x in tr_lgds)].reset_index(drop=True)
    valid = tr_df.loc[tr_df['smiles'].apply(lambda x: x in va_lgds)].reset_index(drop=True)
    print(f'Train: {len(train)}, Valid: {len(valid)}')

    train.to_pickle(os.path.join(output_dir, 'BindingDB_all_affinities_train.pkl'))
    valid.to_pickle(os.path.join(output_dir, 'BindingDB_all_affinities_valid.pkl'))

    print(f"Data split completed for seed {seed}\n")


def main():
    parser = argparse.ArgumentParser(description='Split BindingDB data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing BindingDB_all_affinities.pkl')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 369, 777, 2024],
                        help='Random seeds (default: 42 123 369 777 2024)')
    args = parser.parse_args()

    for seed in args.seeds:
        split_bdb_data(args.data_dir, seed)


if __name__ == '__main__':
    main()
