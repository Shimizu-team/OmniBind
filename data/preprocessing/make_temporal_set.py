"""Build the temporal validation set from two BindingDB releases.

An interaction record is kept when its compound-protein pair is not present in
the earlier (training) release, i.e. the set consists of interaction pairs that
were newly deposited after the training data was collected. Compounds and
proteins themselves may already occur in the training data; only the pair is
new. Pairs are matched on the raw SMILES strings of the two releases, before
canonicalization, as in the original experiments.

The outputs use the same file names as process_bindingdb.py, so the remaining
steps of the pipeline (make_sequence_id.py, add_ids_to_bdb.py,
convert_to_npy.py) can be run on the output directory unchanged.

Usage:
    python data/preprocessing/make_temporal_set.py \
        --train_tsv /path/to/BindingDB_All_2023m5.tsv \
        --test_tsv /path/to/BindingDB_All_2024m11.tsv \
        --output_dir ./data/processed/temporal
"""

import argparse

import pandas as pd

from process_bindingdb import clean_bindingdb, load_bindingdb_tsv, save_pickles


def select_new_pairs(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Keep rows of the later release whose compound-protein pair is new.

    Args:
        train: Earlier release, as returned by load_bindingdb_tsv.
        test: Later release, as returned by load_bindingdb_tsv.

    Returns:
        Rows of the later release whose (smiles, sequences) pair does not occur
        in the earlier release.
    """
    print(f'Unique SMILES    - training release: {len(train.smiles.unique())}, '
          f'later release: {len(test.smiles.unique())}')
    print(f'Unique sequences - training release: {len(train.sequences.unique())}, '
          f'later release: {len(test.sequences.unique())}')

    train_pairs = set(zip(train['smiles'], train['sequences']))
    test_keys = list(zip(test['smiles'], test['sequences']))
    is_new = [key not in train_pairs for key in test_keys]

    new_pairs = test[is_new]
    print(f'Interaction pairs only in the later release: {len(new_pairs)}')
    print(f'  unique SMILES: {len(new_pairs.smiles.unique())}')
    print(f'  unique sequences: {len(new_pairs.sequences.unique())}')
    return new_pairs


def main():
    parser = argparse.ArgumentParser(
        description='Build the temporal validation set from two BindingDB releases')
    parser.add_argument('--train_tsv', type=str, required=True,
                        help='BindingDB TSV of the release used for training')
    parser.add_argument('--test_tsv', type=str, required=True,
                        help='BindingDB TSV of the later release')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the temporal validation pickles')
    args = parser.parse_args()

    print('Loading the training release...')
    train = load_bindingdb_tsv(args.train_tsv)
    print('Loading the later release...')
    test = load_bindingdb_tsv(args.test_tsv)

    new_pairs = select_new_pairs(train, test)
    del train, test

    # The temporal set additionally drops sequences containing digits, as in the
    # original experiments.
    cleaned = clean_bindingdb(new_pairs, drop_sequences_with_digits=True)
    save_pickles(cleaned, args.output_dir)


if __name__ == '__main__':
    main()
