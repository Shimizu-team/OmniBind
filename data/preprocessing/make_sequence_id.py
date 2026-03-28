"""Create protein sequence ID mappings and FASTA file for Foldseek.

Reads the processed BindingDB pickle, extracts unique amino acid sequences,
and creates:
  - id_2_seq.pkl: {int_id: aa_sequence}
  - seq_2_id.pkl: {aa_sequence: int_id}
  - bdb_sequences.fasta: FASTA file for Foldseek 3Di generation

Usage:
    python data/preprocessing/make_sequence_id.py \
        --data_dir ./data/processed
"""

import argparse
import os
import pickle

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Create protein sequence ID mappings and FASTA for Foldseek')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing BindingDB_all_affinities.pkl')
    args = parser.parse_args()

    pkl_path = os.path.join(args.data_dir, 'BindingDB_all_affinities.pkl')
    df = pd.read_pickle(pkl_path)

    seq_uniq = list(df['sequences'].unique())
    print(f'Unique sequences: {len(seq_uniq)}')

    id_2_seq = {}
    seq_2_id = {}
    for i, seq in enumerate(seq_uniq):
        id_2_seq[i] = seq
        seq_2_id[seq] = i

    # Save dictionaries
    with open(os.path.join(args.data_dir, 'id_2_seq.pkl'), 'wb') as f:
        pickle.dump(id_2_seq, f)
    with open(os.path.join(args.data_dir, 'seq_2_id.pkl'), 'wb') as f:
        pickle.dump(seq_2_id, f)

    # Write FASTA file for Foldseek
    fasta_path = os.path.join(args.data_dir, 'bdb_sequences.fasta')
    with open(fasta_path, 'w') as f:
        for i, seq in enumerate(seq_uniq):
            f.write(f">{i}\n")
            f.write(f"{seq}\n")

    print(f'Saved id_2_seq.pkl, seq_2_id.pkl, and bdb_sequences.fasta to {args.data_dir}')


if __name__ == '__main__':
    main()
