"""Process raw BindingDB TSV into filtered pickle files.

Usage:
    python data/preprocessing/process_bindingdb.py \
        --input_path /path/to/BindingDB_All.tsv \
        --output_dir ./data/processed

The loading and cleaning steps are also reused by make_temporal_set.py.
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem

LABELS = ['Ki', 'IC50', 'Kd', 'EC50']

# The sequence column is spelled with two spaces in some BindingDB releases
# (e.g. May 2023) and with one space in others (e.g. November 2024).
SEQUENCE_COLUMNS = (
    'BindingDB Target Chain  Sequence',
    'BindingDB Target Chain Sequence',
)


def convert_y_unit(y, from_='nM', to_='p'):
    """Convert affinity units (nM <-> pKi/pKd/pIC50/pEC50)."""
    array_flag = False
    if isinstance(y, (int, float)):
        y = np.array([y])
        array_flag = True
    y = y.astype(float)

    if from_ == 'nM':
        pass
    elif from_ == 'p':
        y = 10 ** (-y) / 1e-9

    if to_ == 'p':
        y = -np.log10(y * 1e-9 + 1e-10)
    elif to_ == 'nM':
        pass

    if array_flag:
        return y[0]
    return y


def find_sequence_column(input_path: str) -> str:
    """Return the target-sequence column name used by this BindingDB release.

    Args:
        input_path: Path to a BindingDB TSV file.

    Returns:
        The column name holding the target chain sequence.

    Raises:
        ValueError: If no known sequence column is present.
    """
    header = pd.read_csv(input_path, sep='\t', nrows=0)
    for column in SEQUENCE_COLUMNS:
        if column in header.columns:
            return column
    raise ValueError(
        f'No target sequence column found in {input_path}. '
        f'Expected one of: {SEQUENCE_COLUMNS}')


def load_bindingdb_tsv(input_path: str) -> pd.DataFrame:
    """Load a BindingDB TSV and keep single-chain entries with SMILES and sequence.

    Args:
        input_path: Path to a BindingDB TSV file.

    Returns:
        DataFrame with renamed columns (smiles, sequences, PDB_ids, Ki, IC50, Kd, EC50).
    """
    sequence_column = find_sequence_column(input_path)
    chain_column = (
        'Number of Protein Chains in Target (>1 implies a multichain complex)')

    bdb = pd.read_csv(input_path, sep='\t', usecols=[
        'Ligand SMILES', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
        sequence_column, 'PDB ID(s) of Target Chain', chain_column,
    ], dtype={'Ki (nM)': 'string', 'IC50 (nM)': 'string',
              'Kd (nM)': 'string', 'EC50 (nM)': 'string'})

    print(f'Raw rows: {len(bdb)}')

    bdb = bdb[bdb[chain_column] == 1.0]
    bdb = bdb[bdb['Ligand SMILES'].notnull()]
    bdb = bdb[bdb[sequence_column].notnull()]
    print(f'Single-chain rows with SMILES and sequence: {len(bdb)}')

    bdb = bdb.rename(columns={
        'Ligand SMILES': 'smiles',
        sequence_column: 'sequences',
        'PDB ID(s) of Target Chain': 'PDB_ids',
        'Ki (nM)': 'Ki', 'IC50 (nM)': 'IC50',
        'Kd (nM)': 'Kd', 'EC50 (nM)': 'EC50',
    })
    return bdb


def clean_bindingdb(bdb: pd.DataFrame,
                    drop_sequences_with_digits: bool = False) -> pd.DataFrame:
    """Filter affinities, convert units and canonicalize SMILES.

    Args:
        bdb: DataFrame as returned by load_bindingdb_tsv.
        drop_sequences_with_digits: Also drop sequences containing digits. This
            was applied when building the temporal validation set, but not when
            building the training data, and is kept configurable so that both
            datasets can be reproduced as they were originally generated.

    Returns:
        Cleaned DataFrame with pK values and a 'binds' column (mean over labels).
    """
    # Keep rows with at least one affinity value
    bdb_want = bdb.dropna(thresh=1, subset=LABELS)

    # Convert string values to float
    for y in LABELS:
        bdb_want[y] = bdb_want[y].str.replace('>', '', regex=False)
        bdb_want[y] = bdb_want[y].str.replace('<', '', regex=False)
        bdb_want[y] = bdb_want[y].astype('float32')

    bdb_want['mean'] = bdb_want[LABELS].mean(axis=1, skipna=True)
    bdb_want = bdb_want[bdb_want['mean'] <= 10000000.0]

    # Convert nM to pK values
    bdb_want.reset_index(inplace=True, drop=True)
    for y in LABELS:
        bdb_want[y] = convert_y_unit(np.array(bdb_want[y]), 'nM', 'p')
    bdb_want['mean'] = bdb_want[LABELS].mean(axis=1, skipna=True)

    # Clean sequences
    bdb_want['sequences'] = bdb_want['sequences'].str.upper()
    bdb_want = bdb_want.loc[~(bdb_want['sequences'].str.contains('-'))]
    if drop_sequences_with_digits:
        bdb_want = bdb_want.loc[~(bdb_want['sequences'].str.contains(r'\d'))]

    # Canonicalize SMILES
    def convert_to_mol(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return mol
        except Exception:
            return None

    bdb_want['mol'] = bdb_want['smiles'].apply(convert_to_mol)
    bdb_want = bdb_want[bdb_want['mol'].notnull()]
    bdb_want['canonical_smiles'] = bdb_want['mol'].apply(
        lambda x: Chem.MolToSmiles(x, canonical=True))

    bdb_want = bdb_want.drop(columns=['smiles'])
    bdb_want = bdb_want.rename(columns={'canonical_smiles': 'smiles'})

    print(f'Processed rows: {len(bdb_want)}')
    print(f'Unique SMILES: {len(bdb_want.smiles.unique())}')
    print(f'Unique sequences: {len(bdb_want.sequences.unique())}')

    return bdb_want.rename(columns={'mean': 'binds'})


def save_pickles(bdb_want: pd.DataFrame, output_dir: str) -> None:
    """Write the combined pickle and one pickle per affinity label.

    Args:
        bdb_want: Cleaned DataFrame as returned by clean_bindingdb.
        output_dir: Directory to write the pickle files to.
    """
    os.makedirs(output_dir, exist_ok=True)
    bdb_want.to_pickle(os.path.join(output_dir, 'BindingDB_all_affinities.pkl'))

    for label in LABELS:
        subset = bdb_want[bdb_want[label].notna()].reset_index(drop=True)
        subset = subset[['smiles', 'mol', 'sequences', label, 'PDB_ids']].rename(
            columns={label: 'binds'})
        subset.to_pickle(os.path.join(output_dir, f'BindingDB_{label}.pkl'))
        print(f'{label}: {len(subset)} rows')


def main():
    parser = argparse.ArgumentParser(description='Process BindingDB data')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to BindingDB_All.tsv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed pickle files')
    args = parser.parse_args()

    print('Loading BindingDB data...')
    bdb = load_bindingdb_tsv(args.input_path)
    bdb_want = clean_bindingdb(bdb)
    save_pickles(bdb_want, args.output_dir)


if __name__ == '__main__':
    main()
