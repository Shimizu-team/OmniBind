"""Process raw BindingDB TSV into filtered pickle files.

Usage:
    python data/preprocessing/process_bindingdb.py \
        --input_path /path/to/BindingDB_All.tsv \
        --output_dir ./data/processed
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem


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
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10
        y = -np.log10(y * 1e-9)
    elif to_ == 'nM':
        pass

    if array_flag:
        return y[0]
    return y


def main():
    parser = argparse.ArgumentParser(description='Process BindingDB data')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to BindingDB_All.tsv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed pickle files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading BindingDB data...')
    bdb = pd.read_csv(args.input_path, sep='\t', usecols=[
        'Ligand SMILES', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
        'BindingDB Target Chain  Sequence',
        'PDB ID(s) of Target Chain',
        'Number of Protein Chains in Target (>1 implies a multichain complex)',
    ], dtype={'Ki (nM)': 'string', 'IC50 (nM)': 'string',
              'Kd (nM)': 'string', 'EC50 (nM)': 'string'})

    print(f'Raw rows: {len(bdb)}')

    # Filter single-chain targets
    bdb = bdb[bdb['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
    bdb = bdb[bdb['Ligand SMILES'].notnull()]
    bdb = bdb[bdb['BindingDB Target Chain  Sequence'].notnull()]

    bdb.rename(columns={
        'Ligand SMILES': 'smiles',
        'BindingDB Target Chain  Sequence': 'sequences',
        'PDB ID(s) of Target Chain': 'PDB_ids',
        'Ki (nM)': 'Ki', 'IC50 (nM)': 'IC50',
        'Kd (nM)': 'Kd', 'EC50 (nM)': 'EC50',
    }, inplace=True)

    idx_strs = ['Ki', 'IC50', 'Kd', 'EC50']

    # Keep rows with at least one affinity value
    bdb_want = bdb.dropna(thresh=1, subset=idx_strs)

    # Convert string values to float
    for y in idx_strs:
        bdb_want[y] = bdb_want[y].str.replace('>', '', regex=False)
        bdb_want[y] = bdb_want[y].str.replace('<', '', regex=False)
        bdb_want[y] = bdb_want[y].astype('float32')

    bdb_want['mean'] = bdb_want[idx_strs].mean(axis=1, skipna=True)
    bdb_want = bdb_want[bdb_want['mean'] <= 10000000.0]

    # Convert nM to pK values
    bdb_want.reset_index(inplace=True, drop=True)
    for y in idx_strs:
        bdb_want[y] = convert_y_unit(np.array(bdb_want[y]), 'nM', 'p')
    bdb_want['mean'] = bdb_want[idx_strs].mean(axis=1, skipna=True)

    # Clean sequences
    bdb_want['sequences'] = bdb_want['sequences'].str.upper()
    bdb_want = bdb_want.loc[~(bdb_want['sequences'].str.contains('-'))]

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

    # Split by label and save
    bdb_want = bdb_want.rename(columns={'mean': 'binds'})
    bdb_want.to_pickle(os.path.join(args.output_dir, 'BindingDB_all_affinities.pkl'))

    for label in idx_strs:
        subset = bdb_want[bdb_want[label].notna()].reset_index(drop=True)
        subset = subset[['smiles', 'mol', 'sequences', label, 'PDB_ids']].rename(
            columns={label: 'binds'})
        subset.to_pickle(os.path.join(args.output_dir, f'BindingDB_{label}.pkl'))
        print(f'{label}: {len(subset)} rows')


if __name__ == '__main__':
    main()
