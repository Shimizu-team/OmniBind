"""Molecular and protein featurization utilities.

Converts SMILES strings to atom feature matrices and adjacency matrices,
and protein sequences to token ID arrays.
"""

import numpy as np
from rdkit import Chem
from tape import TAPETokenizer
import torch

NUM_ATOM_FEAT = 35

# 3Di structural alphabet to index mapping (Foldseek format)
ALPHABET_TO_INDEX = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
}


def one_of_k_encoding(x, allowable_set: list) -> list:
    """Strict one-hot encoding (raises if x not in set)."""
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set: list) -> list:
    """One-hot encoding that maps unknown values to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H: bool = False, use_chirality: bool = True) -> list:
    """Generate atom features (35-dim).

    Features: atom symbol (10), degree (8), formal charge (1),
    radical electrons (1), hybridization (6), aromatic (1),
    total H (5), chirality (3).
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6, 'other']
    hybridization_type = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]
    results = (
        one_of_k_encoding_unk(atom.GetSymbol(), symbol)
        + one_of_k_encoding_unk(atom.GetDegree(), degree)
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(atom.GetHybridization(), hybridization_type)
        + [atom.GetIsAromatic()]
    )  # 10+8+2+6+1 = 27

    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # +5 = 32

    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                       [atom.HasProp('_ChiralityPossible')]
        except Exception:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]  # +3 = 35

    return results


def adjacent_matrix(mol) -> np.ndarray:
    """Compute adjacency matrix from an RDKit mol object."""
    return np.array(Chem.GetAdjacencyMatrix(mol))


def mol_features_from_mol(mol) -> tuple:
    """Compute atom feature matrix and adjacency matrix from an RDKit Mol object.

    Args:
        mol: RDKit Mol object.

    Returns:
        Tuple of (atom_feat, adj_matrix) numpy arrays.
    """
    atom_feat = np.zeros((mol.GetNumAtoms(), NUM_ATOM_FEAT))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def mol_features_from_smiles(smiles: str) -> tuple:
    """Compute atom feature matrix and adjacency matrix from a SMILES string.

    Args:
        smiles: SMILES string.

    Returns:
        Tuple of (atom_feat, adj_matrix) numpy arrays.

    Raises:
        RuntimeError: If SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"SMILES cannot be parsed: {smiles}")
    return mol_features_from_mol(mol)


def encode_aa_sequence(sequence: str, tokenizer: TAPETokenizer = None) -> np.ndarray:
    """Encode amino acid sequence to token IDs using TAPE tokenizer.

    Args:
        sequence: Amino acid sequence string.
        tokenizer: TAPETokenizer instance (created if None).

    Returns:
        1D numpy array of token IDs.
    """
    if tokenizer is None:
        tokenizer = TAPETokenizer(vocab='iupac')
    token_ids = tokenizer.encode(sequence)
    with torch.no_grad():
        protein_embedding = torch.tensor([token_ids], dtype=torch.int64)
    return protein_embedding.squeeze(0).numpy()


def encode_3di_sequence(sa_sequence: str) -> np.ndarray:
    """Convert 3Di structural alphabet string to index array.

    Args:
        sa_sequence: String of 3Di alphabet characters.

    Returns:
        1D numpy array of integer indices.
    """
    return np.array([ALPHABET_TO_INDEX[a] for a in sa_sequence])
