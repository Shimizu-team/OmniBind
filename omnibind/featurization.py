"""Molecular and protein featurization utilities.

Converts SMILES strings to atom feature matrices and adjacency matrices,
and protein sequences to token ID arrays.
"""

from typing import Optional

import numpy as np
from rdkit import Chem

NUM_ATOM_FEAT = 35

# Amino acid vocabulary of tape-proteins 0.5 (tape/tokenizers.py, IUPAC_VOCAB).
# It is reproduced here so that OmniBind does not depend on tape-proteins, whose
# own dependency chain no longer installs on current Python versions. The token
# IDs must not be changed: the trained embedding layer depends on them.
IUPAC_VOCAB = {
    '<pad>': 0, '<mask>': 1, '<cls>': 2, '<sep>': 3, '<unk>': 4,
    'A': 5, 'B': 6, 'C': 7, 'D': 8, 'E': 9, 'F': 10, 'G': 11, 'H': 12,
    'I': 13, 'K': 14, 'L': 15, 'M': 16, 'N': 17, 'O': 18, 'P': 19, 'Q': 20,
    'R': 21, 'S': 22, 'T': 23, 'U': 24, 'V': 25, 'W': 26, 'X': 27, 'Y': 28,
    'Z': 29,
}

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


class ProteinTokenizer:
    """Amino acid tokenizer over the IUPAC vocabulary.

    Produces the same token IDs as ``TAPETokenizer(vocab='iupac')`` of
    tape-proteins 0.5: one token per character, wrapped in ``<cls>``/``<sep>``.
    """

    def __init__(self) -> None:
        self.vocab = dict(IUPAC_VOCAB)

    def encode(self, sequence: str) -> np.ndarray:
        """Encode an amino acid sequence to token IDs.

        Args:
            sequence: Amino acid sequence string (uppercase one-letter codes).

        Returns:
            1D numpy array of token IDs, of length ``len(sequence) + 2``.

        Raises:
            KeyError: If the sequence contains a character outside the vocabulary.
        """
        tokens = ['<cls>'] + list(sequence) + ['<sep>']
        try:
            token_ids = [self.vocab[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(f"Unrecognized token: '{exc.args[0]}'") from None
        return np.array(token_ids, dtype=np.int64)


def encode_aa_sequence(sequence: str,
                       tokenizer: Optional[ProteinTokenizer] = None) -> np.ndarray:
    """Encode amino acid sequence to token IDs.

    Args:
        sequence: Amino acid sequence string.
        tokenizer: ProteinTokenizer instance (created if None).

    Returns:
        1D numpy array of token IDs.
    """
    if tokenizer is None:
        tokenizer = ProteinTokenizer()
    return tokenizer.encode(sequence)


def encode_3di_sequence(sa_sequence: str) -> np.ndarray:
    """Convert 3Di structural alphabet string to index array.

    Args:
        sa_sequence: String of 3Di alphabet characters.

    Returns:
        1D numpy array of integer indices.
    """
    return np.array([ALPHABET_TO_INDEX[a] for a in sa_sequence])
