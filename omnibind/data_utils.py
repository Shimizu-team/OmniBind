"""Dataset and collation utilities for CPI data."""

import numpy as np
import torch
from torch.utils.data import Dataset


class CPIDataset(Dataset):
    """Dataset for compound-protein interaction prediction.

    Stores pre-featurized numpy arrays for compounds, adjacency matrices,
    amino acid token IDs, 3Di token IDs, and four binding affinity labels
    (Ki, Kd, IC50, EC50). Missing labels are encoded as -1.
    """

    def __init__(self, compounds: np.ndarray, adjacencies: np.ndarray,
                 aas: np.ndarray, sas: np.ndarray,
                 ki: np.ndarray, kd: np.ndarray,
                 ic50: np.ndarray, ec50: np.ndarray):
        super().__init__()
        self.compounds = compounds
        self.adjacencies = adjacencies
        self.aas = aas
        self.sas = sas
        self.ki = ki
        self.kd = kd
        self.ic50 = ic50
        self.ec50 = ec50

    def __getitem__(self, i):
        return (self.compounds[i], self.adjacencies[i], self.aas[i], self.sas[i],
                self.ki[i], self.kd[i], self.ic50[i], self.ec50[i])

    def __len__(self):
        return len(self.ki)

    def save_memory(self, max_atom_len: int, max_aa_len: int) -> "CPIDataset":
        """Remove samples exceeding length thresholds to save GPU memory.

        Args:
            max_atom_len: Maximum number of atoms in a compound.
            max_aa_len: Maximum amino acid sequence length.

        Returns:
            self (modified in-place).
        """
        delete_inds = []
        for i, compound in enumerate(self.compounds):
            if compound.shape[0] > max_atom_len:
                delete_inds.append(i)
            elif self.aas[i].shape[0] > max_aa_len:
                delete_inds.append(i)

        self.compounds = np.delete(self.compounds, delete_inds, axis=0)
        self.adjacencies = np.delete(self.adjacencies, delete_inds, axis=0)
        self.aas = np.delete(self.aas, delete_inds, axis=0)
        self.sas = np.delete(self.sas, delete_inds, axis=0)
        self.ki = np.delete(self.ki, delete_inds, axis=0)
        self.kd = np.delete(self.kd, delete_inds, axis=0)
        self.ic50 = np.delete(self.ic50, delete_inds, axis=0)
        self.ec50 = np.delete(self.ec50, delete_inds, axis=0)
        return self


def collate_fn(batch: list) -> tuple:
    """Collate a batch of CPI samples with zero-padding.

    Pads compounds, adjacency matrices, amino acid sequences, and 3Di sequences
    to the maximum length in the batch. A virtual super-node is prepended to
    the compound graph.

    Args:
        batch: List of (compound, adj, aa, sa, ki, kd, ic50, ec50) tuples.

    Returns:
        Tuple of (compounds, adjs, aas, sas, kis, kds, ic50s, ec50s,
                  atom_num, aa_num, sa_num).
    """
    compounds, adjs, aas, sas, kis, kds, ic50s, ec50s = list(zip(*batch))

    atoms_len = 0
    aas_len = 0
    sas_len = 0
    N = len(compounds)

    atom_num = []
    for compound in compounds:
        atom_num.append(compound.shape[0] + 1)
        if compound.shape[0] >= atoms_len:
            atoms_len = compound.shape[0]
    atoms_len += 1

    aa_num = []
    for aa in aas:
        aa_num.append(aa.shape[0])
        if aa.shape[0] >= aas_len:
            aas_len = aa.shape[0]

    sa_num = []
    for sa in sas:
        sa_num.append(sa.shape[0])
        if sa.shape[0] >= sas_len:
            sas_len = sa.shape[0]

    compounds_new = torch.zeros((N, atoms_len, 35))
    for i, compound in enumerate(compounds):
        a_len = compound.shape[0]
        compounds_new[i, 1:a_len + 1, :] = torch.from_numpy(compound)

    adjs_new = torch.zeros((N, atoms_len, atoms_len))
    for i, adj in enumerate(adjs):
        adjs_new[i, 0, :] = 1
        adjs_new[i, :, 0] = 1
        a_len = adj.shape[0]
        adj_t = torch.from_numpy(adj) + torch.eye(a_len)
        adjs_new[i, 1:a_len + 1, 1:a_len + 1] = adj_t

    aas_new = torch.zeros((N, aas_len), dtype=torch.int64)
    for i, aa in enumerate(aas):
        a_len = aa.shape[0]
        aas_new[i, :a_len] = torch.from_numpy(aa)

    sas_new = torch.zeros((N, sas_len), dtype=torch.int64)
    for i, sa in enumerate(sas):
        a_len = sa.shape[0]
        sas_new[i, :a_len] = torch.from_numpy(sa)

    kis_new = torch.zeros(N, dtype=torch.float)
    for i, ki in enumerate(kis):
        kis_new[i] = torch.from_numpy(ki)

    kds_new = torch.zeros(N, dtype=torch.float)
    for i, kd in enumerate(kds):
        kds_new[i] = torch.from_numpy(kd)

    ic50s_new = torch.zeros(N, dtype=torch.float)
    for i, ic50 in enumerate(ic50s):
        ic50s_new[i] = torch.from_numpy(ic50)

    ec50s_new = torch.zeros(N, dtype=torch.float)
    for i, ec50 in enumerate(ec50s):
        ec50s_new[i] = torch.from_numpy(ec50)

    return (compounds_new, adjs_new, aas_new, sas_new,
            kis_new, kds_new, ic50s_new, ec50s_new,
            atom_num, aa_num, sa_num)
