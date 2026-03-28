"""Inference module for OmniBind models.

Supports single compound-protein pair prediction and batch prediction
for drug repositioning and off-target screening applications.
"""

import gc
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tape import TAPETokenizer
from omegaconf import DictConfig

from omnibind.model import build_model
from omnibind.featurization import (
    mol_features_from_smiles, encode_aa_sequence, encode_3di_sequence, NUM_ATOM_FEAT,
)


def _pack_single(compounds, adjs, aas, sas, atom_dim: int = NUM_ATOM_FEAT):
    """Pack a list of single-sample features into padded tensors.

    This is a lightweight collation for inference (no DataLoader).
    """
    N = len(compounds)
    atoms_len = max(c.shape[0] for c in compounds) + 1
    aas_len = max(a.shape[0] for a in aas)
    sas_len = max(s.shape[0] for s in sas)

    atom_num = [c.shape[0] + 1 for c in compounds]
    aa_num = [a.shape[0] for a in aas]
    sa_num = [s.shape[0] for s in sas]

    compounds_new = torch.zeros((N, atoms_len, atom_dim))
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
        aas_new[i, :aa.shape[0]] = torch.from_numpy(aa)

    sas_new = torch.zeros((N, sas_len), dtype=torch.int64)
    for i, sa in enumerate(sas):
        sas_new[i, :sa.shape[0]] = torch.from_numpy(sa)

    return compounds_new, adjs_new, aas_new, sas_new, atom_num, aa_num, sa_num


def load_model(cfg: DictConfig, checkpoint_path: str, device: str = None):
    """Load a trained OmniBind model from checkpoint.

    Args:
        cfg: Model configuration.
        checkpoint_path: Path to .pth checkpoint file.
        device: Target device (defaults to cfg.training.device).

    Returns:
        Loaded model in eval mode.
    """
    if device is None:
        device = cfg.training.device

    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_single(
    smiles: str,
    aa_sequence: str,
    sa_sequence: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    tokenizer: TAPETokenizer = None,
) -> dict:
    """Predict binding affinity for a single compound-protein pair.

    Args:
        smiles: SMILES string of the compound.
        aa_sequence: Amino acid sequence of the target protein.
        sa_sequence: 3Di structural alphabet sequence of the target protein.
        model: Trained OmniBind model.
        cfg: Configuration.
        tokenizer: TAPETokenizer (created if None).

    Returns:
        Dict with 'smiles', 'predicted_ki', 'predicted_kd',
        'predicted_ic50', 'predicted_ec50' keys.
    """
    if tokenizer is None:
        tokenizer = TAPETokenizer(vocab='iupac')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    atom_feature, adj = mol_features_from_smiles(canonical_smiles)
    aa_processed = encode_aa_sequence(aa_sequence, tokenizer)
    sa_processed = encode_3di_sequence(sa_sequence)

    compounds, adjs, aas, sas, atom_num, aa_num, sa_num = _pack_single(
        [atom_feature], [adj], [aa_processed], [sa_processed])

    device = cfg.training.device
    model.eval()
    with torch.no_grad():
        pred_ki, pred_kd, pred_ic50, pred_ec50 = model(
            compounds.to(device), adjs.to(device),
            aas.to(device), sas.to(device),
            atom_num, aa_num, sa_num,
        )

    return {
        'smiles': canonical_smiles,
        'predicted_ki': float(pred_ki.cpu().numpy().squeeze()),
        'predicted_kd': float(pred_kd.cpu().numpy().squeeze()),
        'predicted_ic50': float(pred_ic50.cpu().numpy().squeeze()),
        'predicted_ec50': float(pred_ec50.cpu().numpy().squeeze()),
    }


def batch_predict_compounds(
    smiles_df: pd.DataFrame,
    aa_sequence: str,
    sa_sequence: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Predict binding for multiple compounds against one protein target.

    Used for drug repositioning: screen a library of compounds against a target.

    Args:
        smiles_df: DataFrame with 'smiles' column (and optionally 'id' column).
        aa_sequence: Target protein amino acid sequence.
        sa_sequence: Target protein 3Di sequence.
        model: Trained model.
        cfg: Configuration.
        output_dir: If provided, save CSV results here.

    Returns:
        DataFrame with prediction results.
    """
    tokenizer = TAPETokenizer(vocab='iupac')
    results = []

    for i, row in smiles_df.iterrows():
        smiles = row['smiles']
        compound_id = row.get('id', row.get('zinc_id', str(i)))

        try:
            result = predict_single(smiles, aa_sequence, sa_sequence, model, cfg, tokenizer)
            results.append({
                'ID': compound_id,
                'SMILES': result['smiles'],
                'predicted_ki': result['predicted_ki'],
                'predicted_kd': result['predicted_kd'],
                'predicted_ic50': result['predicted_ic50'],
                'predicted_ec50': result['predicted_ec50'],
            })
        except Exception as e:
            print(f"Error processing compound {compound_id}: {e}")
            results.append({
                'ID': compound_id,
                'SMILES': smiles,
                'predicted_ki': None,
                'predicted_kd': None,
                'predicted_ic50': None,
                'predicted_ec50': None,
            })

    df = pd.DataFrame(results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        valid_df = df.dropna(subset=['predicted_ki'])
        df_sorted = valid_df.sort_values('predicted_ki', ascending=False).reset_index(drop=True)
        df_sorted.to_csv(os.path.join(output_dir, 'predictions_sorted.csv'), index=False)
        df.to_csv(os.path.join(output_dir, 'predictions_all.csv'), index=False)
        print(f"Results saved to: {output_dir}")

    return df


def batch_predict_proteins(
    smiles: str,
    protein_data: dict,
    model: torch.nn.Module,
    cfg: DictConfig,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Predict binding for one compound against multiple protein targets.

    Used for off-target screening: identify which proteins a compound may bind.

    Args:
        smiles: SMILES string of the compound.
        protein_data: Dict of {protein_id: (aa_sequence, sa_sequence)}.
        model: Trained model.
        cfg: Configuration.
        output_dir: If provided, save CSV results here.

    Returns:
        DataFrame with prediction results.
    """
    tokenizer = TAPETokenizer(vocab='iupac')
    results = []

    for protein_id, (aa_seq, sa_seq) in protein_data.items():
        try:
            result = predict_single(smiles, aa_seq, sa_seq, model, cfg, tokenizer)
            results.append({
                'protein_id': protein_id,
                'aa_length': len(aa_seq),
                'predicted_ki': result['predicted_ki'],
                'predicted_kd': result['predicted_kd'],
                'predicted_ic50': result['predicted_ic50'],
                'predicted_ec50': result['predicted_ec50'],
            })
        except Exception as e:
            print(f"Error processing protein {protein_id}: {e}")
            results.append({
                'protein_id': protein_id,
                'aa_length': len(aa_seq),
                'predicted_ki': None,
                'predicted_kd': None,
                'predicted_ic50': None,
                'predicted_ec50': None,
            })

    df = pd.DataFrame(results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        valid_df = df.dropna(subset=['predicted_ki'])
        df_sorted = valid_df.sort_values('predicted_ki', ascending=False).reset_index(drop=True)
        df_sorted.to_csv(os.path.join(output_dir, 'protein_screening_sorted.csv'), index=False)
        df.to_csv(os.path.join(output_dir, 'protein_screening_all.csv'), index=False)
        print(f"Results saved to: {output_dir}")

    return df
