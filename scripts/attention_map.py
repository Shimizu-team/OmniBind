"""Extract attention maps from OmniBind models.

Usage:
    python scripts/attention_map.py \
        test.checkpoint_path=./checkpoints/best_model.pth \
        test.smiles="CN1CCN(Cc2ccc(cc2)C(=O)Nc2ccc(C)c(Nc3nccc(n3)-c3cccnc3)c2)CC1" \
        test.aa="MSHHWGY..." \
        test.sa="DAFCDPP..." \
        test.name="imatinib"
"""

import gc
import os
import pickle
import sys
import warnings

import numpy as np
import torch
from rdkit import Chem
from tape import TAPETokenizer
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omnibind.model import build_model
from omnibind.featurization import (
    mol_features_from_smiles, encode_aa_sequence, encode_3di_sequence,
)
from omnibind.predict import _pack_single

warnings.filterwarnings('ignore')


def main(cfg: DictConfig):
    smiles = cfg.test.smiles
    aa = cfg.test.aa
    sa = cfg.test.sa

    print(f'SMILES: {smiles}')

    # Canonicalize SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    smiles = Chem.MolToSmiles(mol, canonical=True)

    # Featurize
    atom_feature, adj = mol_features_from_smiles(smiles)
    tokenizer = TAPETokenizer(vocab='iupac')
    aa_processed = encode_aa_sequence(aa, tokenizer)
    sa_processed = encode_3di_sequence(sa)

    compounds, adjs, aas, sas, atom_num, aa_num, sa_num = _pack_single(
        [atom_feature], [adj], [aa_processed], [sa_processed])

    # Build and load model
    print('Building model')
    model = build_model(cfg)

    checkpoint = torch.load(cfg.test.checkpoint_path, map_location=cfg.training.device)
    print(f'Loading model from {cfg.test.checkpoint_path}')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(cfg.training.device)
    model.eval()

    with torch.no_grad():
        compounds = compounds.to(cfg.training.device)
        adjs = adjs.to(cfg.training.device)
        aas = aas.to(cfg.training.device)
        sas = sas.to(cfg.training.device)

        attn_results = model.get_attn_maps(
            compounds, adjs, aas, sas, atom_num, aa_num, sa_num)
        pred_ki, pred_kd, pred_ic50, pred_ec50 = model(
            compounds, adjs, aas, sas, atom_num, aa_num, sa_num)

    print(f'Predicted Ki: {float(pred_ki.cpu().numpy().squeeze()):.4f}')
    print(f'Predicted Kd: {float(pred_kd.cpu().numpy().squeeze()):.4f}')
    print(f'Predicted IC50: {float(pred_ic50.cpu().numpy().squeeze()):.4f}')
    print(f'Predicted EC50: {float(pred_ec50.cpu().numpy().squeeze()):.4f}')

    # Extract and save attention maps
    maps_comp_pro = attn_results[-1]  # Last element is always the attention maps
    maps_comp_pro_np = []
    for i, attn_map in enumerate(maps_comp_pro):
        attn_map = attn_map.cpu().numpy().squeeze(axis=0)
        print(f'Attention map layer {i} shape: {attn_map.shape}')
        maps_comp_pro_np.append(attn_map)

    out_dir = os.path.join(cfg.out_dir, 'attention_maps', cfg.test.name or 'default')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'maps_comp_pro.pkl'), 'wb') as f:
        pickle.dump(maps_comp_pro_np, f)

    print(f'Attention maps saved to {out_dir}')


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def entry(cfg: DictConfig) -> None:
    main(cfg=cfg)


if __name__ == "__main__":
    entry()
