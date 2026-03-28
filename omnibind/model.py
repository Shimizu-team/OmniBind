"""Model architectures for compound-protein interaction prediction.

Includes five model variants:
- CPIModelAA: amino acid sequence only
- CPIModel3Di: 3Di structural alphabet only
- CPIModelAA3Di: simple addition fusion of AA + 3Di
- CPIModelAA3DiwithCAF: cross-attention fusion of AA + 3Di
- CPIModelAA3DiwithGMF: gated mechanism fusion of AA + 3Di (default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_masks(
    atom_num: list,
    aa_num: list,
    sa_num: list,
    compound_max_len: int,
    aa_max_len: int,
    sa_max_len: int,
    device: torch.device,
) -> tuple:
    """Create padding masks for compound, amino acid, and 3Di sequences.

    Args:
        atom_num: Number of atoms per sample in batch.
        aa_num: Number of amino acid tokens per sample.
        sa_num: Number of 3Di tokens per sample.
        compound_max_len: Max compound length in batch.
        aa_max_len: Max amino acid length in batch.
        sa_max_len: Max 3Di length in batch.
        device: Target device.

    Returns:
        Tuple of (compound_mask, aa_mask, sa_mask) where 0 = real token, 1 = padding.
    """
    N = len(atom_num)
    compound_mask = torch.ones((N, compound_max_len), device=device)
    aa_mask = torch.ones((N, aa_max_len), device=device)
    sa_mask = torch.ones((N, sa_max_len), device=device)
    for i in range(N):
        compound_mask[i, :atom_num[i]] = 0
        aa_mask[i, :aa_num[i]] = 0
        sa_mask[i, :sa_num[i]] = 0
    return compound_mask, aa_mask, sa_mask


class EncoderofAA(nn.Module):
    """Amino acid sequence encoder using Transformer."""

    def __init__(self, cfg):
        super().__init__()
        self.hid_dim = cfg.model.encoder_aa.hid_dim
        self.n_layers = cfg.model.encoder_aa.n_layers
        self.n_head = cfg.model.encoder_aa.n_head
        self.dropout = cfg.model.encoder_aa.dropout

        self.aa_emb = nn.Embedding(num_embeddings=30, embedding_dim=self.hid_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_dim, nhead=self.n_head,
            dim_feedforward=self.hid_dim * 4, dropout=self.dropout,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """Encode amino acid token IDs.

        Args:
            src: [batch, aa_len] integer token IDs.
            src_mask: [batch, aa_len] padding mask (0 = real, 1 = pad).

        Returns:
            Tuple of (encoded, mask) where encoded is [aa_len, batch, hid_dim].
        """
        src = self.aa_emb(src)                      # [batch, aa_len, hid_dim]
        src = src.permute(1, 0, 2).contiguous()     # [aa_len, batch, hid_dim]
        src_mask = (src_mask == 1)
        protein = self.encoder(src, src_key_padding_mask=src_mask)
        return protein, src_mask


class Encoderof3Di(nn.Module):
    """3Di structural alphabet encoder using Transformer."""

    def __init__(self, cfg):
        super().__init__()
        self.hid_dim = cfg.model.encoder_sa.hid_dim
        self.n_layers = cfg.model.encoder_sa.n_layers
        self.n_head = cfg.model.encoder_sa.n_head
        self.dropout = cfg.model.encoder_sa.dropout

        self.sa_emb = nn.Embedding(num_embeddings=21, embedding_dim=self.hid_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_dim, nhead=self.n_head,
            dim_feedforward=self.hid_dim * 4, dropout=self.dropout,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """Encode 3Di token IDs.

        Args:
            src: [batch, sa_len] integer token IDs.
            src_mask: [batch, sa_len] padding mask.

        Returns:
            Tuple of (encoded, mask) where encoded is [sa_len, batch, hid_dim].
        """
        src = self.sa_emb(src)
        src = src.permute(1, 0, 2).contiguous()
        src_mask = (src_mask == 1)
        protein = self.encoder(src, src_key_padding_mask=src_mask)
        return protein, src_mask


class Decoder(nn.Module):
    """Compound feature decoder with cross-attention to protein representation."""

    def __init__(self, cfg):
        super().__init__()
        self.hid_dim = cfg.model.decoder.hid_dim
        self.n_layers = cfg.model.decoder.n_layers
        self.n_head = cfg.model.decoder.n_head
        self.dropout_rate = cfg.model.decoder.dropout
        self.hid_dim_fc = cfg.model.decoder.hid_dim_fc

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hid_dim, nhead=self.n_head,
            dim_feedforward=self.hid_dim * 4, dropout=self.dropout_rate,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        self.fc_1 = nn.Linear(self.hid_dim, self.hid_dim_fc)
        self.fc_ki = nn.Linear(self.hid_dim_fc, 1)
        self.fc_kd = nn.Linear(self.hid_dim_fc, 1)
        self.fc_ic50 = nn.Linear(self.hid_dim_fc, 1)
        self.fc_ec50 = nn.Linear(self.hid_dim_fc, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, trg: torch.Tensor, src: torch.Tensor,
                trg_mask: torch.Tensor = None, src_mask: torch.Tensor = None):
        """Decode compound features with protein context.

        Args:
            trg: [batch, compound_len, hid_dim] compound features.
            src: [protein_len, batch, hid_dim] protein encoder output.
            trg_mask: compound padding mask.
            src_mask: protein padding mask.

        Returns:
            Tuple of (ki, kd, ic50, ec50) predictions, each [batch, 1].
        """
        trg = trg.permute(1, 0, 2).contiguous()
        trg_mask = (trg_mask == 1)
        trg = self.decoder(trg, src, tgt_key_padding_mask=trg_mask,
                           memory_key_padding_mask=src_mask)
        trg = trg.permute(1, 0, 2).contiguous()
        x = trg[:, 0, :]
        interaction = F.relu(self.fc_1(x))
        ki = self.fc_ki(interaction)
        kd = self.fc_kd(interaction)
        ic50 = self.fc_ic50(interaction)
        ec50 = self.fc_ec50(interaction)
        return ki, kd, ic50, ec50

    def get_attn_maps(self, trg: torch.Tensor, src: torch.Tensor,
                      trg_mask: torch.Tensor = None, src_mask: torch.Tensor = None):
        """Extract cross-attention maps from all decoder layers.

        Returns:
            List of attention weight tensors, one per layer.
        """
        trg = trg.permute(1, 0, 2).contiguous()
        trg_mask = (trg_mask == 1)
        attn_maps = []
        for i in range(self.n_layers):
            out = self.decoder.layers[i].norm1(
                trg + self.decoder.layers[i]._sa_block(trg, attn_mask=None, key_padding_mask=trg_mask)
            )
            attn = self.decoder.layers[i].multihead_attn(
                out, src, src,
                attn_mask=None, key_padding_mask=src_mask, need_weights=True,
            )[1]
            attn_maps.append(attn)
            trg = self.decoder.layers[i](trg, src, tgt_key_padding_mask=trg_mask,
                                         memory_key_padding_mask=src_mask)
        return attn_maps


class _CPIBase(nn.Module):
    """Base class with shared GCN and mask-creation logic."""

    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.training.device
        self.atom_dim = cfg.model.atom_dim
        self.hid_dim = cfg.model.hid_dim
        self.fc_1 = nn.Linear(self.atom_dim, self.atom_dim)
        self.fc_2 = nn.Linear(self.atom_dim, self.hid_dim)

    def gcn(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Single-layer graph convolution.

        Args:
            input: [batch, num_node, atom_dim]
            adj: [batch, num_node, num_node]

        Returns:
            [batch, num_node, atom_dim]
        """
        support = self.fc_1(input)
        return torch.bmm(adj, support)

    def _prepare(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        """Apply GCN and create masks."""
        compound_max_len = compound.shape[1]
        aa_max_len = aa.shape[1]
        sa_max_len = sa.shape[1]
        compound_mask, aa_mask, sa_mask = make_masks(
            atom_num, aa_num, sa_num,
            compound_max_len, aa_max_len, sa_max_len, self.device,
        )
        compound = self.gcn(compound, adj)
        compound = F.relu(self.fc_2(compound))
        return compound, compound_mask, aa_mask, sa_mask


class CPIModelAA(_CPIBase):
    """CPI model using amino acid sequence only."""

    def __init__(self, aa_encoder: EncoderofAA, decoder: Decoder, cfg):
        super().__init__(cfg)
        self.aa_encoder = aa_encoder
        self.decoder = decoder

    def forward(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, _ = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        enc_src, src_mask = self.aa_encoder(aa, aa_mask)
        return self.decoder(compound, enc_src, compound_mask, src_mask)

    def get_attn_maps(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, _ = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        enc_src, src_mask = self.aa_encoder(aa, aa_mask)
        maps = self.decoder.get_attn_maps(compound, enc_src, compound_mask, src_mask)
        return enc_src, maps


class CPIModel3Di(_CPIBase):
    """CPI model using 3Di structural alphabet only."""

    def __init__(self, sa_encoder: Encoderof3Di, decoder: Decoder, cfg):
        super().__init__(cfg)
        self.sa_encoder = sa_encoder
        self.decoder = decoder

    def forward(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, _, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        enc_src, src_mask = self.sa_encoder(sa, sa_mask)
        return self.decoder(compound, enc_src, compound_mask, src_mask)

    def get_attn_maps(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, _, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        enc_src, src_mask = self.sa_encoder(sa, sa_mask)
        maps = self.decoder.get_attn_maps(compound, enc_src, compound_mask, src_mask)
        return enc_src, maps


class CPIModelAA3Di(_CPIBase):
    """CPI model with simple addition fusion of AA and 3Di representations."""

    def __init__(self, aa_encoder: EncoderofAA, sa_encoder: Encoderof3Di,
                 decoder: Decoder, cfg):
        super().__init__(cfg)
        self.aa_encoder = aa_encoder
        self.sa_encoder = sa_encoder
        self.decoder = decoder

    def forward(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        aa_src, aa_mask = self.aa_encoder(aa, aa_mask)
        sa_src, sa_mask = self.sa_encoder(sa, sa_mask)
        # Remove CLS/SEP tokens from AA, then add
        aa_src = aa_src[1:-1, :, :]
        pro_src = aa_src + sa_src
        return self.decoder(compound, pro_src, compound_mask, sa_mask)

    def get_attn_maps(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        aa_src, aa_mask = self.aa_encoder(aa, aa_mask)
        sa_src, sa_mask = self.sa_encoder(sa, sa_mask)
        aa_src = aa_src[1:-1, :, :]
        pro_src = aa_src + sa_src
        maps = self.decoder.get_attn_maps(compound, pro_src, compound_mask, sa_mask)
        return aa_src, sa_src, pro_src, maps


class CrossAttentionFusionBlock(nn.Module):
    """Bidirectional cross-attention fusion between AA and 3Di representations."""

    def __init__(self, cfg):
        super().__init__()
        self.cross_attn_AtoS = nn.MultiheadAttention(
            cfg.model.encoder_aa.hid_dim, cfg.model.cafb.n_head)
        self.cross_attn_StoA = nn.MultiheadAttention(
            cfg.model.encoder_sa.hid_dim, cfg.model.cafb.n_head)

    def forward(self, encoder_aa_output, encoder_sa_output, sa_mask):
        # AA as query, 3Di as key/value
        attn_output_sa, _ = self.cross_attn_AtoS(
            query=encoder_aa_output, key=encoder_sa_output,
            value=encoder_sa_output, key_padding_mask=sa_mask)
        # 3Di as query, AA as key/value
        attn_output_aa, _ = self.cross_attn_StoA(
            query=encoder_sa_output, key=encoder_aa_output,
            value=encoder_aa_output, key_padding_mask=sa_mask)
        return attn_output_sa + attn_output_aa


class CPIModelAA3DiwithCAF(_CPIBase):
    """CPI model with cross-attention fusion of AA and 3Di."""

    def __init__(self, aa_encoder: EncoderofAA, sa_encoder: Encoderof3Di,
                 decoder: Decoder, cafb: CrossAttentionFusionBlock, cfg):
        super().__init__(cfg)
        self.aa_encoder = aa_encoder
        self.sa_encoder = sa_encoder
        self.decoder = decoder
        self.cafb = cafb

    def forward(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        aa_src, aa_mask = self.aa_encoder(aa, aa_mask)
        sa_src, sa_mask = self.sa_encoder(sa, sa_mask)
        aa_src = aa_src[1:-1, :, :]
        pro_src = self.cafb(aa_src, sa_src, sa_mask)
        return self.decoder(compound, pro_src, compound_mask, sa_mask)

    def get_attn_maps(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        aa_src, aa_mask = self.aa_encoder(aa, aa_mask)
        sa_src, sa_mask = self.sa_encoder(sa, sa_mask)
        aa_src = aa_src[1:-1, :, :]
        pro_src = self.cafb(aa_src, sa_src, sa_mask)
        maps = self.decoder.get_attn_maps(compound, pro_src, compound_mask, sa_mask)
        return aa_src, sa_src, pro_src, maps


class GateMechanismFusionBlock(nn.Module):
    """Adaptive gated fusion of AA and 3Di representations.

    Learns per-position gate weights to combine the two modalities.
    """

    def __init__(self, cfg):
        super().__init__()
        self.gate_aa = nn.Linear(cfg.model.encoder_aa.hid_dim, 1)
        self.gate_sa = nn.Linear(cfg.model.encoder_sa.hid_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_aa_output: torch.Tensor,
                encoder_sa_output: torch.Tensor) -> torch.Tensor:
        gate_aa_weight = self.sigmoid(self.gate_aa(encoder_aa_output)).mean(dim=0)
        gate_sa_weight = self.sigmoid(self.gate_sa(encoder_sa_output)).mean(dim=0)
        return gate_aa_weight * encoder_aa_output + gate_sa_weight * encoder_sa_output


class CPIModelAA3DiwithGMF(_CPIBase):
    """CPI model with gated mechanism fusion of AA and 3Di (default model)."""

    def __init__(self, aa_encoder: EncoderofAA, sa_encoder: Encoderof3Di,
                 decoder: Decoder, cfg):
        super().__init__(cfg)
        self.aa_encoder = aa_encoder
        self.sa_encoder = sa_encoder
        self.decoder = decoder
        self.gmfb = GateMechanismFusionBlock(cfg)

    def forward(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        aa_src, aa_mask = self.aa_encoder(aa, aa_mask)
        sa_src, sa_mask = self.sa_encoder(sa, sa_mask)
        aa_src = aa_src[1:-1, :, :]
        pro_src = self.gmfb(aa_src, sa_src)
        return self.decoder(compound, pro_src, compound_mask, sa_mask)

    def get_attn_maps(self, compound, adj, aa, sa, atom_num, aa_num, sa_num):
        compound, compound_mask, aa_mask, sa_mask = self._prepare(
            compound, adj, aa, sa, atom_num, aa_num, sa_num)
        aa_src, aa_mask = self.aa_encoder(aa, aa_mask)
        sa_src, sa_mask = self.sa_encoder(sa, sa_mask)
        aa_src = aa_src[1:-1, :, :]
        pro_src = self.gmfb(aa_src, sa_src)
        maps = self.decoder.get_attn_maps(compound, pro_src, compound_mask, sa_mask)
        return aa_src, sa_src, pro_src, maps


def build_model(cfg):
    """Construct model from config.

    Args:
        cfg: Hydra/OmegaConf config with model.type field.

    Returns:
        Constructed model instance.
    """
    model_type = cfg.model.type if hasattr(cfg.model, 'type') else 'aa3di_gmf'

    aa_encoder = EncoderofAA(cfg)
    sa_encoder = Encoderof3Di(cfg)
    decoder = Decoder(cfg)

    if model_type == 'aa':
        return CPIModelAA(aa_encoder, decoder, cfg)
    elif model_type == '3di':
        return CPIModel3Di(sa_encoder, decoder, cfg)
    elif model_type == 'aa3di':
        return CPIModelAA3Di(aa_encoder, sa_encoder, decoder, cfg)
    elif model_type == 'aa3di_caf':
        cafb = CrossAttentionFusionBlock(cfg)
        return CPIModelAA3DiwithCAF(aa_encoder, sa_encoder, decoder, cafb, cfg)
    elif model_type == 'aa3di_gmf':
        return CPIModelAA3DiwithGMF(aa_encoder, sa_encoder, decoder, cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
