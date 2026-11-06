"""
Molecular encoder: bond graph -> one embedding vector per molecule.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.featurization import ATOM_TYPES, NUM_ATOM_FEATS, NUM_RELATIONS
from src.modules.layers.rgcn_message_passing import MessagePassingLayer

__all__ = ["MolecularRGCN"]


class MolecularRGCN(nn.Module):
    def __init__(
            self,
            d_h: int = 1024,
            d_out: int = 1024,
            num_layers: int = 5,
            num_relations: int = NUM_RELATIONS,
            dropout: float = 0.1,
            normalize_output: bool = True):
        super().__init__()
        self.normalize_output = normalize_output

        self.atom_embedding = nn.Embedding(len(ATOM_TYPES), d_h)
        self.atom_feat_proj = nn.Linear(NUM_ATOM_FEATS, d_h)
        self.input_norm = nn.LayerNorm(d_h)

        # Uniform width so every layer gets a residual. The pathway stack
        # changes dimension at the first layer, where `x + h if h.shape == x.shape`
        # silently falls through to no residual at all.
        self.layers = nn.ModuleList(
            MessagePassingLayer(d_h, d_h, num_relations) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_h) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

        # Mean and max capture average composition and the
        # presence of any one distinctive atom environment.
        self.readout = nn.Sequential(
            nn.Linear(2 * d_h, d_h), nn.GELU(), nn.Linear(d_h, d_out)
        )

    def forward(self, atom_types, atom_feats, adjacency_mats, node_mask, return_nodes=False):
        x = self.atom_embedding(atom_types) + self.atom_feat_proj(atom_feats)
        x = self.input_norm(x) * node_mask.unsqueeze(-1)

        for messagepass, norm in zip(self.layers, self.norms):
            h = messagepass(norm(x), adjacency_mats, node_mask)
            x = x + self.dropout(F.gelu(h))

        x = x * node_mask.unsqueeze(-1)

        denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = x.sum(dim=1) / denom
        max_pool = x.masked_fill(node_mask.unsqueeze(-1) == 0, float("-inf")).max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)

        z = self.readout(torch.cat([mean_pool, max_pool], dim=-1))
        if self.normalize_output:
            z = F.normalize(z, dim=-1) # cosine similarity becomes a dot product

        return (z, x) if return_nodes else z
