"""Graph-free readout which gates and sums the components, then an MLP."""

import torch
import torch.nn as nn

__all__ = ["SumPoolReadout"]


class SumPoolReadout(nn.Module):
    def __init__(self, d_in, n_nodes, d_h, d_out=1):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_out = d_out
        # The same gated additive pooling the additive RGCN binds with, over one
        # pooled vector instead of per-node queries (ADR-020).
        self.bind_value = nn.Linear(d_in, d_in, bias=False)
        self.bind_score = nn.Linear(d_in, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, n_nodes * d_out),
        )

    @classmethod
    def matched(cls, d_in, n_nodes, target_params, d_out=1):
        """Smallest instance whose parameter count clears ``target_params``."""
        head = n_nodes * d_out
        fixed = d_in * d_in + d_in + 1 + head
        per_unit = d_in + 3 + head
        width = max(1, int((target_params - fixed) // per_unit))

        def count(w):
            return sum(p.numel() for p in cls(d_in, n_nodes, w, d_out).parameters())

        while count(width) < target_params:
            width += 1
        while width > 1 and count(width - 1) >= target_params:
            width -= 1
        return cls(d_in, n_nodes, width, d_out)

    @property
    def width(self):
        return self.mlp[0].out_features

    def forward(self, z_in, node_features, adjacency_mats, gate=None):
        weight = torch.sigmoid(self.bind_score(z_in))
        if gate is not None:
            weight = weight * gate
        pooled = (weight * self.bind_value(z_in)).sum(dim=1)
        out = self.mlp(pooled)
        return out.view(-1, self.n_nodes, self.d_out)
