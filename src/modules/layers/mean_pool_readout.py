"""Graph-free readout which mean-pool over components, then an MLP."""

import torch
import torch.nn as nn

__all__ = ["MeanPoolReadout"]


class MeanPoolReadout(nn.Module):
    def __init__(self, d_in, n_nodes, d_h, d_out=1):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_out = d_out
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, n_nodes * d_out),
        )

    @classmethod
    def matched(cls, d_in, n_nodes, target_params, d_out=1):
        """Smallest instance whose parameter count clears ``target_params``.

        Closed form first, then a short walk on the real count, so the answer
        cannot drift if the layer definition above changes.
        """
        per_unit = d_in + 3 + n_nodes * d_out   # weights + bias + LayerNorm pair + head
        width = max(1, int((target_params - n_nodes * d_out) // per_unit))

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
        # z_in: (batch, n_comp, d_in)
        if gate is None:
            pooled = z_in.mean(dim=1)                   # (batch, d_in)
        else:
            # Renormalised so a gated-shut component is equivalent to deleting its
            # slot rather than shrinking the pooled mean (AC-M4).
            weight = gate.squeeze(-1)                   # (batch, n_comp)
            pooled = ((z_in * weight.unsqueeze(-1)).sum(dim=1)
                      / weight.sum(dim=1, keepdim=True).clamp_min(1e-12))
        out = self.mlp(pooled)                          # (batch, n_nodes*d_out)
        return out.view(-1, self.n_nodes, self.d_out)
