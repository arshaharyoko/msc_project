"""
Relational graph convolution over the signalling pathway.

Per-node cross-attention binds the compound latents to the pathway nodes, then
the message-passing stack propagates along the relation adjacencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.layers.rgcn_message_passing import MessagePassingLayer

__all__ = ["RGCN"]


class RGCN(nn.Module):
    def __init__(self, d_in, d_h, num_relations, d_out=1, num_layers=5, readout="softmax"):
        super().__init__()
        if readout not in ("softmax", "additive"):
            raise ValueError(f"readout must be 'softmax' or 'additive', got {readout!r}")
        self.readout = readout
        self.node_query = nn.Linear(d_in, d_in, bias=False)

        if readout == "additive":
            # Additive binding: a closed gate removes its term outright instead of
            # leaving a bias vector in a convex average (ADR-020).
            self.bind_key = nn.Linear(d_in, d_in, bias=False)
            self.bind_value = nn.Linear(d_in, d_in, bias=False)
            self.bind_threshold = nn.Linear(d_in, 1)

        dims = [d_in] + [d_h] * (num_layers - 1) + [d_out]
        self.layers = nn.ModuleList(
            MessagePassingLayer(dims[i], dims[i + 1], num_relations) for i in range(num_layers)
        )

    def with_readout(self, readout):
        """A fresh stack of the same shape using another aggregation rule."""
        first, last = self.layers[0], self.layers[-1]
        d_in, d_h = first.relation_weights.shape[1], first.relation_weights.shape[2]
        return RGCN(d_in, d_h, first.num_relations, d_out=last.relation_weights.shape[2],
                    num_layers=len(self.layers), readout=readout)

    def _bind(self, z_in, node_features, gate):
        batch, n_comp, d_f = z_in.shape
        queries = self.node_query(node_features).unsqueeze(0).expand(batch, -1, -1)

        if self.readout == "additive":
            key = self.bind_key(z_in)
            bias = self.bind_threshold(node_features).view(1, -1, 1)
            attn = torch.sigmoid(
                torch.bmm(queries, key.transpose(1, 2)) / (d_f ** 0.5) + bias)
            if gate is not None:
                attn = attn * gate.transpose(1, 2)
            return torch.bmm(attn, self.bind_value(z_in))

        scores = torch.bmm(queries, z_in.transpose(1, 2)) / (d_f ** 0.5)
        if gate is not None:
            # Gate as a score offset, under the readout's single normalisation: a
            # component at gate 0 loses its share of the average (AC-M4).
            scores = scores + torch.log(gate.clamp_min(1e-12)).transpose(1, 2)
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, z_in)

    def forward(self, z_in, node_features, adjacency_mats, gate=None):
        context = self._bind(z_in, node_features, gate)
        x = node_features.unsqueeze(0) * context

        *hidden, readout = self.layers
        for messagepass in hidden:
            h = F.gelu(messagepass(x, adjacency_mats))
            x = x + h if h.shape == x.shape else h

        x = readout(x, adjacency_mats)
        # Logits, not probabilities: the readout saturates to exactly 1.0 in float32, and
        # a probability cannot be inverted back past that. Call sites squash (ADR-011).
        return x
