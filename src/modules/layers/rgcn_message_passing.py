"""Relational message passing over the signalling-pathway adjacency."""

import torch
import torch.nn as nn

__all__ = ["MessagePassingLayer"]


class MessagePassingLayer(nn.Module):
    # L stacked layers compute (A^T)^L X
    def __init__(self, d_in, d_out, num_relations):
        super().__init__()
        self.num_relations = num_relations

        relation_weights = torch.empty(num_relations, d_in, d_out)
        for w in relation_weights:
            nn.init.xavier_uniform_(w)
        self.relation_weights = nn.Parameter(relation_weights)

        self_weight = torch.empty(d_in, d_out)
        nn.init.xavier_uniform_(self_weight)
        self.self_weight = nn.Parameter(self_weight)

        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, node_features, adjacency_mats, node_mask=None):
        # node_features: (batch, n_nodes, d_in)
        # adjacency_mats: (num_relations, n_nodes, n_nodes)
        out = torch.matmul(node_features, self.self_weight)

        for i in range(self.num_relations):
            A = adjacency_mats[..., i, :, :]

            # A[j][i] = 1
            # A^T aggregates over incoming neighbours, column i holds node i's incoming edges
            messages = torch.matmul(A.transpose(-1, -2), torch.matmul(node_features, self.relation_weights[i]))
            in_degree = A.sum(dim=-2).clamp(min=1.0).unsqueeze(-1)  # (n_nodes, 1)

            out = out + messages / in_degree

        out = out + self.bias
        return out if node_mask is None else out * node_mask.unsqueeze(-1)
