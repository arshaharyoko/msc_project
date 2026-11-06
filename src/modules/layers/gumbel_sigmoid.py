"""Relaxed Bernoulli (Gumbel-sigmoid) sampling for the component gate."""

import torch
from torch import Tensor
from torch.distributions import RelaxedBernoulli

__all__ = ["gumbel_sigmoid"]


def gumbel_sigmoid(logits, tau = 1.0, sample = True, hard = False, threshold = 0.5):
    eps = torch.finfo(logits.dtype).eps
    if sample:
        u = torch.rand_like(logits).clamp_(eps, 1 - eps)
        noise = u.log() - (-u).log1p()
        gate = torch.sigmoid((logits + noise) / tau)
    else:
        gate = torch.sigmoid(logits / tau)

    if hard:
        # Straight-through: the forward pass is a 0/1 gate, the gradient is the
        # relaxed one, which stays finite where the hard gate's would be zero.
        hard_gate = (gate > threshold).to(gate.dtype)
        return gate + (hard_gate - gate).detach()
    return gate