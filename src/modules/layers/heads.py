"""
Output heads for the molecular encoder.
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ProjectionHead", "PropertyHead"]


class ProjectionHead(nn.Module):
    """Contrastive projection."""

    def __init__(self, d_in=256, d_proj=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_proj))

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class PropertyHead(nn.Module):
    """Regression head for QM9-style multi-target supervision."""

    def __init__(self, d_in=256, num_targets=12, d_h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Linear(d_h, num_targets))

    def forward(self, z):
        return self.net(z)
