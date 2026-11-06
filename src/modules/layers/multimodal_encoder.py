"""Two-branch encoder fusing the PubChem and UniProt modalities into one latent."""

import torch
import torch.nn as nn

__all__ = ["MultimodalEncoder"]


class MultimodalEncoder(nn.Module):
    def __init__(self, da_in, db_in, da_h, db_h, d_h):
        super().__init__()
        self.fc_a_1 = nn.Sequential(
            nn.Linear(da_in, da_h),
            nn.LayerNorm(da_h),
            nn.GELU()
        )
        self.fc_a_2 = nn.Sequential(
            nn.Linear(da_h, d_h*2),
            nn.LayerNorm(d_h*2),
            nn.GELU()
        )

        self.fc_b_1 = nn.Sequential(
            nn.Linear(db_in, db_h),
            nn.LayerNorm(db_h),
            nn.GELU()
        )
        self.fc_b_2 = nn.Sequential(
            nn.Linear(db_h, d_h*2),
            nn.LayerNorm(d_h*2),
            nn.GELU()
        )

        self.fc_ab_3 = nn.Sequential(
            nn.Linear(d_h*2, d_h),
            nn.LayerNorm(d_h),
            nn.GELU()
        )

    def forward(self, x1, x2):
        # -> (batch, n_comp_1, d_in1)
        # -> (batch, n_comp_2, d_in2)
        y1 = self.fc_a_1(x1)  # (batch, n_comp_1, da_h)
        y2 = self.fc_b_1(x2)  # (batch, n_comp_2, db_h)

        yh1 = self.fc_a_2(y1) # (batch, n_comp_1, d_h*2)
        yh2 = self.fc_b_2(y2) # (batch, n_comp_2, d_h*2)

        y12 = torch.cat((yh1, yh2), dim=1)  # (batch, n_comp_1+n_comp2, d_h*2)
        y12 = self.fc_ab_3(y12)             # (batch, n_comp_1+n_comp_2, d_h)
        return y12
