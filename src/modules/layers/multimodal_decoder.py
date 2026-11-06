"""Two-branch decoder splitting one latent back into the two modalities."""

import torch
import torch.nn as nn

__all__ = ["MultimodalDecoder"]


class MultimodalDecoder(nn.Module):
    def __init__(self, da_out, db_out, da_h, db_h, d_h, d_z):
        super().__init__()
        self.fc_ab_1 = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.LayerNorm(d_h),
            nn.GELU()
        )
        self.fc_ab_2 = nn.Sequential(
            nn.Linear(d_h, d_h*2),
            nn.LayerNorm(d_h*2),
            nn.GELU()
        )

        self.fc_a_3 = nn.Sequential(
            nn.Linear(d_h*2, da_h),
            nn.LayerNorm(da_h),
            nn.GELU()
        )
        self.fc_a_4 = nn.Linear(da_h, da_out)

        self.fc_b_3 = nn.Sequential(
            nn.Linear(d_h*2, db_h),
            nn.LayerNorm(db_h),
            nn.GELU()
        )
        self.fc_b_4 = nn.Linear(db_h, db_out)

    def forward(self, y, n_in_1, n_in_2):
        x = self.fc_ab_1(y)
        x = self.fc_ab_2(x)

        x1_input_rc, x2_input_rc = torch.split(x, [n_in_1, n_in_2], dim=1)

        x1 = self.fc_a_3(x1_input_rc)
        x1 = self.fc_a_4(x1)

        x2 = self.fc_b_3(x2_input_rc)
        x2 = self.fc_b_4(x2)
        return x1, x2
