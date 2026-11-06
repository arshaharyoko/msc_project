"""Constant-logit stand-in for the learned component gate."""

import torch
import torch.nn as nn

__all__ = ["OPEN_GATE_LOGIT", "OpenGate"]

OPEN_GATE_LOGIT = float("inf")


class OpenGate(nn.Module):
    def __init__(self, logit=OPEN_GATE_LOGIT):
        super().__init__()
        self.register_buffer("logit", torch.tensor(float(logit)))

    def forward(self, y):
        return self.logit.expand(*y.shape[:-1], 1)
