#!/usr/bin/env python3
"""The original task3 direct latent cocycle network, copied as the fixed control."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    result = value.to(dtype=reference.dtype, device=reference.device).reshape(-1, 1)
    if result.shape[0] == 1 and reference.shape[0] != 1:
        result = result.expand(reference.shape[0], 1)
    if result.shape[0] != reference.shape[0]:
        raise ValueError("Time/condition batch size does not match latent batch size")
    return result


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.fc1(self.norm(value)))
        hidden = self.dropout(self.fc2(hidden))
        return F.silu(value + hidden)


class DirectLatentCocycle(nn.Module):
    """Direct non-ODE flow: ``z + (t-s) [v_CN + d v_AD]``."""

    variant = "direct"
    ode_used = False
    attention_used = False
    coboundary_used = False

    def __init__(
        self, latent_dim: int, width: int, residual_blocks: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input = nn.Linear(self.latent_dim + 5, int(width))
        self.blocks = nn.ModuleList(
            ResidualBlock(int(width), float(dropout)) for _ in range(int(residual_blocks))
        )
        self.cn_head = nn.Linear(int(width), self.latent_dim)
        self.ad_residual_head = nn.Linear(int(width), self.latent_dim)
        for head in (self.cn_head, self.ad_residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def average_velocity(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        source = _column(source_time, latent)
        target = _column(target_time, latent)
        label = _column(condition, latent)
        delta = target - source
        midpoint = 0.5 * (source + target)
        features = torch.cat((latent, source, target, delta, delta.abs(), midpoint), dim=1)
        hidden = F.silu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.cn_head(hidden) + label * self.ad_residual_head(hidden)

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: torch.Tensor,
        context: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context, context_time
        source = _column(source_time, latent)
        target = _column(target_time, latent)
        return latent + (target - source) * self.average_velocity(
            latent, source, target, condition
        )

    def instantaneous_velocity(
        self, latent: torch.Tensor, age: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.average_velocity(latent, age, age, condition)
