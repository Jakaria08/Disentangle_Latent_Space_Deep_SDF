#!/usr/bin/env python3
"""Latent cocycle-flow networks.

Every variant keeps the direct, non-ODE transport and the two structural guarantees of
the published baseline:

    Phi(z,s,t,d) = z + (t-s) * [v_CN(z,s,t) + d * v_AD(z,s,t)]

  * exact identity at s=t, because the elapsed-time factor multiplies the output;
  * exact no-change at initialization, because both heads are zero-initialized.

Only the trunk that produces the velocity differs between variants.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

ARCHITECTURES = ("postact", "preact", "film")


def _column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    output = value.to(dtype=reference.dtype, device=reference.device).reshape(-1, 1)
    if output.shape[0] == 1 and reference.shape[0] != 1:
        output = output.expand(reference.shape[0], 1)
    if output.shape[0] != reference.shape[0]:
        raise ValueError("Time/condition batch size does not match latent batch size")
    return output


class PostActBlock(nn.Module):
    """Baseline block: SiLU(x + f(x)). Kept verbatim as the control condition."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, value: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        del condition
        hidden = F.silu(self.fc1(self.norm(value)))
        return F.silu(value + self.dropout(self.fc2(hidden)))


class PreActBlock(nn.Module):
    """Pre-activation residual: x + f(LN(x)); the identity path stays clean."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, value: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        del condition
        hidden = F.silu(self.fc1(self.norm(value)))
        return value + self.dropout(self.fc2(hidden))


class FiLMBlock(nn.Module):
    """Pre-activation residual with zero-initialized FiLM time/disease modulation.

    The baseline injects the interval only once, at the input layer; after that the trunk
    carries no time information. FiLM re-injects it at every block.
    """

    def __init__(self, width: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(condition_dim, 2 * width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        hidden = self.norm(value) * (1.0 + gamma) + beta
        hidden = F.silu(self.fc1(hidden))
        return value + self.dropout(self.fc2(hidden))


class IntervalEmbedding(nn.Module):
    """Interval encoding used by the FiLM variant.

    Ages arrive already standardized on the training split, so the Fourier features are
    taken directly on the normalized values.
    """

    def __init__(self, condition_dim: int, frequencies: int) -> None:
        super().__init__()
        self.frequencies = int(frequencies)
        self.disease = nn.Embedding(2, 8)
        raw_dim = 6 + 4 * self.frequencies + 8
        self.network = nn.Sequential(
            nn.Linear(raw_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        delta = target - source
        raw = [source, target, delta, delta.abs(), 0.5 * (source + target), torch.log1p(delta.abs())]
        for frequency in range(self.frequencies):
            scale = math.pi * float(2**frequency)
            raw.extend((torch.sin(scale * source), torch.cos(scale * source)))
            raw.extend((torch.sin(scale * target), torch.cos(scale * target)))
        raw.append(self.disease(label.reshape(-1).round().long().clamp(0, 1)))
        return self.network(torch.cat(raw, dim=1))


class LeanCocycleFlow(nn.Module):
    """Direct non-ODE latent cocycle flow with a selectable trunk."""

    variant = "direct"
    ode_used = False
    attention_used = False
    coboundary_used = False

    def __init__(
        self,
        latent_dim: int,
        width: int = 256,
        blocks: int = 2,
        dropout: float = 0.0,
        arch: str = "preact",
        condition_dim: int = 64,
        frequencies: int = 2,
    ) -> None:
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {ARCHITECTURES}, got {arch!r}")
        self.latent_dim = int(latent_dim)
        self.arch = str(arch)
        width = int(width)
        if self.arch == "film":
            self.condition = IntervalEmbedding(int(condition_dim), int(frequencies))
            self.input = nn.Linear(self.latent_dim, width)
            self.blocks = nn.ModuleList(
                FiLMBlock(width, int(condition_dim), float(dropout)) for _ in range(int(blocks))
            )
        else:
            block = PostActBlock if self.arch == "postact" else PreActBlock
            self.condition = None
            self.input = nn.Linear(self.latent_dim + 5, width)
            self.blocks = nn.ModuleList(block(width, float(dropout)) for _ in range(int(blocks)))
        self.cn_head = nn.Linear(width, self.latent_dim)
        self.ad_residual_head = nn.Linear(width, self.latent_dim)
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
        if self.arch == "film":
            code = self.condition(source, target, label)
            hidden = F.silu(self.input(latent))
            for block in self.blocks:
                hidden = block(hidden, code)
        else:
            delta = target - source
            features = torch.cat(
                (latent, source, target, delta, torch.abs(delta), 0.5 * (source + target)), dim=1
            )
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
        return latent + (target - source) * self.average_velocity(latent, source, target, condition)

    def instantaneous_velocity(
        self, latent: torch.Tensor, age: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.average_velocity(latent, age, age, condition)


def build_flow(latent_dim: int, model_config: dict) -> LeanCocycleFlow:
    return LeanCocycleFlow(
        latent_dim,
        width=int(model_config.get("width", 256)),
        blocks=int(model_config.get("blocks", 2)),
        dropout=float(model_config.get("dropout", 0.0)),
        arch=str(model_config.get("arch", "preact")),
        condition_dim=int(model_config.get("condition_dim", 64)),
        frequencies=int(model_config.get("frequencies", 2)),
    )
