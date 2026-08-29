#!/usr/bin/env python3
"""Exact baseline-conditioned group-coboundary transport in 128-D latent space."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    output = value.to(dtype=reference.dtype, device=reference.device).reshape(-1, 1)
    if output.shape[0] == 1 and reference.shape[0] != 1:
        output = output.expand(reference.shape[0], 1)
    if output.shape[0] != reference.shape[0]:
        raise ValueError("Condition/time batch size does not match latent batch size")
    return output


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.fc2(F.silu(self.fc1(self.norm(value)))))


class CouplingConditioner(nn.Module):
    """CN base plus AD residual conditioner with an identity initialization."""

    def __init__(self, input_dim: int, output_dim: int, width: int, residual_blocks: int) -> None:
        super().__init__()
        self.input = nn.Linear(int(input_dim), int(width))
        self.blocks = nn.ModuleList(ResidualBlock(int(width)) for _ in range(int(residual_blocks)))
        self.cn_head = nn.Linear(int(width), int(output_dim))
        self.ad_residual_head = nn.Linear(int(width), int(output_dim))
        for head in (self.cn_head, self.ad_residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.cn_head(hidden) + label * self.ad_residual_head(hidden)


class TimeConditionedAffineCoupling(nn.Module):
    """One analytically invertible, time-anchored affine coupling layer."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        time_feature_dim: int,
        width: int,
        residual_blocks: int,
        transform_even: bool,
        max_log_scale: float,
        max_shift: float,
    ) -> None:
        super().__init__()
        if int(latent_dim) % 2:
            raise ValueError("Affine coupling requires an even latent dimension")
        indices = torch.arange(int(latent_dim), dtype=torch.long)
        active = indices[(indices % 2 == 0) if transform_even else (indices % 2 == 1)]
        passive = indices[(indices % 2 == 1) if transform_even else (indices % 2 == 0)]
        self.register_buffer("active", active)
        self.register_buffer("passive", passive)
        self.max_log_scale = float(max_log_scale)
        self.max_shift = float(max_shift)
        half = int(latent_dim) // 2
        self.conditioner = CouplingConditioner(
            half + int(context_dim) + int(time_feature_dim),
            2 * half,
            int(width),
            int(residual_blocks),
        )

    def forward(
        self,
        latent: torch.Tensor,
        context_embedding: torch.Tensor,
        time_features: torch.Tensor,
        gate: torch.Tensor,
        label: torch.Tensor,
        inverse: bool,
    ) -> torch.Tensor:
        passive = latent[:, self.passive]
        raw = self.conditioner(
            torch.cat((passive, context_embedding, time_features), dim=1),
            label,
        )
        raw_scale, raw_shift = raw.chunk(2, dim=1)
        log_scale = gate * self.max_log_scale * torch.tanh(raw_scale)
        shift = gate * self.max_shift * torch.tanh(raw_shift)
        active = latent[:, self.active]
        transformed = (active - shift) * torch.exp(-log_scale) if inverse else active * torch.exp(log_scale) + shift
        output = latent.clone()
        output[:, self.active] = transformed
        return output


class ExactCouplingCoboundaryFlow(nn.Module):
    r"""Exact non-abelian coboundary ``Phi(s,t)=F_t o F_s^{-1}``.

    ``F_t`` is a stack of time-conditioned affine coupling maps.  The subject's
    first observed latent and age are fixed conditioning variables throughout a
    composed trajectory.  Consequently identity, inverse, and the two-parameter
    composition law hold by construction (up to floating-point roundoff).
    """

    variant = "exact_coupling_group_coboundary"
    coboundary_used = True
    ode_used = False
    attention_used = False

    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 64,
        width: int = 192,
        residual_blocks: int = 2,
        coupling_layers: int = 4,
        time_frequencies: tuple[float, ...] = (0.5, 1.0, 2.0),
        max_log_scale: float = 0.15,
        max_shift: float = 0.5,
        gate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if int(coupling_layers) < 2:
            raise ValueError("Use at least two alternating coupling layers")
        if float(gate_scale) <= 0.0 or float(max_log_scale) <= 0.0 or float(max_shift) <= 0.0:
            raise ValueError("Coupling bounds and gate scale must be positive")
        self.latent_dim = int(latent_dim)
        self.gate_scale = float(gate_scale)
        self.time_frequencies = tuple(float(value) for value in time_frequencies)
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(context_dim)),
            nn.SiLU(),
        )
        time_feature_dim = 2 + 2 * len(self.time_frequencies)
        self.layers = nn.ModuleList(
            TimeConditionedAffineCoupling(
                self.latent_dim,
                int(context_dim),
                time_feature_dim,
                int(width),
                int(residual_blocks),
                transform_even=(index % 2 == 0),
                max_log_scale=float(max_log_scale),
                max_shift=float(max_shift),
            )
            for index in range(int(coupling_layers))
        )

    def _conditions(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        context: torch.Tensor,
        context_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B,{self.latent_dim}], got {tuple(latent.shape)}")
        if context is None or context_time is None:
            raise ValueError("Exact coboundary transport requires fixed subject context and context_time")
        if context.shape != latent.shape:
            raise ValueError("Context must have the same [B,D] shape as latent")
        label = _column(condition, latent)
        relative = _column(time, latent) - _column(context_time, latent)
        gate = torch.tanh(relative / self.gate_scale)
        features = [relative, torch.tanh(relative)]
        for frequency in self.time_frequencies:
            angle = math.pi * frequency * relative
            features.extend((torch.sin(angle), torch.cos(angle) - 1.0))
        return self.context_encoder(context), torch.cat(features, dim=1), gate, label

    def chart(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        context: torch.Tensor,
        context_time: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        context_embedding, time_features, gate, label = self._conditions(
            latent, time, condition, context, context_time
        )
        value = latent
        layers = reversed(self.layers) if inverse else self.layers
        for layer in layers:
            value = layer(value, context_embedding, time_features, gate, label, inverse)
        return value

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: torch.Tensor,
        context: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        canonical = self.chart(latent, source_time, condition, context, context_time, inverse=True)
        return self.chart(canonical, target_time, condition, context, context_time, inverse=False)


def build_flow(config: dict) -> ExactCouplingCoboundaryFlow:
    model = config["model"]
    return ExactCouplingCoboundaryFlow(
        latent_dim=int(model["latent_dim"]),
        context_dim=int(model["context_dim"]),
        width=int(model["width"]),
        residual_blocks=int(model["residual_blocks"]),
        coupling_layers=int(model["coupling_layers"]),
        time_frequencies=tuple(float(value) for value in model["time_frequencies"]),
        max_log_scale=float(model["max_log_scale"]),
        max_shift=float(model["max_shift"]),
        gate_scale=float(model["gate_scale"]),
    )
