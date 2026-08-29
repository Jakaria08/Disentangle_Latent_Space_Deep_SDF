#!/usr/bin/env python3
"""Exact volume-axis/shape coboundary for frozen 128-D representations."""

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
        self.norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(width))
        self.fc2 = nn.Linear(int(width), int(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.fc2(F.silu(self.fc1(self.norm(value)))))


class DiagnosisResidualHead(nn.Module):
    """CN base plus AD residual, initialized to the no-change map."""

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


class ShapeAffineCoupling(nn.Module):
    """Invertible coupling on 127 volume-orthogonal shape coordinates."""

    def __init__(
        self,
        shape_dim: int,
        context_dim: int,
        time_feature_dim: int,
        width: int,
        residual_blocks: int,
        transform_even: bool,
        max_log_scale: float,
        max_shift: float,
    ) -> None:
        super().__init__()
        indices = torch.arange(int(shape_dim), dtype=torch.long)
        active_mask = indices % 2 == (0 if transform_even else 1)
        active, passive = indices[active_mask], indices[~active_mask]
        if active.numel() == 0 or passive.numel() == 0:
            raise ValueError("Shape coupling requires at least two coordinates")
        self.register_buffer("active", active)
        self.register_buffer("passive", passive)
        self.max_log_scale = float(max_log_scale)
        self.max_shift = float(max_shift)
        self.conditioner = DiagnosisResidualHead(
            int(passive.numel()) + 1 + int(context_dim) + int(time_feature_dim),
            2 * int(active.numel()),
            int(width),
            int(residual_blocks),
        )

    def forward(
        self,
        shape: torch.Tensor,
        volume_coordinate: torch.Tensor,
        context_embedding: torch.Tensor,
        time_features: torch.Tensor,
        gate: torch.Tensor,
        label: torch.Tensor,
        inverse: bool,
    ) -> torch.Tensor:
        passive = shape[:, self.passive]
        raw = self.conditioner(
            torch.cat((passive, volume_coordinate, context_embedding, time_features), dim=1), label
        )
        raw_scale, raw_shift = raw.chunk(2, dim=1)
        log_scale = gate * self.max_log_scale * torch.tanh(raw_scale)
        shift = gate * self.max_shift * torch.tanh(raw_shift)
        active = shape[:, self.active]
        transformed = (active - shift) * torch.exp(-log_scale) if inverse else active * torch.exp(log_scale) + shift
        output = shape.clone()
        output[:, self.active] = transformed
        return output


def householder_volume_rotation(coefficient: torch.Tensor) -> torch.Tensor:
    """Return orthogonal Q whose first rotated coordinate is w/||w|| dot z."""

    coefficient = coefficient.detach().to(dtype=torch.float64).reshape(-1)
    norm = torch.linalg.vector_norm(coefficient)
    if not torch.isfinite(norm) or float(norm) <= 1.0e-10:
        raise ValueError("Volume coefficient must be finite and non-zero")
    unit = coefficient / norm
    first = torch.zeros_like(unit)
    first[0] = 1.0
    difference = unit - first
    squared = torch.dot(difference, difference)
    if float(squared) <= 1.0e-20:
        rotation = torch.eye(len(unit), dtype=torch.float64)
    else:
        rotation = torch.eye(len(unit), dtype=torch.float64) - 2.0 * torch.outer(difference, difference) / squared
    if not torch.allclose(rotation @ rotation.T, torch.eye(len(unit), dtype=rotation.dtype), atol=1.0e-10, rtol=1.0e-10):
        raise RuntimeError("Failed to construct an orthogonal volume rotation")
    if not torch.allclose(rotation[:, 0], unit, atol=1.0e-10, rtol=1.0e-10):
        raise RuntimeError("Volume direction is not the first rotated coordinate")
    return rotation.to(dtype=torch.float32)


class ExactVolumeCoboundaryFlow(nn.Module):
    r"""Exact group coboundary with an explicit decoded-log-volume potential.

    A train-only ridge direction defines one volume coordinate.  ``F_t`` first
    translates that coordinate by a diagnosis-specific scalar potential, then
    applies invertible coupling maps only to the orthogonal shape coordinates.
    Therefore volume control cannot be erased by the linearized shape map, and
    ``Phi(s,t)=F_t o F_s^{-1}`` remains exact.
    """

    variant = "exact_volume_axis_group_coboundary_v2"
    coboundary_used = True
    structural_exactness = True
    ode_used = False
    attention_used = False

    def __init__(
        self,
        volume_coefficient: torch.Tensor | list[float],
        latent_dim: int = 128,
        context_dim: int = 64,
        width: int = 192,
        residual_blocks: int = 2,
        coupling_layers: int = 4,
        time_frequencies: tuple[float, ...] = (0.5, 1.0, 2.0),
        max_log_scale: float = 0.15,
        max_shape_shift: float = 0.5,
        max_log_volume_potential: float = 0.30,
        gate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        coefficient = torch.as_tensor(volume_coefficient, dtype=torch.float32).reshape(-1)
        if coefficient.numel() != int(latent_dim):
            raise ValueError(f"Expected {latent_dim} volume coefficients, got {coefficient.numel()}")
        if int(coupling_layers) < 2:
            raise ValueError("Use at least two alternating coupling layers")
        self.latent_dim = int(latent_dim)
        self.shape_dim = self.latent_dim - 1
        self.gate_scale = float(gate_scale)
        self.max_log_volume_potential = float(max_log_volume_potential)
        self.time_frequencies = tuple(float(value) for value in time_frequencies)
        coefficient_norm = torch.linalg.vector_norm(coefficient)
        self.register_buffer("volume_coefficient", coefficient)
        self.register_buffer("volume_coefficient_norm", coefficient_norm.reshape(1))
        self.register_buffer("rotation", householder_volume_rotation(coefficient))
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(context_dim)),
            nn.SiLU(),
        )
        time_feature_dim = 2 + 2 * len(self.time_frequencies)
        self.volume_head = DiagnosisResidualHead(
            int(context_dim) + time_feature_dim,
            1,
            int(width),
            int(residual_blocks),
        )
        self.shape_layers = nn.ModuleList(
            ShapeAffineCoupling(
                self.shape_dim,
                int(context_dim),
                time_feature_dim,
                int(width),
                int(residual_blocks),
                transform_even=(index % 2 == 0),
                max_log_scale=float(max_log_scale),
                max_shift=float(max_shape_shift),
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
        if context is None or context_time is None or context.shape != latent.shape:
            raise ValueError("Exact volume coboundary requires fixed [B,D] context and context_time")
        label = _column(condition, latent)
        relative = _column(time, latent) - _column(context_time, latent)
        gate = torch.tanh(relative / self.gate_scale)
        features = [relative, torch.tanh(relative)]
        for frequency in self.time_frequencies:
            angle = math.pi * frequency * relative
            features.extend((torch.sin(angle), torch.cos(angle) - 1.0))
        return self.context_encoder(context), torch.cat(features, dim=1), gate, label

    def volume_potential(
        self,
        context: torch.Tensor,
        context_time: torch.Tensor,
        query_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        context_embedding, time_features, gate, label = self._conditions(
            context, query_time, condition, context, context_time
        )
        raw = self.volume_head(torch.cat((context_embedding, time_features), dim=1), label)
        return gate * self.max_log_volume_potential * torch.tanh(raw)

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
        potential = gate * self.max_log_volume_potential * torch.tanh(
            self.volume_head(torch.cat((context_embedding, time_features), dim=1), label)
        )
        rotated = latent @ self.rotation
        volume_coordinate, shape = rotated[:, :1], rotated[:, 1:]
        if inverse:
            for layer in reversed(self.shape_layers):
                shape = layer(shape, volume_coordinate, context_embedding, time_features, gate, label, True)
            volume_coordinate = volume_coordinate - potential / self.volume_coefficient_norm
        else:
            volume_coordinate = volume_coordinate + potential / self.volume_coefficient_norm
            for layer in self.shape_layers:
                shape = layer(shape, volume_coordinate, context_embedding, time_features, gate, label, False)
        return torch.cat((volume_coordinate, shape), dim=1) @ self.rotation.T

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


def build_flow(config: dict, volume_coefficient: list[float] | torch.Tensor) -> ExactVolumeCoboundaryFlow:
    model = config["model"]
    return ExactVolumeCoboundaryFlow(
        volume_coefficient=volume_coefficient,
        latent_dim=int(model["latent_dim"]),
        context_dim=int(model["context_dim"]),
        width=int(model["width"]),
        residual_blocks=int(model["residual_blocks"]),
        coupling_layers=int(model["coupling_layers"]),
        time_frequencies=tuple(float(value) for value in model["time_frequencies"]),
        max_log_scale=float(model["max_log_scale"]),
        max_shape_shift=float(model["max_shape_shift"]),
        max_log_volume_potential=float(model["max_log_volume_potential"]),
        gate_scale=float(model["gate_scale"]),
    )

