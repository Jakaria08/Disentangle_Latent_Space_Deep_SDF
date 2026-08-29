#!/usr/bin/env python3
"""Matched 256-D direct-C4, plain-ODE, and BrainODE transport models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    output = value.to(dtype=reference.dtype, device=reference.device).reshape(-1, 1)
    if output.shape[0] == 1 and reference.shape[0] != 1:
        output = output.expand(reference.shape[0], 1)
    if output.shape[0] != reference.shape[0]:
        raise ValueError("Time/condition batch size does not match latent batch size")
    return output


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.fc1(self.norm(value)))
        return F.silu(value + self.dropout(self.fc2(hidden)))


class PlainODEFunc(nn.Module):
    attention_contract = "none"

    def __init__(self, latent_dim: int, width: int, residual_blocks: int, dropout: float) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input = nn.Linear(self.latent_dim + 2, int(width))
        self.blocks = nn.ModuleList(ResidualBlock(int(width), float(dropout)) for _ in range(int(residual_blocks)))
        self.output = nn.Linear(int(width), self.latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, time: torch.Tensor, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B,{self.latent_dim}], got {tuple(latent.shape)}")
        features = torch.cat((latent, _column(time, latent), _column(condition, latent)), dim=1)
        hidden = F.silu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class BrainODEAttentionFunc(nn.Module):
    attention_contract = "one_case_one_trajectory; singleton QKV; no cross-subject batch attention"

    def __init__(self, latent_dim: int, attention_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        input_dim = self.latent_dim + 2
        self.query = nn.Linear(input_dim, int(attention_dim))
        self.key = nn.Linear(input_dim, int(attention_dim))
        self.value = nn.Linear(input_dim, int(attention_dim))
        self.scale = int(attention_dim) ** -0.5
        self.fc1 = nn.Linear(int(attention_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(int(hidden_dim), self.latent_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _singleton(self, features: torch.Tensor) -> torch.Tensor:
        query, key, value = self.query(features), self.key(features), self.value(features)
        attention = torch.softmax(query @ key.transpose(0, 1) * self.scale, dim=-1)
        return self.fc2(self.dropout(F.gelu(self.fc1(attention @ value))))

    def forward(self, time: torch.Tensor, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B,{self.latent_dim}], got {tuple(latent.shape)}")
        times, labels = _column(time, latent), _column(condition, latent)
        return torch.cat([
            self._singleton(torch.cat((latent[i:i + 1], times[i:i + 1], labels[i:i + 1]), dim=1))
            for i in range(latent.shape[0])
        ], dim=0)


def rk4_step(function: nn.Module, state: torch.Tensor, time_start: torch.Tensor, delta_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    time_start, delta_time = time_start.reshape(-1).to(state), delta_time.reshape(-1).to(state)
    dt, half = delta_time.reshape(-1, 1), 0.5 * delta_time
    k1 = function(time_start, state, condition)
    k2 = function(time_start + half, state + 0.5 * dt * k1, condition)
    k3 = function(time_start + half, state + 0.5 * dt * k2, condition)
    k4 = function(time_start + delta_time, state + dt * k3, condition)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_sequence_rk4(function: nn.Module, initial_state: torch.Tensor, times: torch.Tensor, condition: torch.Tensor, substeps: int) -> torch.Tensor:
    if initial_state.ndim != 2 or times.ndim != 2 or times.shape[0] != initial_state.shape[0]:
        raise ValueError("Expected initial_state [B,D] and times [B,T]")
    current, states = initial_state, [initial_state]
    for step in range(times.shape[1] - 1):
        start = times[:, step]
        delta = (times[:, step + 1] - start) / float(substeps)
        current_time = start
        for _ in range(int(substeps)):
            current = rk4_step(function, current, current_time, delta, condition)
            current_time = current_time + delta
        states.append(current)
    return torch.stack(states, dim=1)


def transport_rk4(function: nn.Module, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, condition: torch.Tensor, substeps: int) -> torch.Tensor:
    times = torch.stack((source_time.reshape(-1), target_time.reshape(-1)), dim=1).to(latent)
    return integrate_sequence_rk4(function, latent, times, condition, substeps)[:, -1]


class DirectC4Flow(nn.Module):
    variant = "direct"
    coboundary_used = False

    def __init__(self, latent_dim: int, width: int, residual_blocks: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input = nn.Linear(self.latent_dim + 5, int(width))
        self.blocks = nn.ModuleList(ResidualBlock(int(width), float(dropout)) for _ in range(int(residual_blocks)))
        self.cn_head = nn.Linear(int(width), self.latent_dim)
        self.ad_residual_head = nn.Linear(int(width), self.latent_dim)
        for head in (self.cn_head, self.ad_residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def average_velocity(self, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        source, target, label = _column(source_time, latent), _column(target_time, latent), _column(condition, latent)
        delta, midpoint = target - source, 0.5 * (source + target)
        hidden = F.silu(self.input(torch.cat((latent, source, target, delta, torch.abs(delta), midpoint), dim=1)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.cn_head(hidden) + label * self.ad_residual_head(hidden)

    def transport(self, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, condition: torch.Tensor, context=None, context_time=None) -> torch.Tensor:
        del context, context_time
        source, target = _column(source_time, latent), _column(target_time, latent)
        return latent + (target - source) * self.average_velocity(latent, source, target, condition)


def build_ode(config: dict) -> nn.Module:
    model, method = config["model"], str(config["method"])
    if method == "plain_ode":
        return PlainODEFunc(int(model["latent_dim"]), int(model["width"]), int(model["residual_blocks"]), float(model.get("dropout", 0.0)))
    if method == "brainode":
        return BrainODEAttentionFunc(int(model["latent_dim"]), int(model["attention_dim"]), int(model["hidden_dim"]), float(model.get("dropout", 0.0)))
    raise ValueError(f"Not an ODE method: {method}")
