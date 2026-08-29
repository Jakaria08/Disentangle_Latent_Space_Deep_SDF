#!/usr/bin/env python3
"""Transport models used by the matched 128-D comparison."""

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
        hidden = self.dropout(self.fc2(hidden))
        return F.silu(value + hidden)


class PlainODEFunc(nn.Module):
    """Plain diagnosis-conditioned MLP vector field dz/dt=f(z,t,d)."""

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
    """Released BrainODE Q/K/V form with enforced singleton-trajectory semantics.

    The released local implementation applies Q/K/V attention across the batch.
    That makes one subject's derivative depend on unrelated subjects placed in
    the same optimization batch.  This implementation follows the repository's
    later matched protocol: each case is passed through Q/K/V as a singleton,
    then the independent outputs are concatenated.  It is therefore invariant
    to batch composition and ordering.
    """

    attention_contract = "one_case_one_trajectory; singleton QKV; no cross-subject batch attention"

    def __init__(
        self,
        latent_dim: int,
        attention_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
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
        if features.shape[0] != 1:
            raise ValueError("BrainODE singleton attention received more than one case")
        query = self.query(features)
        key = self.key(features)
        value = self.value(features)
        attention = torch.softmax(query @ key.transpose(0, 1) * self.scale, dim=-1)
        hidden = attention @ value
        return self.fc2(self.dropout(F.gelu(self.fc1(hidden))))

    def forward(self, time: torch.Tensor, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B,{self.latent_dim}], got {tuple(latent.shape)}")
        times = _column(time, latent)
        conditions = _column(condition, latent)
        outputs = [
            self._singleton(torch.cat((latent[index : index + 1], times[index : index + 1], conditions[index : index + 1]), dim=1))
            for index in range(latent.shape[0])
        ]
        return torch.cat(outputs, dim=0)


def rk4_step(
    function: nn.Module,
    state: torch.Tensor,
    time_start: torch.Tensor,
    delta_time: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    time_start = time_start.reshape(-1).to(state)
    delta_time = delta_time.reshape(-1).to(state)
    dt = delta_time.reshape(-1, 1)
    half = 0.5 * delta_time
    k1 = function(time_start, state, condition)
    k2 = function(time_start + half, state + 0.5 * dt * k1, condition)
    k3 = function(time_start + half, state + 0.5 * dt * k2, condition)
    k4 = function(time_start + delta_time, state + dt * k3, condition)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_sequence_rk4(
    function: nn.Module,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    condition: torch.Tensor,
    substeps: int,
) -> torch.Tensor:
    if initial_state.ndim != 2 or times.ndim != 2 or times.shape[0] != initial_state.shape[0]:
        raise ValueError("Expected initial_state [B,D] and times [B,T]")
    if times.shape[1] < 1 or int(substeps) < 1:
        raise ValueError("Trajectory needs at least one time and one RK4 substep")
    current = initial_state
    states = [current]
    for step in range(times.shape[1] - 1):
        start = times[:, step]
        delta = (times[:, step + 1] - start) / float(substeps)
        current_time = start
        for _ in range(int(substeps)):
            current = rk4_step(function, current, current_time, delta, condition)
            current_time = current_time + delta
        states.append(current)
    return torch.stack(states, dim=1)


def transport_rk4(
    function: nn.Module,
    latent: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    condition: torch.Tensor,
    substeps: int,
) -> torch.Tensor:
    times = torch.stack((source_time.reshape(-1), target_time.reshape(-1)), dim=1).to(latent)
    return integrate_sequence_rk4(function, latent, times, condition, substeps)[:, -1, :]


class DirectC4Flow(nn.Module):
    """Direct non-ODE C4: z + (t-s)[v_CN + d v_AD]."""

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
        hidden = F.silu(self.input(torch.cat((latent, source, target, delta, torch.abs(delta), midpoint), dim=1)))
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


def build_ode(config: dict) -> nn.Module:
    model = config["model"]
    method = str(config["method"])
    common = {
        "latent_dim": int(model["latent_dim"]),
    }
    if method == "plain_ode":
        return PlainODEFunc(
            **common,
            width=int(model["width"]),
            residual_blocks=int(model["residual_blocks"]),
            dropout=float(model.get("dropout", 0.0)),
        )
    if method == "brainode":
        return BrainODEAttentionFunc(
            **common,
            attention_dim=int(model["attention_dim"]),
            hidden_dim=int(model["hidden_dim"]),
            dropout=float(model.get("dropout", 0.0)),
        )
    raise ValueError(f"Not an ODE method: {method}")
