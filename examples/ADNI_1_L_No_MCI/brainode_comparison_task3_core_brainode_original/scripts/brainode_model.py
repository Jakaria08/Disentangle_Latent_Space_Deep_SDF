#!/usr/bin/env python3
from __future__ import annotations

import math

import torch
from torch import nn


class ODEFuncWithAttention(nn.Module):
    """BrainODE-style attention ODE function adapted to batch-safe inputs."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int = 1,
        attention_dim: int = 256,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + 1 + condition_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = attention_dim ** -0.5
        self.fc1 = nn.Linear(attention_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.condition_dim = int(condition_dim)

    def forward(
        self,
        time_value: torch.Tensor,
        latent_state: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if latent_state.dim() != 2:
            raise ValueError(
                f"latent_state must have shape [batch, latent_dim], got {latent_state.shape}"
            )

        batch_size = latent_state.shape[0]
        if time_value.dim() == 0:
            time_feature = time_value.view(1, 1).expand(batch_size, 1)
        elif time_value.dim() == 1:
            if time_value.shape[0] != batch_size:
                raise ValueError(
                    "time_value batch dimension must match latent_state batch dimension."
                )
            time_feature = time_value.view(batch_size, 1)
        else:
            raise ValueError(
                f"time_value must be scalar or [batch], got shape {time_value.shape}"
            )

        if condition.dim() == 1:
            condition_feature = condition.view(-1, 1)
        elif condition.dim() == 2:
            condition_feature = condition
        else:
            raise ValueError(
                f"condition must have shape [batch] or [batch, condition_dim], got {condition.shape}"
            )

        if condition_feature.shape[0] == 1 and batch_size != 1:
            condition_feature = condition_feature.expand(batch_size, -1)
        if condition_feature.shape[0] != batch_size:
            raise ValueError(
                "condition batch dimension must match latent_state batch dimension."
            )
        if condition_feature.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition feature dimension must be {self.condition_dim}, got {condition_feature.shape[1]}"
            )

        model_input = torch.cat(
            [latent_state, time_feature.to(latent_state.dtype), condition_feature.to(latent_state.dtype)],
            dim=-1,
        )
        query = self.query(model_input)
        key = self.key(model_input)
        value = self.value(model_input)
        attention = torch.softmax(query @ key.transpose(0, 1) * self.scale, dim=-1)
        hidden = attention @ value
        hidden = self.gelu(self.fc1(hidden))
        return self.fc2(hidden)


def rk4_step(
    func: nn.Module,
    state: torch.Tensor,
    time_start: torch.Tensor,
    delta_time: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    if delta_time.dim() == 0:
        delta_time = delta_time.view(1).expand(state.shape[0])
    if time_start.dim() == 0:
        time_start = time_start.view(1).expand(state.shape[0])
    dt = delta_time.view(-1, 1).to(state.dtype)
    half_dt = 0.5 * dt
    half_t = time_start + 0.5 * delta_time

    k1 = func(time_start, state, condition)
    k2 = func(half_t, state + half_dt * k1, condition)
    k3 = func(half_t, state + half_dt * k2, condition)
    k4 = func(time_start + delta_time, state + dt * k3, condition)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_sequence_rk4(
    func: nn.Module,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    condition: torch.Tensor,
    substeps: int = 1,
) -> torch.Tensor:
    if initial_state.dim() != 2:
        raise ValueError(
            f"initial_state must have shape [batch, latent_dim], got {initial_state.shape}"
        )
    if times.dim() != 2:
        raise ValueError(f"times must have shape [batch, steps], got {times.shape}")
    if times.shape[0] != initial_state.shape[0]:
        raise ValueError("times batch dimension must match initial_state.")
    if times.shape[1] < 1:
        raise ValueError("times must contain at least one time point.")
    if substeps < 1:
        raise ValueError("substeps must be at least 1.")

    states = [initial_state]
    current_state = initial_state
    if times.shape[1] == 1:
        return torch.stack(states, dim=1)

    for step_index in range(times.shape[1] - 1):
        time_start = times[:, step_index]
        time_end = times[:, step_index + 1]
        total_delta = time_end - time_start
        delta = total_delta / float(substeps)
        current_time = time_start
        for _ in range(substeps):
            current_state = rk4_step(
                func=func,
                state=current_state,
                time_start=current_time,
                delta_time=delta,
                condition=condition,
            )
            current_time = current_time + delta
        states.append(current_state)
    return torch.stack(states, dim=1)
