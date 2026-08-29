#!/usr/bin/env python3
"""Spiral mesh layers, including a differentiable Adaptive-Spiral support predictor."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add


def sparse_pool(features: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    row, column = transform._indices()
    weights = transform._values().to(features).unsqueeze(-1)
    gathered = torch.index_select(features, 1, column) * weights
    return scatter_add(gathered, row, dim=1, dim_size=transform.size(0))


class SpiralConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, indices: torch.Tensor) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.register_buffer("indices", indices.long())
        self.sequence_length = int(indices.shape[1])
        self.linear = nn.Linear(self.in_channels * self.sequence_length, self.out_channels)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,N,C], received {tuple(features.shape)}")
        batch = features.shape[0]
        gathered = torch.index_select(features, 1, self.indices.reshape(-1))
        return self.linear(gathered.reshape(batch, self.indices.shape[0], -1))


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if int(channels) % value == 0:
            return value
    return 1


class GroupLinear(nn.Module):
    def __init__(self, channels: int, groups: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.groups = _groups(channels) if groups is None else int(groups)
        if self.channels % self.groups:
            raise ValueError("GroupLinear channels must be divisible by groups")
        width = self.channels // self.groups
        self.weight = nn.Parameter(torch.empty(width, width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, vertices, _ = features.shape
        grouped = features.reshape(batch, vertices, self.groups, -1)
        output = torch.einsum("bngc,cd->bngd", grouped, self.weight) + self.bias
        return output.reshape(batch, vertices, self.channels)


class GatedSpiralDepthwise(nn.Module):
    def __init__(self, channels: int, indices: torch.Tensor) -> None:
        super().__init__()
        self.channels = int(channels)
        self.register_buffer("indices", indices.long())
        self.gate = nn.Linear(self.channels, self.channels)
        self.weight = nn.Parameter(torch.empty(indices.shape[0], indices.shape[1]))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        neighbors = torch.index_select(features, 1, self.indices.reshape(-1))
        neighbors = neighbors.reshape(batch, self.indices.shape[0], self.indices.shape[1], self.channels)
        filtered = torch.einsum("bnkc,nk->bnc", neighbors, self.weight)
        return filtered * torch.sigmoid(self.gate(features))


class LearnablePrefixPool(nn.Module):
    """Differentiable bounded prefix length with a nonzero initialization gradient."""

    def __init__(self, channels: int, indices: torch.Tensor, initial_support: float = 8.5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.register_buffer("indices", indices.long())
        self.max_sequence = int(indices.shape[1])
        if self.max_sequence < 2:
            raise ValueError("Adaptive prefix pool requires at least two spiral positions")
        self.predictor = nn.Linear(self.channels, 1)
        initial = min(max(float(initial_support), 0.5), self.max_sequence - 1.5)
        probability = initial / float(self.max_sequence - 1)
        nn.init.zeros_(self.predictor.weight)
        nn.init.constant_(self.predictor.bias, math.log(probability / (1.0 - probability)))
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.last_support: torch.Tensor | None = None

    def support(self, sequence_features: torch.Tensor) -> torch.Tensor:
        # Fractional support gradients are small early in training. Keep this scalar path
        # in FP32 even when the surrounding mesh network uses CUDA autocast; otherwise the
        # end-to-end predictor can remain numerically frozen although its analytic gradient
        # is nonzero.
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.predictor(sequence_features.float().mean(dim=2))
            return torch.sigmoid(logits) * float(self.max_sequence - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        sequence = torch.index_select(features, 1, self.indices.reshape(-1))
        sequence = sequence.reshape(batch, self.indices.shape[0], self.max_sequence, self.channels)
        support = self.support(sequence)
        prefix = sequence.float().cumsum(dim=2)
        lower = support.floor().long().detach()
        upper = support.ceil().long().detach().clamp_max(self.max_sequence - 1)
        fraction = support - lower.to(support.dtype)
        gather_shape = (batch, self.indices.shape[0], 1, self.channels)
        low_value = torch.gather(prefix, 2, lower.unsqueeze(-1).expand(gather_shape))
        high_value = torch.gather(prefix, 2, upper.unsqueeze(-1).expand(gather_shape))
        pooled = low_value + fraction.unsqueeze(-1) * (high_value - low_value)
        pooled = pooled.squeeze(2)
        self.last_support = support.detach()
        return self.norm(pooled.transpose(1, 2)).transpose(1, 2)


class AdaptiveSpiralConv(nn.Module):
    """Adaptive-Spiral operator with corrected learnable support parameterization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        indices: torch.Tensor,
        dynamic_indices: torch.Tensor,
        initial_support: float = 8.5,
    ) -> None:
        super().__init__()
        if int(in_channels) != int(out_channels):
            self.input_projection = nn.Linear(int(in_channels), int(out_channels))
        else:
            self.input_projection = nn.Identity()
        channels = int(out_channels)
        self.channels = channels
        self.local = SpiralConv(channels, channels, indices)
        self.prefix_pool = LearnablePrefixPool(channels, dynamic_indices, initial_support)
        self.channel_in = GroupLinear(channels)
        self.depthwise = GatedSpiralDepthwise(channels, indices)
        self.channel_out = GroupLinear(channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.full((channels,), 0.1))
        self.delta = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(features)
        value = value + self.gamma.view(1, 1, -1) * self.channel_in(value)
        value = value + self.beta * self.depthwise(value)
        value = value + self.alpha * self.prefix_pool(value)
        value = value + self.delta.view(1, 1, -1) * self.channel_out(value)
        return self.local(value)

    def support_statistics(self) -> dict[str, float] | None:
        support = self.prefix_pool.last_support
        if support is None:
            return None
        flat = support.float().reshape(-1)
        return {
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "p05": float(torch.quantile(flat, 0.05).cpu()),
            "p95": float(torch.quantile(flat, 0.95).cpu()),
        }


class ConditionalMeshBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        indices: torch.Tensor,
        operator: str,
        dynamic_indices: torch.Tensor | None = None,
        initial_support: float = 8.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        operator = str(operator)
        if operator == "spiral":
            self.conv: nn.Module = SpiralConv(in_channels, out_channels, indices)
        elif operator == "adaptive":
            if dynamic_indices is None:
                raise ValueError("Adaptive block requires dynamic spiral indices")
            self.conv = AdaptiveSpiralConv(
                in_channels, out_channels, indices, dynamic_indices, initial_support
            )
        else:
            raise ValueError(f"Unknown mesh operator: {operator}")
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.film = nn.Linear(int(condition_dim), 2 * int(out_channels))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(features)
        hidden = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        hidden = (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) * hidden + beta.unsqueeze(1)
        return F.silu(self.residual(features) + self.dropout(hidden))


def adaptive_modules(module: nn.Module) -> list[AdaptiveSpiralConv]:
    return [item for item in module.modules() if isinstance(item, AdaptiveSpiralConv)]


def adaptive_support_report(module: nn.Module) -> list[dict[str, Any]]:
    output = []
    for name, item in module.named_modules():
        if isinstance(item, AdaptiveSpiralConv):
            output.append({"module": name, "statistics": item.support_statistics()})
    return output
