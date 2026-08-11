#!/usr/bin/env python3
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def activation_layer(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU(inplace=False)
    if normalized in ("silu", "swish"):
        return nn.SiLU(inplace=False)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name!r}")


class RegisteredPCAFlowMap(nn.Module):
    """Two-time flow map in PCA coefficient space.

    The map predicts an annualized PCA velocity and applies it in one step:

        Phi(z, s, t, c) = z + (t_years - s_years) * v_theta(z, s, t, c)

    ``v_theta`` is initialized around empirical CN/AD training velocities, then
    learns residual coefficients in a dynamic PCA velocity basis. This keeps the
    model solver-free while making disease-specific volume speed explicit.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        dynamic_basis: torch.Tensor,
        coefficient_mean: torch.Tensor,
        coefficient_std: torch.Tensor,
        cn_velocity_mean: torch.Tensor,
        ad_velocity_mean: torch.Tensor,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "silu",
        dropout: float = 0.05,
        latent_condition_dim: int = 32,
        residual_scale: float = 0.0,
        zero_initialize_output: bool = True,
    ) -> None:
        super().__init__()
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.dynamic_dim = int(dynamic_basis.shape[0])
        self.latent_condition_dim = int(latent_condition_dim)
        self.residual_scale = float(residual_scale)
        if not 0 < self.latent_condition_dim <= self.latent_dim:
            raise ValueError("latent_condition_dim must be in [1, latent_dim]")
        if dynamic_basis.shape != (self.dynamic_dim, self.latent_dim):
            raise ValueError("dynamic_basis must have shape [dynamic_dim, latent_dim]")

        self.register_buffer("dynamic_basis", dynamic_basis.float())
        self.register_buffer("coefficient_mean", coefficient_mean.float().view(1, -1))
        self.register_buffer("coefficient_std", coefficient_std.float().view(1, -1))
        self.register_buffer("cn_velocity_mean", cn_velocity_mean.float().view(1, -1))
        self.register_buffer("ad_velocity_mean", ad_velocity_mean.float().view(1, -1))

        output_dim = self.dynamic_dim
        if self.residual_scale > 0.0:
            output_dim += self.latent_dim
        input_dim = self.latent_condition_dim + 4
        dims = [input_dim, *[int(width) for width in hidden_dims], output_dim]
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(left, right)
            layers.append(linear)
            if index < len(dims) - 2:
                layers.append(activation_layer(activation))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
        self.net = nn.Sequential(*layers)

        if zero_initialize_output:
            final = next(layer for layer in reversed(self.net) if isinstance(layer, nn.Linear))
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def base_rate(self, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.to(dtype=self.cn_velocity_mean.dtype, device=self.cn_velocity_mean.device)
        if condition.ndim == 1:
            condition = condition.unsqueeze(1)
        return self.cn_velocity_mean + condition * (self.ad_velocity_mean - self.cn_velocity_mean)

    def average_rate(
        self,
        latent: torch.Tensor,
        source_age_norm: torch.Tensor,
        target_age_norm: torch.Tensor,
        delta_years: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B,{self.latent_dim}], got {tuple(latent.shape)}")
        source_age_norm = source_age_norm.to(device=latent.device, dtype=latent.dtype).view(-1, 1)
        target_age_norm = target_age_norm.to(device=latent.device, dtype=latent.dtype).view(-1, 1)
        delta_years = delta_years.to(device=latent.device, dtype=latent.dtype).view(-1, 1)
        condition = condition.to(device=latent.device, dtype=latent.dtype).view(-1, 1)
        z_norm = (latent - self.coefficient_mean.to(dtype=latent.dtype)) / self.coefficient_std.to(
            dtype=latent.dtype
        )
        features = torch.cat(
            [
                z_norm[:, : self.latent_condition_dim],
                source_age_norm,
                target_age_norm,
                torch.log1p(torch.clamp(delta_years, min=0.0)),
                condition,
            ],
            dim=1,
        )
        raw = self.net(features)
        dynamic_coeff = raw[:, : self.dynamic_dim]
        residual_rate = dynamic_coeff @ self.dynamic_basis.to(dtype=latent.dtype)
        if self.residual_scale > 0.0:
            residual = raw[:, self.dynamic_dim :]
            residual_rate = residual_rate + float(self.residual_scale) * residual
        return self.base_rate(condition).to(dtype=latent.dtype) + residual_rate

    def transport(
        self,
        latent: torch.Tensor,
        source_age_norm: torch.Tensor,
        target_age_norm: torch.Tensor,
        source_age_years: torch.Tensor,
        target_age_years: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        delta_years = target_age_years.to(device=latent.device, dtype=latent.dtype).view(-1, 1)
        delta_years = delta_years - source_age_years.to(device=latent.device, dtype=latent.dtype).view(-1, 1)
        rate = self.average_rate(latent, source_age_norm, target_age_norm, delta_years, condition)
        return latent + delta_years * rate

