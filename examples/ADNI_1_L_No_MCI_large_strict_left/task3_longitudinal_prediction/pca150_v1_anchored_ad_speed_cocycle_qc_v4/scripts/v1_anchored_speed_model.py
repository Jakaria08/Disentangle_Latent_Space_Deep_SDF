#!/usr/bin/env python3
"""Frozen v1 direct-flow plus an AD-only learned displacement-speed calibrator."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from v1_speed_utils import decode_pca_torch, mesh_volume_torch


class AdSpeedHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        dimensions = [int(input_dim), *[int(value) for value in hidden_dims], 1]
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(nn.SiLU())
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)
        final = next(layer for layer in reversed(self.net) if isinstance(layer, nn.Linear))
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class V1AnchoredAdSpeedCalibrator(nn.Module):
    """Preserve v1 spatial deformation and learn AD displacement magnitude.

    The frozen direct flow supplies ``d0 = Phi_v1(z,s,t,c)-z``.  CN receives
    exactly ``d0``; AD receives a positive learned scalar times ``d0``.  This
    prevents the CN endpoint regression observed in the v3 full-loss model.
    """

    def __init__(
        self,
        *,
        base_flow: nn.Module,
        feature_pcs: int,
        hidden_dims: Sequence[int],
        dropout: float,
        minimum_speed: float,
        maximum_speed: float,
        individual_log_span: float,
        initial_ad_speed: float = 1.0,
        coefficient_mean: torch.Tensor,
        coefficient_std: torch.Tensor,
        feature_scalar_mean: torch.Tensor,
        feature_scalar_std: torch.Tensor,
        mean_flat: torch.Tensor,
        pca_components: torch.Tensor,
        faces: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base_flow = base_flow
        self.feature_pcs = int(feature_pcs)
        self.minimum_speed = float(minimum_speed)
        self.maximum_speed = float(maximum_speed)
        self.individual_log_span = float(individual_log_span)
        if not 0.0 < self.minimum_speed < 1.0 < self.maximum_speed:
            raise ValueError("Speed bounds must contain one")
        for parameter in self.base_flow.parameters():
            parameter.requires_grad_(False)
        self.base_flow.eval()

        self.register_buffer("coefficient_mean", coefficient_mean.float().view(1, -1))
        self.register_buffer("coefficient_std", torch.clamp(coefficient_std.float().view(1, -1), min=1.0e-6))
        self.register_buffer("feature_scalar_mean", feature_scalar_mean.float().view(1, -1))
        self.register_buffer("feature_scalar_std", torch.clamp(feature_scalar_std.float().view(1, -1), min=1.0e-6))
        self.register_buffer("mean_flat", mean_flat.float().view(1, -1))
        self.register_buffer("pca_components", pca_components.float())
        self.register_buffer("faces", faces.long())

        # Feature scalars: source age, target age, log gap, source log-volume,
        # v1 log-volume rate, and v1 displacement norm per year.
        self.head = AdSpeedHead(self.feature_pcs + 6, hidden_dims, dropout)
        if not self.minimum_speed < float(initial_ad_speed) < self.maximum_speed:
            raise ValueError("Initial AD speed must lie strictly within the speed bounds")
        initial_fraction = (float(initial_ad_speed) - self.minimum_speed) / (self.maximum_speed - self.minimum_speed)
        self.global_speed_logit = nn.Parameter(torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction))))

    @property
    def global_speed(self) -> torch.Tensor:
        return self.minimum_speed + (self.maximum_speed - self.minimum_speed) * torch.sigmoid(self.global_speed_logit)

    def vertices_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return decode_pca_torch(latent, self.mean_flat, self.pca_components)

    def volume_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return mesh_volume_torch(self.vertices_from_latent(latent), self.faces)

    def _base_prediction(
        self,
        source: torch.Tensor,
        source_age_norm: torch.Tensor,
        target_age_norm: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        self.base_flow.eval()
        with torch.no_grad():
            return self.base_flow.transport(source, source_age_norm, target_age_norm, condition)

    def feature_tensor(
        self,
        *,
        source: torch.Tensor,
        base_prediction: torch.Tensor,
        source_age_norm: torch.Tensor,
        target_age_norm: torch.Tensor,
        source_age_years: torch.Tensor,
        target_age_years: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            latent = (source[:, : self.feature_pcs] - self.coefficient_mean[:, : self.feature_pcs]) / self.coefficient_std[:, : self.feature_pcs]
            source_volume = self.volume_from_latent(source)
            base_volume = self.volume_from_latent(base_prediction)
            delta_years = target_age_years.view(-1) - source_age_years.view(-1)
            safe_delta = torch.where(delta_years.abs() < 1.0e-6, torch.full_like(delta_years, 1.0e-6), delta_years)
            base_log_rate = (torch.log(base_volume) - torch.log(source_volume)) / safe_delta
            displacement_rate = torch.linalg.norm(base_prediction - source, dim=1) / safe_delta.abs().clamp_min(1.0e-6)
            scalars = torch.stack(
                [
                    source_age_norm.view(-1),
                    target_age_norm.view(-1),
                    torch.log1p(safe_delta.abs()),
                    torch.log(source_volume),
                    base_log_rate,
                    displacement_rate,
                ],
                dim=1,
            )
            scalars = (scalars - self.feature_scalar_mean) / self.feature_scalar_std
        return torch.cat([latent, scalars], dim=1)

    def transport(
        self,
        source: torch.Tensor,
        source_age_norm: torch.Tensor,
        target_age_norm: torch.Tensor,
        source_age_years: torch.Tensor,
        target_age_years: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_prediction = self._base_prediction(source, source_age_norm, target_age_norm, condition)
        features = self.feature_tensor(
            source=source,
            base_prediction=base_prediction,
            source_age_norm=source_age_norm,
            target_age_norm=target_age_norm,
            source_age_years=source_age_years,
            target_age_years=target_age_years,
        )
        individual_log = self.individual_log_span * torch.tanh(self.head(features))
        ad_speed = torch.clamp(self.global_speed * torch.exp(individual_log), self.minimum_speed, self.maximum_speed)
        condition = condition.view(-1).to(dtype=source.dtype)
        speed = torch.where(condition >= 0.5, ad_speed, torch.ones_like(ad_speed))
        prediction = source + speed.unsqueeze(1) * (base_prediction - source)
        return prediction, {
            "base_prediction": base_prediction,
            "speed": speed,
            "ad_speed": ad_speed,
            "individual_log_speed": individual_log,
            "global_speed": self.global_speed.expand_as(speed),
        }

    def train(self, mode: bool = True) -> "V1AnchoredAdSpeedCalibrator":
        super().train(mode)
        # Dropout is only in the calibrator head. The teacher remains deterministic.
        self.base_flow.eval()
        return self


def model_config_from_instance(model: V1AnchoredAdSpeedCalibrator) -> dict[str, Any]:
    hidden_dims = [layer.out_features for layer in model.head.net if isinstance(layer, nn.Linear)][:-1]
    dropout = next((float(layer.p) for layer in model.head.net if isinstance(layer, nn.Dropout)), 0.0)
    return {
        "feature_pcs": int(model.feature_pcs),
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "minimum_speed": float(model.minimum_speed),
        "maximum_speed": float(model.maximum_speed),
        "initial_ad_speed": float(model.global_speed.detach().cpu().item()),
        "individual_log_span": float(model.individual_log_span),
    }
