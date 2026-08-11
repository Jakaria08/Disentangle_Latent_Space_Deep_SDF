"""Frozen-base, AD-only speed-plus-residual transport for full INR latents."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def _column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    return value.reshape(-1, 1)


class AnchoredAdResidualCocycle(nn.Module):
    """CN-anchored calibration of a frozen disease-conditioned direct flow.

    The base flow is deliberately not registered as a child module: checkpoints
    contain only the compact calibrator and train-only basis, while callers
    explicitly reload and freeze the documented base flow.
    """

    def __init__(
        self,
        *,
        base_flow: nn.Module,
        feature_mean: torch.Tensor,
        feature_components: torch.Tensor,
        residual_basis: torch.Tensor,
        hidden_dims: Sequence[int] = (96, 64),
        dropout: float = 0.05,
        speed_lower: float = 0.25,
        speed_upper: float = 4.0,
        speed_individual_log_span: float = 0.45,
        residual_coefficient_span: float = 0.08,
    ) -> None:
        super().__init__()
        if not (0.0 < speed_lower <= 1.0 <= speed_upper):
            raise ValueError("Speed bounds must contain the identity speed 1.")
        # Keep the frozen base out of this module's state_dict.  The checkpoint
        # records its source path separately and reloads it explicitly.
        object.__setattr__(self, "base_flow", base_flow)
        self.base_flow.eval()
        for parameter in self.base_flow.parameters():
            parameter.requires_grad_(False)
        feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(1, -1)
        feature_components = torch.as_tensor(feature_components, dtype=torch.float32)
        residual_basis = torch.as_tensor(residual_basis, dtype=torch.float32)
        if feature_components.ndim != 2 or feature_components.shape[1] != feature_mean.shape[1]:
            raise ValueError("Feature PCA shape mismatch.")
        if residual_basis.ndim != 2 or residual_basis.shape[1] != feature_mean.shape[1]:
            raise ValueError("Residual basis shape mismatch.")
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_components", feature_components)
        self.register_buffer("residual_basis", residual_basis)
        self.speed_lower = float(speed_lower)
        self.speed_upper = float(speed_upper)
        self.speed_individual_log_span = float(speed_individual_log_span)
        self.residual_coefficient_span = float(residual_coefficient_span)
        input_dim = int(feature_components.shape[0]) + 5
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.SiLU(inplace=False)])
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        layers.append(nn.Linear(previous, 1 + int(residual_basis.shape[0])))
        self.head = nn.Sequential(*layers)
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        # exp(0)=1 and residual=0 make this exactly the base transport initially.
        self.global_log_speed = nn.Parameter(torch.zeros(1))
        self.global_residual_coefficients = nn.Parameter(
            torch.zeros(int(residual_basis.shape[0]))
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.base_flow.eval()
        return self

    @property
    def latent_size(self) -> int:
        return int(self.feature_mean.shape[1])

    def _base_branches(
        self, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            cn = self.base_flow.transport(
                latent, source_time, target_time,
                torch.zeros(latent.shape[0], 1, device=latent.device, dtype=latent.dtype),
            )
            ad = self.base_flow.transport(
                latent, source_time, target_time,
                torch.ones(latent.shape[0], 1, device=latent.device, dtype=latent.dtype),
            )
        return cn, ad

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if latent.ndim != 2 or latent.shape[1] != self.latent_size:
            raise ValueError(f"Expected latent [B,{self.latent_size}], got {tuple(latent.shape)}")
        source_time = _column(source_time, latent)
        target_time = _column(target_time, latent)
        if condition is None:
            condition = torch.zeros_like(source_time)
        condition = _column(condition, latent).clamp(0.0, 1.0)
        cn, ad = self._base_branches(latent, source_time, target_time)
        delta_time = target_time - source_time
        pc = (latent - self.feature_mean.to(dtype=latent.dtype)) @ self.feature_components.to(dtype=latent.dtype).T
        disease_norm = (ad - cn).norm(dim=1, keepdim=True)
        inputs = torch.cat(
            [pc, source_time, target_time, delta_time, delta_time.abs(), disease_norm], dim=1
        )
        raw = self.head(inputs)
        log_speed = self.global_log_speed + self.speed_individual_log_span * torch.tanh(raw[:, :1])
        speed = torch.exp(log_speed).clamp(self.speed_lower, self.speed_upper)
        coefficients = self.global_residual_coefficients.unsqueeze(0) + (
            self.residual_coefficient_span * torch.tanh(raw[:, 1:])
        )
        residual_velocity = coefficients @ self.residual_basis.to(dtype=latent.dtype)
        ad_prediction = cn + speed * (ad - cn) + delta_time * residual_velocity
        prediction = cn + condition * (ad_prediction - cn)
        if not return_aux:
            return prediction
        return prediction, {
            "base_cn": cn,
            "base_ad": ad,
            "speed": speed,
            "residual_velocity": residual_velocity,
            "residual_coefficients": coefficients,
        }

    forward = transport

    def regularization(self) -> dict[str, torch.Tensor]:
        return {
            "speed_anchor": self.global_log_speed.square().mean(),
            "residual_l2": self.global_residual_coefficients.square().mean(),
        }
