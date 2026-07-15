"""Reusable direct continuous-age flow and modular longitudinal losses.

The module contains the initial objectives used by the ADNI no-MCI direct-flow
experiment:

1. real source-scan latent -> real target-scan SDF reconstruction;
2. direct-vs-composed latent consistency through an observed intermediate;
3. direct-vs-composed latent consistency through a random virtual age.

It also supports optional diagnostic extensions behind explicit config flags:

4. source-forward vs target-backward agreement at the same virtual age;
5. future extrapolation direct-vs-composed consistency beyond the target age.

The decoder and real scan latents are expected to be frozen.  Additional
losses can be added as separate methods without changing the transport API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


def _column(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.view(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"Expected scalar column tensor, got {tuple(value.shape)}")
    return value


def _activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU(inplace=False)
    if normalized in ("silu", "swish"):
        return nn.SiLU(inplace=False)
    if normalized == "softplus":
        return nn.Softplus()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown flow activation: {name!r}")


class DirectAgeFlow(nn.Module):
    """One-shot residual flow conditioned on source and target age.

    ``transport`` implements

        Phi(z, s, t, c) = z + (t - s) * G(z, s, t, c)

    where ``s`` and ``t`` use the fixed ADNI normalized chronological-age
    coordinate.  The diagonal output ``G(z, t, t, c)`` is velocity per one
    normalized-age unit; divide by the age range in years for velocity/year.
    """

    def __init__(
        self,
        latent_size: int,
        hidden_dims: Sequence[int],
        condition_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        zero_initialize_output: bool = True,
        latent_condition_mode: str = "full",
        latent_condition_dim: Optional[int] = None,
        include_delta_time_input: bool = False,
        pca_mean: Optional[torch.Tensor] = None,
        pca_components: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if int(latent_size) <= 0:
            raise ValueError("latent_size must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        self.latent_size = int(latent_size)
        self.condition_dim = max(0, int(condition_dim))
        self.include_delta_time_input = bool(include_delta_time_input)
        self.latent_condition_mode = str(latent_condition_mode).strip().lower()
        if self.latent_condition_mode == "population":
            self.latent_condition_mode = "none"
        if self.latent_condition_mode not in {"full", "none", "slice", "pca"}:
            raise ValueError(
                "latent_condition_mode must be one of full, none, population, "
                f"slice, pca; got {latent_condition_mode!r}"
            )
        if latent_condition_dim is None:
            if self.latent_condition_mode == "full":
                latent_condition_dim = self.latent_size
            elif self.latent_condition_mode == "none":
                latent_condition_dim = 0
            else:
                raise ValueError(
                    f"latent_condition_dim is required for {self.latent_condition_mode!r}"
                )
        self.latent_condition_dim = int(latent_condition_dim)
        if self.latent_condition_mode == "full" and self.latent_condition_dim != self.latent_size:
            raise ValueError(
                f"full latent conditioning requires dim={self.latent_size}, "
                f"got {self.latent_condition_dim}"
            )
        if self.latent_condition_mode == "none" and self.latent_condition_dim != 0:
            raise ValueError("none/population latent conditioning requires dim=0")
        if self.latent_condition_mode == "slice" and not (
            0 < self.latent_condition_dim <= self.latent_size
        ):
            raise ValueError(
                f"slice latent conditioning dim must be in [1,{self.latent_size}]"
            )
        if self.latent_condition_mode == "pca":
            if not (0 < self.latent_condition_dim <= self.latent_size):
                raise ValueError(
                    f"pca latent conditioning dim must be in [1,{self.latent_size}]"
                )
            if pca_mean is None or pca_components is None:
                raise ValueError("pca_mean and pca_components are required for pca mode")
            pca_mean = torch.as_tensor(pca_mean, dtype=torch.float32).view(1, -1)
            pca_components = torch.as_tensor(pca_components, dtype=torch.float32)
            if pca_mean.shape != (1, self.latent_size):
                raise ValueError(
                    f"Expected pca_mean shape [1,{self.latent_size}], got {tuple(pca_mean.shape)}"
                )
            if pca_components.shape != (self.latent_condition_dim, self.latent_size):
                raise ValueError(
                    "Expected pca_components shape "
                    f"[{self.latent_condition_dim},{self.latent_size}], "
                    f"got {tuple(pca_components.shape)}"
                )
            self.register_buffer("pca_mean", pca_mean)
            self.register_buffer("pca_components", pca_components)
        else:
            self.register_buffer("pca_mean", torch.zeros(1, self.latent_size))
            self.register_buffer(
                "pca_components",
                torch.empty(0, self.latent_size),
            )

        dims = [
            self.latent_condition_dim
            + 2
            + (1 if self.include_delta_time_input else 0)
            + self.condition_dim,
            *[int(width) for width in hidden_dims],
            self.latent_size,
        ]
        layers = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(input_dim, output_dim)
            layers.append(linear)
            if index < len(dims) - 2:
                layers.append(_activation(activation))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
        self.net = nn.Sequential(*layers)

        if zero_initialize_output:
            final_linear = next(
                layer for layer in reversed(self.net) if isinstance(layer, nn.Linear)
            )
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _condition(self, latent: torch.Tensor, condition: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.condition_dim == 0:
            return None
        if condition is None:
            return torch.zeros(
                latent.shape[0],
                self.condition_dim,
                device=latent.device,
                dtype=latent.dtype,
            )
        condition = condition.to(device=latent.device, dtype=latent.dtype)
        if condition.ndim == 1:
            condition = condition.unsqueeze(1)
        if condition.ndim != 2:
            raise ValueError(
                f"Expected rank-2 condition tensor, got {tuple(condition.shape)}"
            )
        if condition.shape[0] != latent.shape[0]:
            raise ValueError(
                "Condition batch size does not match latent batch size: "
                f"{condition.shape[0]} vs {latent.shape[0]}"
            )
        if condition.shape[1] == self.condition_dim:
            return condition
        if condition.shape[1] == 1 and self.condition_dim > 1:
            return condition.repeat(1, self.condition_dim)
        raise ValueError(
            f"Expected condition width {self.condition_dim}, got {condition.shape[1]}"
        )

    def _latent_condition(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        if self.latent_condition_mode == "none":
            return None
        if self.latent_condition_mode == "full":
            return latent
        if self.latent_condition_mode == "slice":
            return latent[:, : self.latent_condition_dim]
        if self.latent_condition_mode == "pca":
            return (latent - self.pca_mean.to(dtype=latent.dtype)) @ self.pca_components.to(
                dtype=latent.dtype
            ).T
        raise RuntimeError(f"Unhandled latent condition mode {self.latent_condition_mode!r}")

    def average_velocity(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_size:
            raise ValueError(
                f"Expected latent shape [B,{self.latent_size}], got {tuple(latent.shape)}"
            )
        source_time = _column(source_time, device=latent.device, dtype=latent.dtype)
        target_time = _column(target_time, device=latent.device, dtype=latent.dtype)
        if source_time.shape[0] != latent.shape[0] or target_time.shape[0] != latent.shape[0]:
            raise ValueError("Time batch size must match latent batch size")
        parts = []
        latent_condition = self._latent_condition(latent)
        if latent_condition is not None:
            parts.append(latent_condition)
        parts.extend([source_time, target_time])
        if self.include_delta_time_input:
            parts.append(target_time - source_time)
        prepared_condition = self._condition(latent, condition)
        if prepared_condition is not None:
            parts.append(prepared_condition)
        return self.net(torch.cat(parts, dim=1))

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source_time = _column(source_time, device=latent.device, dtype=latent.dtype)
        target_time = _column(target_time, device=latent.device, dtype=latent.dtype)
        velocity = self.average_velocity(
            latent,
            source_time,
            target_time,
            condition,
        )
        return latent + (target_time - source_time) * velocity

    def instantaneous_velocity_per_year(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor],
        age_range_years: float,
    ) -> torch.Tensor:
        if float(age_range_years) <= 0:
            raise ValueError("age_range_years must be positive")
        return (
            self.average_velocity(latent, time, time, condition)
            / float(age_range_years)
        )


def sample_virtual_intermediate(
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    *,
    minimum_ratio: float = 0.1,
    maximum_ratio: float = 0.9,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one strictly interior relative time for every batch row."""

    if not 0.0 < float(minimum_ratio) < float(maximum_ratio) < 1.0:
        raise ValueError(
            "Virtual midpoint ratios must satisfy 0 < minimum < maximum < 1"
        )
    source_time = source_time if source_time.ndim == 2 else source_time.unsqueeze(1)
    target_time = target_time if target_time.ndim == 2 else target_time.unsqueeze(1)
    if source_time.shape != target_time.shape or source_time.shape[1] != 1:
        raise ValueError("source_time and target_time must both have shape [B,1]")
    ratio = torch.rand(
        source_time.shape,
        device=source_time.device,
        dtype=source_time.dtype,
        generator=generator,
    )
    ratio = float(minimum_ratio) + ratio * (
        float(maximum_ratio) - float(minimum_ratio)
    )
    intermediate_time = source_time + ratio * (target_time - source_time)
    return intermediate_time, ratio


def direct_and_composed(
    flow: DirectAgeFlow,
    source_latent: torch.Tensor,
    source_time: torch.Tensor,
    intermediate_time: torch.Tensor,
    target_time: torch.Tensor,
    condition: Optional[torch.Tensor],
    *,
    direct_target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return direct target, predicted intermediate, and composed target."""

    direct = (
        direct_target
        if direct_target is not None
        else flow.transport(source_latent, source_time, target_time, condition)
    )
    intermediate = flow.transport(
        source_latent,
        source_time,
        intermediate_time,
        condition,
    )
    composed = flow.transport(
        intermediate,
        intermediate_time,
        target_time,
        condition,
    )
    return direct, intermediate, composed


def per_row_latent_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"Expected matching rank-2 tensors, got {left.shape} and {right.shape}"
        )
    return torch.mean((left - right) ** 2, dim=1)


@dataclass(frozen=True)
class MinimalLossConfig:
    real_prediction_weight: float = 1.0
    observed_consistency_weight: float = 0.01
    virtual_consistency_weight: float = 0.01
    virtual_ratio_min: float = 0.1
    virtual_ratio_max: float = 0.9
    backward_virtual_latent_weight: float = 0.0
    backward_virtual_shape_weight: float = 0.0
    backward_shape_samples: int = 1024
    future_extrapolation_latent_weight: float = 0.0
    future_extrapolation_shape_weight: float = 0.0
    future_extrapolation_ratio_min: float = 0.25
    future_extrapolation_ratio_max: float = 1.0
    future_shape_samples: int = 1024


@dataclass
class MinimalLossOutput:
    total: torch.Tensor
    real_prediction: torch.Tensor
    observed_consistency: torch.Tensor
    virtual_consistency: torch.Tensor
    backward_virtual_latent: torch.Tensor
    backward_virtual_shape: torch.Tensor
    future_extrapolation_latent: torch.Tensor
    future_extrapolation_shape: torch.Tensor
    direct_target_latent: torch.Tensor
    virtual_intermediate_latent: torch.Tensor
    virtual_composed_target_latent: torch.Tensor
    virtual_ratio: torch.Tensor
    future_ratio: torch.Tensor

    def detached_scalars(self) -> Dict[str, float]:
        return {
            "total": float(self.total.detach().cpu().item()),
            "real_prediction": float(
                self.real_prediction.detach().cpu().item()
            ),
            "observed_consistency": float(
                self.observed_consistency.detach().cpu().item()
            ),
            "virtual_consistency": float(
                self.virtual_consistency.detach().cpu().item()
            ),
            "backward_virtual_latent": float(
                self.backward_virtual_latent.detach().cpu().item()
            ),
            "backward_virtual_shape": float(
                self.backward_virtual_shape.detach().cpu().item()
            ),
            "future_extrapolation_latent": float(
                self.future_extrapolation_latent.detach().cpu().item()
            ),
            "future_extrapolation_shape": float(
                self.future_extrapolation_shape.detach().cpu().item()
            ),
        }


class MinimalDirectFlowLoss(nn.Module):
    """Compute enabled direct-flow losses from a frozen decoder."""

    def __init__(
        self,
        decoder: nn.Module,
        flow: DirectAgeFlow,
        clamp_distance: float,
        config: MinimalLossConfig,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.flow = flow
        self.clamp_distance = float(clamp_distance)
        self.config = config

    @staticmethod
    def zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def predict_real_target_latent(
        self,
        source_latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.flow.transport(
            source_latent,
            source_time,
            target_time,
            condition,
        )

    def decode_target_sdf_loss(
        self,
        latent: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode_target_sdf_loss_per_row(
            latent,
            target_samples,
        ).mean()

    def decode_sdf_at_xyz(
        self,
        latent: torch.Tensor,
        xyz: torch.Tensor,
    ) -> torch.Tensor:
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError(f"xyz must have shape [B,N,3], got {tuple(xyz.shape)}")
        if latent.ndim != 2 or latent.shape[0] != xyz.shape[0]:
            raise ValueError(
                "latent and xyz batch sizes must match; got "
                f"{tuple(latent.shape)} and {tuple(xyz.shape)}"
            )
        batch_size, sample_count, _ = xyz.shape
        expanded_latent = latent.unsqueeze(1).expand(
            batch_size,
            sample_count,
            latent.shape[1],
        )
        decoder_input = torch.cat([expanded_latent, xyz], dim=2).reshape(
            batch_size * sample_count,
            latent.shape[1] + 3,
        )
        prediction = self.decoder(decoder_input).reshape(
            batch_size,
            sample_count,
            -1,
        )
        return torch.clamp(
            prediction,
            -self.clamp_distance,
            self.clamp_distance,
        )

    def decode_target_sdf_loss_per_row(
        self,
        latent: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean absolute target-SDF error independently per pair."""

        if target_samples.ndim != 3 or target_samples.shape[2] < 4:
            raise ValueError(
                "target_samples must have shape [B,N,>=4], got "
                f"{tuple(target_samples.shape)}"
            )
        if latent.ndim != 2 or latent.shape[0] != target_samples.shape[0]:
            raise ValueError(
                "latent and target_samples batch sizes must match; got "
                f"{tuple(latent.shape)} and {tuple(target_samples.shape)}"
            )
        batch_size, sample_count, _ = target_samples.shape
        xyz = target_samples[:, :, :3]
        target_sdf = torch.clamp(
            target_samples[:, :, 3:4],
            -self.clamp_distance,
            self.clamp_distance,
        )
        prediction = self.decode_sdf_at_xyz(latent, xyz)
        return torch.mean(torch.abs(prediction - target_sdf), dim=(1, 2))

    def decode_latent_sdf_agreement_loss(
        self,
        left_latent: torch.Tensor,
        right_latent: torch.Tensor,
        target_samples: torch.Tensor,
        *,
        sample_count: int,
    ) -> torch.Tensor:
        """Compare two predicted shapes by SDF agreement at sampled xyz points."""

        if left_latent.shape != right_latent.shape or left_latent.ndim != 2:
            raise ValueError(
                "Expected matching rank-2 latents, got "
                f"{tuple(left_latent.shape)} and {tuple(right_latent.shape)}"
            )
        if target_samples.ndim != 3 or target_samples.shape[2] < 3:
            raise ValueError(
                "target_samples must have shape [B,N,>=3], got "
                f"{tuple(target_samples.shape)}"
            )
        available_count = int(target_samples.shape[1])
        selected_count = int(sample_count)
        if selected_count <= 0 or selected_count > available_count:
            selected_count = available_count
        xyz = target_samples[:, :selected_count, :3]
        left_sdf = self.decode_sdf_at_xyz(left_latent, xyz)
        right_sdf = self.decode_sdf_at_xyz(right_latent, xyz)
        return torch.mean(torch.abs(left_sdf - right_sdf))

    def observed_cocycle_loss(
        self,
        *,
        source_latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor],
        direct_target: torch.Tensor,
        intermediate_time: Optional[torch.Tensor],
        intermediate_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Consistency through a real age strictly inside a long pair."""

        zero = direct_target.sum() * 0.0
        if intermediate_time is None or intermediate_mask is None:
            return zero
        intermediate_time = _column(
            intermediate_time,
            device=source_latent.device,
            dtype=source_latent.dtype,
        )
        intermediate_mask = intermediate_mask.to(
            device=source_latent.device,
            dtype=torch.bool,
        ).view(-1)
        if intermediate_mask.shape[0] != source_latent.shape[0]:
            raise ValueError("Observed intermediate mask batch mismatch")
        if not bool(intermediate_mask.any()):
            return zero
        _, _, composed = direct_and_composed(
            self.flow,
            source_latent,
            source_time,
            intermediate_time,
            target_time,
            condition,
            direct_target=direct_target,
        )
        return per_row_latent_mse(
            direct_target,
            composed,
        )[intermediate_mask].mean()

    def virtual_cocycle_loss(
        self,
        *,
        source_latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor],
        direct_target: torch.Tensor,
        virtual_ratio: Optional[torch.Tensor],
        virtual_generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Consistency through one random or explicitly supplied virtual age."""

        if virtual_ratio is None:
            virtual_time, virtual_ratio = sample_virtual_intermediate(
                source_time,
                target_time,
                minimum_ratio=self.config.virtual_ratio_min,
                maximum_ratio=self.config.virtual_ratio_max,
                generator=virtual_generator,
            )
        else:
            virtual_ratio = _column(
                virtual_ratio,
                device=source_latent.device,
                dtype=source_latent.dtype,
            )
            virtual_time = source_time + virtual_ratio * (
                target_time - source_time
            )
        _, intermediate, composed = direct_and_composed(
            self.flow,
            source_latent,
            source_time,
            virtual_time,
            target_time,
            condition,
            direct_target=direct_target,
        )
        loss = per_row_latent_mse(direct_target, composed).mean()
        return loss, intermediate, composed, virtual_ratio

    def backward_virtual_agreement_losses(
        self,
        *,
        target_latent: Optional[torch.Tensor],
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor],
        virtual_ratio: torch.Tensor,
        source_time: torch.Tensor,
        source_virtual: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Agreement between source->virtual and target->virtual predictions."""

        zero = self.zero_like_loss(source_virtual)
        if (
            float(self.config.backward_virtual_latent_weight) == 0.0
            and float(self.config.backward_virtual_shape_weight) == 0.0
        ):
            return zero, zero
        if target_latent is None:
            raise ValueError(
                "target_latent is required when backward virtual losses are enabled"
            )
        target_latent = target_latent.to(
            device=source_virtual.device,
            dtype=source_virtual.dtype,
        )
        virtual_time = source_time + virtual_ratio * (target_time - source_time)
        target_virtual = self.flow.transport(
            target_latent,
            target_time,
            virtual_time,
            condition,
        )
        latent_loss = (
            per_row_latent_mse(source_virtual, target_virtual).mean()
            if float(self.config.backward_virtual_latent_weight) != 0.0
            else zero
        )
        shape_loss = (
            self.decode_latent_sdf_agreement_loss(
                source_virtual,
                target_virtual,
                target_samples,
                sample_count=int(self.config.backward_shape_samples),
            )
            if float(self.config.backward_virtual_shape_weight) != 0.0
            else zero
        )
        return latent_loss, shape_loss

    def future_extrapolation_consistency_losses(
        self,
        *,
        source_latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor],
        direct_target: torch.Tensor,
        target_samples: torch.Tensor,
        future_ratio: Optional[torch.Tensor],
        future_generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Future direct-vs-composed consistency beyond the observed target."""

        zero = self.zero_like_loss(direct_target)
        if (
            float(self.config.future_extrapolation_latent_weight) == 0.0
            and float(self.config.future_extrapolation_shape_weight) == 0.0
        ):
            if future_ratio is None:
                future_ratio = torch.zeros_like(source_time)
            else:
                future_ratio = _column(
                    future_ratio,
                    device=source_latent.device,
                    dtype=source_latent.dtype,
                )
            return zero, zero, future_ratio
        if future_ratio is None:
            if float(self.config.future_extrapolation_ratio_min) < 0.0:
                raise ValueError("Future extrapolation ratio min must be non-negative")
            if not (
                float(self.config.future_extrapolation_ratio_min)
                < float(self.config.future_extrapolation_ratio_max)
            ):
                raise ValueError(
                    "Future extrapolation ratios must satisfy minimum < maximum"
                )
            future_ratio = torch.rand(
                source_time.shape,
                device=source_time.device,
                dtype=source_time.dtype,
                generator=future_generator,
            )
            future_ratio = float(self.config.future_extrapolation_ratio_min) + (
                future_ratio
                * (
                    float(self.config.future_extrapolation_ratio_max)
                    - float(self.config.future_extrapolation_ratio_min)
                )
            )
        else:
            future_ratio = _column(
                future_ratio,
                device=source_latent.device,
                dtype=source_latent.dtype,
            )
        future_time = target_time + future_ratio * (target_time - source_time)
        future_direct = self.flow.transport(
            source_latent,
            source_time,
            future_time,
            condition,
        )
        future_composed = self.flow.transport(
            direct_target,
            target_time,
            future_time,
            condition,
        )
        latent_loss = (
            per_row_latent_mse(future_direct, future_composed).mean()
            if float(self.config.future_extrapolation_latent_weight) != 0.0
            else zero
        )
        shape_loss = (
            self.decode_latent_sdf_agreement_loss(
                future_direct,
                future_composed,
                target_samples,
                sample_count=int(self.config.future_shape_samples),
            )
            if float(self.config.future_extrapolation_shape_weight) != 0.0
            else zero
        )
        return latent_loss, shape_loss, future_ratio

    def forward(
        self,
        *,
        source_latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor],
        target_samples: torch.Tensor,
        target_latent: Optional[torch.Tensor] = None,
        observed_intermediate_time: Optional[torch.Tensor] = None,
        observed_intermediate_mask: Optional[torch.Tensor] = None,
        virtual_ratio: Optional[torch.Tensor] = None,
        virtual_generator: Optional[torch.Generator] = None,
        future_ratio: Optional[torch.Tensor] = None,
        future_generator: Optional[torch.Generator] = None,
    ) -> MinimalLossOutput:
        source_time = _column(
            source_time,
            device=source_latent.device,
            dtype=source_latent.dtype,
        )
        target_time = _column(
            target_time,
            device=source_latent.device,
            dtype=source_latent.dtype,
        )
        direct_target = self.predict_real_target_latent(
            source_latent,
            source_time,
            target_time,
            condition,
        )
        real_prediction = self.decode_target_sdf_loss(
            direct_target,
            target_samples,
        )

        observed_consistency = self.observed_cocycle_loss(
            source_latent=source_latent,
            source_time=source_time,
            target_time=target_time,
            condition=condition,
            direct_target=direct_target,
            intermediate_time=observed_intermediate_time,
            intermediate_mask=observed_intermediate_mask,
        )
        (
            virtual_consistency,
            virtual_intermediate,
            virtual_composed,
            virtual_ratio,
        ) = self.virtual_cocycle_loss(
            source_latent=source_latent,
            source_time=source_time,
            target_time=target_time,
            condition=condition,
            direct_target=direct_target,
            virtual_ratio=virtual_ratio,
            virtual_generator=virtual_generator,
        )
        (
            backward_virtual_latent,
            backward_virtual_shape,
        ) = self.backward_virtual_agreement_losses(
            target_latent=target_latent,
            target_time=target_time,
            condition=condition,
            virtual_ratio=virtual_ratio,
            source_time=source_time,
            source_virtual=virtual_intermediate,
            target_samples=target_samples,
        )
        (
            future_extrapolation_latent,
            future_extrapolation_shape,
            future_ratio,
        ) = self.future_extrapolation_consistency_losses(
            source_latent=source_latent,
            source_time=source_time,
            target_time=target_time,
            condition=condition,
            direct_target=direct_target,
            target_samples=target_samples,
            future_ratio=future_ratio,
            future_generator=future_generator,
        )

        total = (
            float(self.config.real_prediction_weight) * real_prediction
            + float(self.config.observed_consistency_weight)
            * observed_consistency
            + float(self.config.virtual_consistency_weight)
            * virtual_consistency
            + float(self.config.backward_virtual_latent_weight)
            * backward_virtual_latent
            + float(self.config.backward_virtual_shape_weight)
            * backward_virtual_shape
            + float(self.config.future_extrapolation_latent_weight)
            * future_extrapolation_latent
            + float(self.config.future_extrapolation_shape_weight)
            * future_extrapolation_shape
        )
        return MinimalLossOutput(
            total=total,
            real_prediction=real_prediction,
            observed_consistency=observed_consistency,
            virtual_consistency=virtual_consistency,
            backward_virtual_latent=backward_virtual_latent,
            backward_virtual_shape=backward_virtual_shape,
            future_extrapolation_latent=future_extrapolation_latent,
            future_extrapolation_shape=future_extrapolation_shape,
            direct_target_latent=direct_target,
            virtual_intermediate_latent=virtual_intermediate,
            virtual_composed_target_latent=virtual_composed,
            virtual_ratio=virtual_ratio,
            future_ratio=future_ratio,
        )
