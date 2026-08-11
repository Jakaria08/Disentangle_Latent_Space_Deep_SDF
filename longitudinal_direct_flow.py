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


class LatentODEFlow(nn.Module):
    """Neural ODE transport over frozen scan latents.

    The vector field is instantaneous:

        dz / dt = f(z(t), t, c)

    and ``transport`` integrates it from source age to target age with RK4.
    The class intentionally matches the direct-flow API so existing training,
    evaluation, and visualization code can compare ODE and flow transports.
    """

    def __init__(
        self,
        latent_size: int,
        hidden_dims: Sequence[int],
        condition_dim: int = 1,
        activation: str = "silu",
        dropout: float = 0.0,
        zero_initialize_output: bool = True,
        latent_condition_mode: str = "full",
        latent_condition_dim: Optional[int] = None,
        integration_substeps: int = 4,
        max_step_norm: Optional[float] = None,
        pca_mean: Optional[torch.Tensor] = None,
        pca_components: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if int(latent_size) <= 0:
            raise ValueError("latent_size must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if int(integration_substeps) < 1:
            raise ValueError("integration_substeps must be at least 1")
        if max_step_norm is not None and float(max_step_norm) <= 0.0:
            raise ValueError("max_step_norm must be positive when provided")

        self.latent_size = int(latent_size)
        self.condition_dim = max(0, int(condition_dim))
        self.integration_substeps = int(integration_substeps)
        self.max_step_norm = None if max_step_norm is None else float(max_step_norm)
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
            self.register_buffer("pca_components", torch.empty(0, self.latent_size))

        dims = [
            self.latent_condition_dim + 1 + self.condition_dim,
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

    def ode_velocity(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_size:
            raise ValueError(
                f"Expected latent shape [B,{self.latent_size}], got {tuple(latent.shape)}"
            )
        time = _column(time, device=latent.device, dtype=latent.dtype)
        if time.shape[0] != latent.shape[0]:
            raise ValueError("Time batch size must match latent batch size")
        parts = []
        latent_condition = self._latent_condition(latent)
        if latent_condition is not None:
            parts.append(latent_condition)
        parts.append(time)
        prepared_condition = self._condition(latent, condition)
        if prepared_condition is not None:
            parts.append(prepared_condition)
        return self.net(torch.cat(parts, dim=1))

    def _rk4_step(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        delta: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        delta = _column(delta, device=latent.device, dtype=latent.dtype)
        time = _column(time, device=latent.device, dtype=latent.dtype)
        half_delta = 0.5 * delta
        k1 = self.ode_velocity(latent, time, condition)
        k2 = self.ode_velocity(latent + half_delta * k1, time + half_delta, condition)
        k3 = self.ode_velocity(latent + half_delta * k2, time + half_delta, condition)
        k4 = self.ode_velocity(latent + delta * k3, time + delta, condition)
        return latent + delta * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def average_velocity(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del target_time
        return self.ode_velocity(latent, source_time, condition)

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_latent = latent
        current_time = _column(source_time, device=latent.device, dtype=latent.dtype)
        final_time = _column(target_time, device=latent.device, dtype=latent.dtype)
        if current_time.shape[0] != latent.shape[0] or final_time.shape[0] != latent.shape[0]:
            raise ValueError("Time batch size must match latent batch size")

        if self.max_step_norm is None:
            step_count = self.integration_substeps
            delta = (final_time - current_time) / float(step_count)
            for _ in range(step_count):
                current_latent = self._rk4_step(
                    current_latent,
                    current_time,
                    delta,
                    condition,
                )
                current_time = current_time + delta
            return current_latent

        max_steps = int(
            torch.ceil(
                (final_time - current_time).abs().max() / float(self.max_step_norm)
            ).item()
        )
        if max_steps <= 0:
            return current_latent
        max_step = torch.full_like(current_time, float(self.max_step_norm))
        for _ in range(max_steps):
            remaining = final_time - current_time
            active = remaining.abs() > 1.0e-8
            if not bool(active.any().item()):
                break
            step_delta = torch.minimum(remaining.abs(), max_step) * torch.sign(remaining)
            candidate = self._rk4_step(
                current_latent,
                current_time,
                step_delta,
                condition,
            )
            next_time = current_time + step_delta
            current_latent = torch.where(active, candidate, current_latent)
            current_time = torch.where(active, next_time, current_time)
        return current_latent

    def instantaneous_velocity_per_year(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor],
        age_range_years: float,
    ) -> torch.Tensor:
        if float(age_range_years) <= 0:
            raise ValueError("age_range_years must be positive")
        return self.ode_velocity(latent, time, condition) / float(age_range_years)


class LocalDiseaseDecomposedFlow(nn.Module):
    """Composed local flow with healthy velocity plus AD residual velocity.

    The transport is integrated by fixed small steps:

        z(t + h) = z(t) + h * B(g_cn(z(t), t) + c * g_ad(z(t), t)).

    ``B`` is a learned low-rank map from ``rank`` progression coordinates back
    to the full frozen-decoder latent space.  The condition ``c`` is expected
    to be 0 for CN and 1 for AD, which makes counterfactual CN/AD rollout from
    the same source latent explicit.
    """

    def __init__(
        self,
        latent_size: int,
        hidden_dims: Sequence[int],
        rank: int = 32,
        condition_dim: int = 1,
        activation: str = "silu",
        dropout: float = 0.0,
        zero_initialize_output: bool = True,
        latent_condition_mode: str = "full",
        latent_condition_dim: Optional[int] = None,
        include_time_input: bool = True,
        max_step_norm: float = 0.0125,
        basis_init_scale: float = 0.02,
        pca_mean: Optional[torch.Tensor] = None,
        pca_components: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if int(latent_size) <= 0:
            raise ValueError("latent_size must be positive")
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if int(condition_dim) not in {0, 1}:
            raise ValueError("LocalDiseaseDecomposedFlow expects ConditionDim 0 or 1")
        if float(max_step_norm) <= 0.0:
            raise ValueError("max_step_norm must be positive")

        self.latent_size = int(latent_size)
        self.rank = int(rank)
        self.condition_dim = int(condition_dim)
        self.include_time_input = bool(include_time_input)
        self.max_step_norm = float(max_step_norm)
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

        input_dim = self.latent_condition_dim + (1 if self.include_time_input else 0)
        self.cn_net = self._build_branch(
            input_dim=input_dim,
            hidden_dims=[int(value) for value in hidden_dims],
            activation=activation,
            dropout=float(dropout),
            zero_initialize_output=bool(zero_initialize_output),
        )
        self.ad_residual_net = self._build_branch(
            input_dim=input_dim,
            hidden_dims=[int(value) for value in hidden_dims],
            activation=activation,
            dropout=float(dropout),
            zero_initialize_output=bool(zero_initialize_output),
        )
        self.basis = nn.Linear(self.rank, self.latent_size, bias=False)
        nn.init.normal_(
            self.basis.weight,
            mean=0.0,
            std=float(basis_init_scale) / max(float(self.rank) ** 0.5, 1.0),
        )

    def _build_branch(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
        zero_initialize_output: bool,
    ) -> nn.Sequential:
        dims = [int(input_dim), *[int(width) for width in hidden_dims], self.rank]
        layers = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(left, right)
            layers.append(linear)
            if index < len(dims) - 2:
                layers.append(_activation(activation))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
        if zero_initialize_output:
            final_linear = next(
                layer for layer in reversed(layers) if isinstance(layer, nn.Linear)
            )
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
        return nn.Sequential(*layers)

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

    def _branch_input(self, latent: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_size:
            raise ValueError(
                f"Expected latent shape [B,{self.latent_size}], got {tuple(latent.shape)}"
            )
        time = _column(time, device=latent.device, dtype=latent.dtype)
        if time.shape[0] != latent.shape[0]:
            raise ValueError("Time batch size must match latent batch size")
        parts = []
        latent_condition = self._latent_condition(latent)
        if latent_condition is not None:
            parts.append(latent_condition)
        if self.include_time_input:
            parts.append(time)
        if not parts:
            return torch.empty(latent.shape[0], 0, device=latent.device, dtype=latent.dtype)
        return torch.cat(parts, dim=1)

    def _condition_gate(
        self,
        latent: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.condition_dim == 0:
            return torch.zeros(latent.shape[0], 1, device=latent.device, dtype=latent.dtype)
        if condition is None:
            return torch.zeros(latent.shape[0], 1, device=latent.device, dtype=latent.dtype)
        condition = condition.to(device=latent.device, dtype=latent.dtype)
        if condition.ndim == 1:
            condition = condition.unsqueeze(1)
        if condition.ndim != 2 or condition.shape[1] != 1:
            raise ValueError(
                f"Expected scalar condition column, got {tuple(condition.shape)}"
            )
        if condition.shape[0] != latent.shape[0]:
            raise ValueError(
                "Condition batch size does not match latent batch size: "
                f"{condition.shape[0]} vs {latent.shape[0]}"
            )
        return condition

    def low_rank_velocity(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_input = self._branch_input(latent, time)
        cn_rank = self.cn_net(branch_input)
        ad_rank = self.ad_residual_net(branch_input)
        gate = self._condition_gate(latent, condition)
        total_rank = cn_rank + gate * ad_rank
        return total_rank, cn_rank, ad_rank

    def decomposed_velocity(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_rank, cn_rank, ad_rank = self.low_rank_velocity(latent, time, condition)
        return self.basis(total_rank), self.basis(cn_rank), self.basis(ad_rank)

    def average_velocity(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del target_time
        total_velocity, _, _ = self.decomposed_velocity(latent, source_time, condition)
        return total_velocity

    def step(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source_time = _column(source_time, device=latent.device, dtype=latent.dtype)
        target_time = _column(target_time, device=latent.device, dtype=latent.dtype)
        velocity = self.average_velocity(latent, source_time, source_time, condition)
        return latent + (target_time - source_time) * velocity

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_latent = latent
        current_time = _column(source_time, device=latent.device, dtype=latent.dtype)
        final_time = _column(target_time, device=latent.device, dtype=latent.dtype)
        if current_time.shape[0] != latent.shape[0] or final_time.shape[0] != latent.shape[0]:
            raise ValueError("Time batch size must match latent batch size")

        max_steps = int(
            torch.ceil(
                (final_time - current_time).abs().max() / float(self.max_step_norm)
            ).item()
        )
        if max_steps <= 0:
            return current_latent
        max_step = torch.full_like(current_time, float(self.max_step_norm))
        for _ in range(max_steps):
            remaining = final_time - current_time
            active = remaining.abs() > 1.0e-8
            if not bool(active.any().item()):
                break
            step_delta = torch.minimum(remaining.abs(), max_step) * torch.sign(remaining)
            next_time = current_time + step_delta
            candidate = self.step(current_latent, current_time, next_time, condition)
            current_latent = torch.where(active, candidate, current_latent)
            current_time = torch.where(active, next_time, current_time)
        return current_latent

    def instantaneous_velocity_per_year(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor],
        age_range_years: float,
    ) -> torch.Tensor:
        if float(age_range_years) <= 0:
            raise ValueError("age_range_years must be positive")
        return self.average_velocity(latent, time, time, condition) / float(age_range_years)

    def decomposed_velocity_per_year(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor],
        age_range_years: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if float(age_range_years) <= 0:
            raise ValueError("age_range_years must be positive")
        total, cn, ad = self.decomposed_velocity(latent, time, condition)
        scale = float(age_range_years)
        return total / scale, cn / scale, ad / scale


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
    change_weighted_sdf_weight: float = 0.0
    change_weighted_sdf_alpha: float = 2.0
    change_weighted_sdf_eps: float = 1.0e-5
    delta_sdf_direction_weight: float = 0.0
    delta_sdf_min_norm: float = 1.0e-6
    delta_sdf_rmae_weight: float = 0.0
    delta_sdf_rmae_eps: float = 1.0e-4
    delta_sdf_rmae_cap: float = 5.0
    no_change_margin_weight: float = 0.0
    no_change_margin: float = 1.0e-4
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
    change_weighted_sdf: torch.Tensor
    delta_sdf_direction: torch.Tensor
    delta_sdf_rmae: torch.Tensor
    no_change_margin: torch.Tensor
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
            "change_weighted_sdf": float(
                self.change_weighted_sdf.detach().cpu().item()
            ),
            "delta_sdf_direction": float(
                self.delta_sdf_direction.detach().cpu().item()
            ),
            "delta_sdf_rmae": float(
                self.delta_sdf_rmae.detach().cpu().item()
            ),
            "no_change_margin": float(
                self.no_change_margin.detach().cpu().item()
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

    def change_aware_sdf_losses_per_row(
        self,
        *,
        source_latent: torch.Tensor,
        predicted_latent: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return per-pair losses that focus on source-to-target SDF change."""

        if target_samples.ndim != 3 or target_samples.shape[2] < 4:
            raise ValueError(
                "target_samples must have shape [B,N,>=4], got "
                f"{tuple(target_samples.shape)}"
            )
        if source_latent.shape != predicted_latent.shape or source_latent.ndim != 2:
            raise ValueError(
                "Expected matching rank-2 latents, got "
                f"{tuple(source_latent.shape)} and {tuple(predicted_latent.shape)}"
            )
        if source_latent.shape[0] != target_samples.shape[0]:
            raise ValueError(
                "latent and target_samples batch sizes must match; got "
                f"{tuple(source_latent.shape)} and {tuple(target_samples.shape)}"
            )
        xyz = target_samples[:, :, :3]
        target_sdf = torch.clamp(
            target_samples[:, :, 3:4],
            -self.clamp_distance,
            self.clamp_distance,
        )
        predicted_sdf = self.decode_sdf_at_xyz(predicted_latent, xyz)
        source_sdf = self.decode_sdf_at_xyz(source_latent, xyz).detach()

        absolute_error = torch.abs(predicted_sdf - target_sdf)
        delta_gt = target_sdf - source_sdf
        delta_pred = predicted_sdf - source_sdf

        delta_scale = torch.mean(
            torch.abs(delta_gt),
            dim=(1, 2),
            keepdim=True,
        )
        weights = 1.0 + float(self.config.change_weighted_sdf_alpha) * (
            torch.abs(delta_gt)
            / (delta_scale + float(self.config.change_weighted_sdf_eps))
        )
        change_weighted_sdf = torch.mean(
            weights.detach() * absolute_error,
            dim=(1, 2),
        )

        flat_delta_gt = delta_gt.reshape(delta_gt.shape[0], -1)
        flat_delta_pred = delta_pred.reshape(delta_pred.shape[0], -1)
        gt_norm = torch.linalg.vector_norm(flat_delta_gt, dim=1)
        pred_norm = torch.linalg.vector_norm(flat_delta_pred, dim=1)
        gt_valid = gt_norm > float(self.config.delta_sdf_min_norm)
        valid = gt_valid & (pred_norm > float(self.config.delta_sdf_min_norm))
        delta_sdf_direction = flat_delta_gt.new_zeros(flat_delta_gt.shape[0])
        delta_sdf_direction[gt_valid] = 1.0
        if bool(valid.any().item()):
            delta_sdf_direction[valid] = 1.0 - torch.nn.functional.cosine_similarity(
                flat_delta_pred[valid],
                flat_delta_gt[valid],
                dim=1,
                eps=1.0e-8,
            )

        denominator = (
            0.5 * (torch.abs(delta_pred) + torch.abs(delta_gt))
            + float(self.config.delta_sdf_rmae_eps)
        )
        delta_sdf_rmae = torch.abs(delta_pred - delta_gt) / denominator
        cap = float(self.config.delta_sdf_rmae_cap)
        if cap > 0.0:
            delta_sdf_rmae = torch.clamp(delta_sdf_rmae, max=cap)
        delta_sdf_rmae = torch.mean(delta_sdf_rmae, dim=(1, 2))

        prediction_l1 = torch.mean(absolute_error, dim=(1, 2))
        no_change_l1 = torch.mean(torch.abs(source_sdf - target_sdf), dim=(1, 2))
        no_change_margin = torch.relu(
            prediction_l1 - no_change_l1 + float(self.config.no_change_margin)
        ) ** 2

        return {
            "change_weighted_sdf": change_weighted_sdf,
            "delta_sdf_direction": delta_sdf_direction,
            "delta_sdf_rmae": delta_sdf_rmae,
            "no_change_margin": no_change_margin,
        }

    def change_aware_sdf_losses(
        self,
        *,
        source_latent: torch.Tensor,
        predicted_latent: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = self.change_aware_sdf_losses_per_row(
            source_latent=source_latent,
            predicted_latent=predicted_latent,
            target_samples=target_samples,
        )
        return {key: value.mean() for key, value in losses.items()}

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
        change_losses_enabled = any(
            float(weight) != 0.0
            for weight in (
                self.config.change_weighted_sdf_weight,
                self.config.delta_sdf_direction_weight,
                self.config.delta_sdf_rmae_weight,
                self.config.no_change_margin_weight,
            )
        )
        if change_losses_enabled:
            change_losses = self.change_aware_sdf_losses(
                source_latent=source_latent,
                predicted_latent=direct_target,
                target_samples=target_samples,
            )
            change_weighted_sdf = change_losses["change_weighted_sdf"]
            delta_sdf_direction = change_losses["delta_sdf_direction"]
            delta_sdf_rmae = change_losses["delta_sdf_rmae"]
            no_change_margin = change_losses["no_change_margin"]
        else:
            zero = self.zero_like_loss(real_prediction)
            change_weighted_sdf = zero
            delta_sdf_direction = zero
            delta_sdf_rmae = zero
            no_change_margin = zero

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
            + float(self.config.change_weighted_sdf_weight)
            * change_weighted_sdf
            + float(self.config.delta_sdf_direction_weight)
            * delta_sdf_direction
            + float(self.config.delta_sdf_rmae_weight)
            * delta_sdf_rmae
            + float(self.config.no_change_margin_weight)
            * no_change_margin
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
            change_weighted_sdf=change_weighted_sdf,
            delta_sdf_direction=delta_sdf_direction,
            delta_sdf_rmae=delta_sdf_rmae,
            no_change_margin=no_change_margin,
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
