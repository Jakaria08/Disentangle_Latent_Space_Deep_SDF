"""Matched SIREN-256 transport models restricted to one fixed velocity basis."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape != (reference.shape[0], 1):
        raise ValueError(f"Expected [B,1] time/condition, got {tuple(value.shape)}")
    return value


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            [nn.ModuleList((nn.Linear(width, width), nn.Linear(width, width))) for _ in range(blocks)]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output = nn.Linear(width, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.gelu(self.input(value))
        for first, second in self.blocks:
            residual = value
            value = torch.nn.functional.gelu(first(value))
            value = self.dropout(value)
            value = second(value)
            value = torch.nn.functional.gelu(value + residual)
        return self.output(value)


class PCAParityDirectFlow(nn.Module):
    """Full-dimensional direct flow matching the successful PCA flow API.

    SIREN latent coordinates are standardized with train-only statistics, but
    neither the input state nor the predicted velocity is projected to a
    fixed basis.  In standardized coordinates the transport is

        u_t = u_s + (t - s) G(u_s, s, t, t-s, d).

    This is deliberately a single small MLP with a scalar diagnosis input,
    mirroring ``DirectAgeFlow`` used by the PCA-150 experiment.
    """

    attention_contract = "not_applicable"

    def __init__(
        self,
        latent_size: int,
        hidden_dims: list[int],
        dropout: float,
        latent_mean: torch.Tensor | None = None,
        latent_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        mean = torch.zeros(1, self.latent_size) if latent_mean is None else torch.as_tensor(latent_mean, dtype=torch.float32).reshape(1, -1)
        scale = torch.ones(1, self.latent_size) if latent_scale is None else torch.as_tensor(latent_scale, dtype=torch.float32).reshape(1, -1)
        if mean.shape != (1, self.latent_size) or scale.shape != (1, self.latent_size):
            raise ValueError("PCA-parity latent statistics do not match LatentSize.")
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_scale", scale.clamp_min(1.0e-6))

        dims = [self.latent_size + 4, *[int(width) for width in hidden_dims], self.latent_size]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(input_dim, output_dim)
            layers.append(linear)
            if index < len(dims) - 2:
                layers.append(nn.SiLU(inplace=False))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)
        final = next(layer for layer in reversed(self.net) if isinstance(layer, nn.Linear))
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def standardized(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_size:
            raise ValueError(f"Expected latent [B,{self.latent_size}], got {tuple(latent.shape)}")
        return (latent - self.latent_mean.to(latent.dtype)) / self.latent_scale.to(latent.dtype)

    def standardized_velocity(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        source_time, target_time, condition = (_column(value, latent) for value in (source_time, target_time, condition))
        delta = target_time - source_time
        return self.net(torch.cat((self.standardized(latent), source_time, target_time, delta, condition), dim=1))

    def velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.standardized_velocity(latent, time, time, condition) * self.latent_scale.to(latent.dtype)

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        source_time, target_time = (_column(value, latent) for value in (source_time, target_time))
        velocity = self.standardized_velocity(latent, source_time, target_time, condition)
        return latent + (target_time - source_time) * velocity * self.latent_scale.to(latent.dtype)


class BasisTransport(nn.Module):
    """Common transport API.  The orthogonal source component is immutable."""

    attention_contract = "not_applicable"

    def __init__(self, basis: dict[str, Any]) -> None:
        super().__init__()
        velocity_basis = torch.as_tensor(basis["velocity_basis"], dtype=torch.float32)
        feature_mean = torch.as_tensor(basis["feature_mean"], dtype=torch.float32)
        feature_components = torch.as_tensor(basis["feature_components"], dtype=torch.float32)
        feature_scale = torch.as_tensor(basis["feature_scale"], dtype=torch.float32)
        if velocity_basis.ndim != 2 or feature_components.ndim != 2:
            raise ValueError("Invalid train-only basis file.")
        self.register_buffer("velocity_basis", velocity_basis)
        self.register_buffer("feature_mean", feature_mean.reshape(1, -1))
        self.register_buffer("feature_components", feature_components)
        self.register_buffer("feature_scale", feature_scale.reshape(1, -1).clamp_min(1.0e-6))

    @property
    def latent_size(self) -> int:
        return int(self.velocity_basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.velocity_basis.shape[1])

    @property
    def feature_size(self) -> int:
        return int(self.feature_components.shape[0])

    def features(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_size:
            raise ValueError(f"Expected latent [B,{self.latent_size}], got {tuple(latent.shape)}")
        raw = (latent - self.feature_mean.to(latent.dtype)) @ self.feature_components.to(latent.dtype).T
        return raw / self.feature_scale.to(latent.dtype)

    def velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def transport(self, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _lift(self, coefficients: torch.Tensor) -> torch.Tensor:
        return coefficients @ self.velocity_basis.to(coefficients.dtype).T


class V5DirectResidualFlow(BasisTransport):
    """z + (t-s) B [g_CN(Pz,s,t) + d g_AD(Pz,s,t)]."""

    def __init__(self, basis: dict[str, Any], width: int, blocks: int, dropout: float) -> None:
        super().__init__(basis)
        input_dim = self.feature_size + 2
        self.cn = ResidualMLP(input_dim, width, blocks, self.rank, dropout)
        self.ad = ResidualMLP(input_dim, width, blocks, self.rank, dropout)

    def coefficients(self, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        source_time, target_time, condition = (_column(v, latent) for v in (source_time, target_time, condition))
        inputs = torch.cat((self.features(latent), source_time, target_time), dim=1)
        return self.cn(inputs) + condition * self.ad(inputs)

    def velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self._lift(self.coefficients(latent, time, time, condition))

    def transport(self, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        source_time, target_time = (_column(v, latent) for v in (source_time, target_time))
        return latent + (target_time - source_time) * self._lift(self.coefficients(latent, source_time, target_time, condition))


class RK4BasisODE(BasisTransport):
    """Basis-constrained neural ODE with exactly the configured RK4 substeps."""

    def __init__(self, basis: dict[str, Any], substeps: int) -> None:
        super().__init__(basis)
        if substeps != 4:
            raise ValueError("This matched protocol fixes RK4Substeps to 4.")
        self.substeps = int(substeps)

    def coefficient_velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self._lift(self.coefficient_velocity(latent, _column(time, latent), _column(condition, latent)))

    def transport(self, latent: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        source_time, target_time, condition = (_column(v, latent) for v in (source_time, target_time, condition))
        state, current = latent, source_time
        delta = (target_time - source_time) / float(self.substeps)
        for _ in range(self.substeps):
            k1 = self.velocity(state, current, condition)
            k2 = self.velocity(state + 0.5 * delta * k1, current + 0.5 * delta, condition)
            k3 = self.velocity(state + 0.5 * delta * k2, current + 0.5 * delta, condition)
            k4 = self.velocity(state + delta * k3, current + delta, condition)
            state = state + delta * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            current = current + delta
        return state


class PlainMatchedODE(RK4BasisODE):
    def __init__(self, basis: dict[str, Any], width: int, blocks: int, dropout: float, substeps: int) -> None:
        super().__init__(basis, substeps)
        self.net = ResidualMLP(self.feature_size + 2, width, blocks, self.rank, dropout)

    def coefficient_velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((self.features(latent), time, condition), dim=1))


class BrainODEAttention(RK4BasisODE):
    """Released BrainODE Q/K/V form under enforced singleton-token semantics."""

    attention_contract = "one_case_one_trajectory; singleton QKV; no cross-subject batch attention"

    def __init__(self, basis: dict[str, Any], attention_dim: int, width: int, dropout: float, substeps: int) -> None:
        super().__init__(basis, substeps)
        input_dim = self.feature_size + 2
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = attention_dim ** -0.5
        self.fc1 = nn.Linear(attention_dim, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(width, self.rank)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def singleton_attention_coefficients(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != 1:
            raise ValueError("BrainODE Q/K/V must receive one subject state at a time.")
        query, key, projected = self.query(value), self.key(value), self.value(value)
        attention = torch.softmax(query @ key.T * self.scale, dim=-1)
        hidden = attention @ projected
        return self.fc2(self.dropout(torch.nn.functional.gelu(self.fc1(hidden))))

    def coefficient_velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # A Python loop makes the no-cross-subject semantic explicit even when
        # several independent pairs are placed in the optimization batch.
        output = []
        for row in range(latent.shape[0]):
            token = torch.cat((self.features(latent[row : row + 1]), time[row : row + 1], condition[row : row + 1]), dim=1)
            output.append(self.singleton_attention_coefficients(token))
        return torch.cat(output, dim=0)

    def velocity(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        time, condition = _column(time, latent), _column(condition, latent)
        return torch.cat([self._lift(self.coefficient_velocity(latent[row : row + 1], time[row : row + 1], condition[row : row + 1])) for row in range(latent.shape[0])], dim=0)


def build_transport(config: dict[str, Any], basis: dict[str, Any]) -> BasisTransport:
    kind = str(config["ModelType"]).lower()
    if kind == "pca_parity_direct_flow":
        return PCAParityDirectFlow(
            latent_size=int(config["LatentSize"]),
            hidden_dims=[int(value) for value in config.get("HiddenDims", [128, 128])],
            dropout=float(config.get("Dropout", 0.05)),
            latent_mean=basis.get("latent_mean"),
            latent_scale=basis.get("latent_scale"),
        )
    width, blocks, dropout = int(config["HiddenWidth"]), int(config["ResidualBlocks"]), float(config["Dropout"])
    if kind == "v5_direct_flow":
        return V5DirectResidualFlow(basis, width, blocks, dropout)
    if kind == "plain_ode":
        return PlainMatchedODE(basis, width, blocks, dropout, int(config["RK4Substeps"]))
    if kind == "brainode_attention_ode":
        return BrainODEAttention(basis, int(config["AttentionDimensions"]), width, dropout, int(config["RK4Substeps"]))
    raise ValueError(f"Unknown ModelType: {config['ModelType']!r}")
