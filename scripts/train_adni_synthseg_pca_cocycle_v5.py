#!/usr/bin/env python3
"""Train direct or coboundary non-ODE flows on strict ADNI SynthSeg PCA data.

The learned transport is deliberately the same direct-flow family used by the
earlier cocycle experiment:

    Phi(z, s, t, d) = z + (t - s) * phi_theta(z, s, t, d)

Two optional C4-derived coboundary experiments are also supported:

``c4_coboundary_exact``
    Phi(z,s,t;u,d) = z + P(u,t,d) - P(u,s,d), where ``u`` is the fixed
    first-visit PCA context for the subject.  Identity, semigroup consistency,
    and reversal are exact up to floating-point arithmetic.

``c4_coboundary_soft``
    Retains the state-dependent direct V5 transport and adds a potential
    network.  A coboundary loss aligns the direct displacement with
    P(u,t,d)-P(u,s,d), preserving flexibility while encouraging, but not
    guaranteeing, coboundary structure.

There is no ODE solver, recurrence hidden inside the transport, attention over
subjects, or cross-run state.  Every output directory belongs to one
(structure, experiment, run-name) tuple, so independent experiments can be
started concurrently on separate GPUs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_adni_synthseg_pca_cocycle_v4 import (
    PairRow,
    atomic_json,
    atomic_torch_save,
    choose_device,
    load_archive,
    load_pairs,
    read_json,
    set_seed,
    validate_pca_model,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
BASE_ROOT = PROJECT_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
EXPERIMENTS = (
    "c1",
    "c2",
    "c3",
    "c4",
    "c4_time_embedded",
    "c4_coboundary_exact",
    "c4_coboundary_soft",
    "c5_coboundary_exact_volume",
    "c6_coboundary_exact_volume_selection",
    "c7_coboundary_volume_axis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=("hippocampus", "lateral_ventricle"))
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="c3")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", default=None, help="Optional new final directory-name component.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Validate one gradient step and validation pass without writing files.")
    parser.add_argument("--resume", action="store_true", help="Resume only the exact existing output directory.")
    return parser.parse_args()


def default_config_path(structure: str, experiment: str) -> Path:
    return BASE_ROOT / f"{structure}_pca_cocycle_v4" / "cocycle_v5" / "configs" / f"{experiment}_cocycle_v5.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_name(value: str) -> None:
    candidate = Path(value)
    if value in {"", ".", ".."} or candidate.name != value or "/" in value or "\\" in value:
        raise ValueError("--run-name must be one safe directory-name component")


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.norm(value)
        value = self.activation(self.fc1(value))
        value = self.fc2(value)
        return self.activation(residual + value)


class DirectDiagnosisResidualCocycleFlow(nn.Module):
    """One-shot direct flow with shared CN and residual AD average velocities."""

    def __init__(self, latent_dim: int, width: int, residual_blocks: int) -> None:
        super().__init__()
        if latent_dim <= 0 or width <= 0 or residual_blocks <= 0:
            raise ValueError("latent_dim, width, and residual_blocks must be positive")
        # z, source age, target age, signed and absolute gap, and midpoint age.
        self.latent_dim = int(latent_dim)
        self.input = nn.Linear(self.latent_dim + 5, int(width))
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList(ResidualBlock(int(width)) for _ in range(int(residual_blocks)))
        self.cn_head = nn.Linear(int(width), self.latent_dim)
        self.ad_residual_head = nn.Linear(int(width), self.latent_dim)
        nn.init.zeros_(self.cn_head.weight)
        nn.init.zeros_(self.cn_head.bias)
        nn.init.zeros_(self.ad_residual_head.weight)
        nn.init.zeros_(self.ad_residual_head.bias)

    def average_velocity(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B,{self.latent_dim}], got {tuple(latent.shape)}")
        source = source_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        target = target_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        label = label_ad.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        if source.shape[0] != latent.shape[0] or target.shape[0] != latent.shape[0] or label.shape[0] != latent.shape[0]:
            raise ValueError("Latent, time, and diagnosis batch sizes must match")
        delta = target - source
        midpoint = 0.5 * (source + target)
        features = torch.cat([latent, source, target, delta, torch.abs(delta), midpoint], dim=1)
        hidden = self.activation(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.cn_head(hidden) + label * self.ad_residual_head(hidden)

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context, context_time
        source = source_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        target = target_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        return latent + (target - source) * self.average_velocity(latent, source, target, label_ad)


class LowFrequencyTimeEmbedding(nn.Module):
    """Fixed low-frequency age/interval encoding followed by a small MLP.

    Raw source/target ages remain available to the C4 backbone.  This encoder
    instead supplies a compact, explicit representation of absolute age
    (midpoint) and signed transport interval to the residual blocks.
    """

    def __init__(self, embedding_dim: int, frequencies: Iterable[float]) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("time embedding_dim must be positive")
        values = tuple(float(value) for value in frequencies)
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("time frequencies must be a non-empty list of positive finite values")
        self.register_buffer("frequencies", torch.tensor(values, dtype=torch.float32).view(1, -1))
        # Raw [s, t, t-s, |t-s|, midpoint], plus sin/cos of midpoint and signed gap.
        feature_dim = 5 + 4 * len(values)
        self.fc1 = nn.Linear(feature_dim, int(embedding_dim))
        self.fc2 = nn.Linear(int(embedding_dim), int(embedding_dim))
        self.norm = nn.LayerNorm(int(embedding_dim))
        self.activation = nn.SiLU()

    def forward(self, source_time: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
        source = source_time.reshape(-1, 1)
        target = target_time.reshape(-1, 1)
        if source.shape != target.shape:
            raise ValueError("source_time and target_time must have equal batch sizes")
        delta = target - source
        midpoint = 0.5 * (source + target)
        frequencies = self.frequencies.to(dtype=source.dtype, device=source.device)
        midpoint_phase = 2.0 * math.pi * midpoint * frequencies
        delta_phase = 2.0 * math.pi * delta * frequencies
        raw = torch.cat([source, target, delta, torch.abs(delta), midpoint], dim=1)
        features = torch.cat(
            [raw, torch.sin(midpoint_phase), torch.cos(midpoint_phase), torch.sin(delta_phase), torch.cos(delta_phase)],
            dim=1,
        )
        hidden = self.activation(self.fc1(features))
        return self.norm(self.fc2(hidden))


class TimeEmbeddedDirectDiagnosisResidualCocycleFlow(nn.Module):
    """C4 direct flow with zero-initialized temporal FiLM adapters.

    The C4 raw time input, width, residual depth, CN head, AD-residual head,
    and direct transport equation are unchanged.  The only added capacity is
    a low-frequency time encoder and one bounded FiLM adapter per residual
    block.  Zero adapter initialization makes the untrained model exactly the
    original C4 function for identical base parameters.
    """

    def __init__(
        self,
        latent_dim: int,
        width: int,
        residual_blocks: int,
        time_embedding_dim: int,
        time_frequencies: Iterable[float],
        modulation_max_scale: float,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or width <= 0 or residual_blocks <= 0:
            raise ValueError("latent_dim, width, and residual_blocks must be positive")
        if not math.isfinite(modulation_max_scale) or modulation_max_scale <= 0.0:
            raise ValueError("modulation_max_scale must be positive and finite")
        self.latent_dim = int(latent_dim)
        self.modulation_max_scale = float(modulation_max_scale)
        # Keep the original C4 modules and construction order exactly intact.
        # This permits non-strict C4 checkpoint loading and a direct function
        # parity check before the temporal adapters learn.
        self.input = nn.Linear(self.latent_dim + 5, int(width))
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList(ResidualBlock(int(width)) for _ in range(int(residual_blocks)))
        self.cn_head = nn.Linear(int(width), self.latent_dim)
        self.ad_residual_head = nn.Linear(int(width), self.latent_dim)
        nn.init.zeros_(self.cn_head.weight)
        nn.init.zeros_(self.cn_head.bias)
        nn.init.zeros_(self.ad_residual_head.weight)
        nn.init.zeros_(self.ad_residual_head.bias)

        self.time_embedding = LowFrequencyTimeEmbedding(int(time_embedding_dim), time_frequencies)
        self.time_modulations = nn.ModuleList(
            nn.Linear(int(time_embedding_dim), 2 * int(width)) for _ in range(int(residual_blocks))
        )
        for adapter in self.time_modulations:
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def average_velocity(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B,{self.latent_dim}], got {tuple(latent.shape)}")
        source = source_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        target = target_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        label = label_ad.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        if source.shape[0] != latent.shape[0] or target.shape[0] != latent.shape[0] or label.shape[0] != latent.shape[0]:
            raise ValueError("Latent, time, and diagnosis batch sizes must match")
        delta = target - source
        midpoint = 0.5 * (source + target)
        features = torch.cat([latent, source, target, delta, torch.abs(delta), midpoint], dim=1)
        hidden = self.activation(self.input(features))
        embedding = self.time_embedding(source, target)
        for block, adapter in zip(self.blocks, self.time_modulations):
            residual = hidden
            normalized = block.norm(hidden)
            gamma, beta = adapter(embedding).chunk(2, dim=1)
            scale = self.modulation_max_scale
            normalized = (1.0 + scale * torch.tanh(gamma)) * normalized + scale * torch.tanh(beta)
            hidden = block.activation(block.fc2(block.activation(block.fc1(normalized))) + residual)
        return self.cn_head(hidden) + label * self.ad_residual_head(hidden)

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context, context_time
        source = source_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        target = target_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        return latent + (target - source) * self.average_velocity(latent, source, target, label_ad)


class DiagnosisResidualPotential(nn.Module):
    """Cumulative PCA progression potential conditioned on fixed baseline context."""

    def __init__(self, latent_dim: int, width: int, residual_blocks: int) -> None:
        super().__init__()
        if latent_dim <= 0 or width <= 0 or residual_blocks <= 0:
            raise ValueError("latent_dim, width, and residual_blocks must be positive")
        self.latent_dim = int(latent_dim)
        # Baseline PCA context, baseline age, query age, and elapsed normalized age.
        self.input = nn.Linear(self.latent_dim + 3, int(width))
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList(ResidualBlock(int(width)) for _ in range(int(residual_blocks)))
        self.cn_head = nn.Linear(int(width), self.latent_dim)
        self.ad_residual_head = nn.Linear(int(width), self.latent_dim)
        nn.init.zeros_(self.cn_head.weight)
        nn.init.zeros_(self.cn_head.bias)
        nn.init.zeros_(self.ad_residual_head.weight)
        nn.init.zeros_(self.ad_residual_head.bias)

    def forward(
        self,
        context: torch.Tensor,
        context_time: torch.Tensor,
        query_time: torch.Tensor,
        label_ad: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.latent_dim:
            raise ValueError(f"Expected context [B,{self.latent_dim}], got {tuple(context.shape)}")
        baseline = context_time.reshape(-1, 1).to(dtype=context.dtype, device=context.device)
        query = query_time.reshape(-1, 1).to(dtype=context.dtype, device=context.device)
        label = label_ad.reshape(-1, 1).to(dtype=context.dtype, device=context.device)
        if baseline.shape[0] != context.shape[0] or query.shape[0] != context.shape[0] or label.shape[0] != context.shape[0]:
            raise ValueError("Context, time, and diagnosis batch sizes must match")
        features = torch.cat([context, baseline, query, query - baseline], dim=1)
        hidden = self.activation(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.cn_head(hidden) + label * self.ad_residual_head(hidden)


class VolumeAxisDiagnosisResidualPotential(nn.Module):
    """Exact potential with separate volume-axis and volume-orthogonal parts.

    ``volume_coefficient`` is the train-only ridge coefficient ``w`` from
    ``log(volume) ~= intercept + w^T z`` in standardized PCA coordinates.
    The normalized vector ``volume_basis = w / (w^T w)`` has unit response
    under this local linear volume model: ``w^T volume_basis = 1``.  Therefore
    the scalar potential ``q`` is directly calibrated in approximate
    log-volume units, while the learned vector potential is projected to the
    orthogonal complement of ``w``.
    """

    def __init__(
        self,
        latent_dim: int,
        width: int,
        residual_blocks: int,
        volume_coefficient: list[float] | np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or width <= 0 or residual_blocks <= 0:
            raise ValueError("latent_dim, width, and residual_blocks must be positive")
        coefficient = torch.as_tensor(volume_coefficient, dtype=torch.float32).reshape(-1)
        if coefficient.numel() != int(latent_dim):
            raise ValueError(
                f"volume coefficient must contain {latent_dim} PCA entries, got {coefficient.numel()}"
            )
        squared_norm = torch.dot(coefficient, coefficient)
        if not bool(torch.isfinite(coefficient).all()) or not bool(torch.isfinite(squared_norm)) or float(squared_norm) <= 1.0e-12:
            raise ValueError("volume coefficient must be finite and have non-zero norm")
        self.latent_dim = int(latent_dim)
        self.register_buffer("volume_coefficient", coefficient)
        self.register_buffer("volume_basis", coefficient / squared_norm)
        # Baseline PCA context, baseline age, query age, and elapsed normalized age.
        self.input = nn.Linear(self.latent_dim + 3, int(width))
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList(ResidualBlock(int(width)) for _ in range(int(residual_blocks)))
        self.cn_shape_head = nn.Linear(int(width), self.latent_dim)
        self.ad_shape_residual_head = nn.Linear(int(width), self.latent_dim)
        self.cn_volume_head = nn.Linear(int(width), 1)
        self.ad_volume_residual_head = nn.Linear(int(width), 1)
        for head in (
            self.cn_shape_head,
            self.ad_shape_residual_head,
            self.cn_volume_head,
            self.ad_volume_residual_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def components(
        self,
        context: torch.Tensor,
        context_time: torch.Tensor,
        query_time: torch.Tensor,
        label_ad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``w``-orthogonal potential and scalar log-volume potential."""
        if context.ndim != 2 or context.shape[1] != self.latent_dim:
            raise ValueError(f"Expected context [B,{self.latent_dim}], got {tuple(context.shape)}")
        baseline = context_time.reshape(-1, 1).to(dtype=context.dtype, device=context.device)
        query = query_time.reshape(-1, 1).to(dtype=context.dtype, device=context.device)
        label = label_ad.reshape(-1, 1).to(dtype=context.dtype, device=context.device)
        if baseline.shape[0] != context.shape[0] or query.shape[0] != context.shape[0] or label.shape[0] != context.shape[0]:
            raise ValueError("Context, time, and diagnosis batch sizes must match")
        features = torch.cat([context, baseline, query, query - baseline], dim=1)
        hidden = self.activation(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        raw_shape = self.cn_shape_head(hidden) + label * self.ad_shape_residual_head(hidden)
        volume_potential = (self.cn_volume_head(hidden) + label * self.ad_volume_residual_head(hidden)).reshape(-1)
        coefficient = self.volume_coefficient.to(dtype=context.dtype, device=context.device).view(1, -1)
        basis = self.volume_basis.to(dtype=context.dtype, device=context.device).view(1, -1)
        shape_projection = torch.sum(raw_shape * coefficient, dim=1, keepdim=True)
        shape_orthogonal = raw_shape - shape_projection * basis
        return shape_orthogonal, volume_potential

    def forward(
        self,
        context: torch.Tensor,
        context_time: torch.Tensor,
        query_time: torch.Tensor,
        label_ad: torch.Tensor,
    ) -> torch.Tensor:
        shape_orthogonal, volume_potential = self.components(context, context_time, query_time, label_ad)
        basis = self.volume_basis.to(dtype=context.dtype, device=context.device).view(1, -1)
        return shape_orthogonal + volume_potential.view(-1, 1) * basis


class ExactAdditiveCoboundaryFlow(nn.Module):
    """Exact additive coboundary using a fixed first-visit subject context."""

    def __init__(self, latent_dim: int, width: int, residual_blocks: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.potential = DiagnosisResidualPotential(latent_dim, width, residual_blocks)

    def coboundary_increment(
        self,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor,
        context_time: torch.Tensor,
    ) -> torch.Tensor:
        potential_source = self.potential(context, context_time, source_time, label_ad)
        potential_target = self.potential(context, context_time, target_time, label_ad)
        return potential_target - potential_source

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None or context_time is None:
            raise ValueError("Exact coboundary transport requires fixed subject context and context_time")
        return latent + self.coboundary_increment(source_time, target_time, label_ad, context, context_time)


class ExactVolumeAxisCoboundaryFlow(nn.Module):
    """Exact coboundary with a scalar potential along the decoded-volume axis."""

    def __init__(
        self,
        latent_dim: int,
        width: int,
        residual_blocks: int,
        volume_coefficient: list[float] | np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.potential = VolumeAxisDiagnosisResidualPotential(
            latent_dim, width, residual_blocks, volume_coefficient
        )

    def coboundary_increment(
        self,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor,
        context_time: torch.Tensor,
    ) -> torch.Tensor:
        potential_source = self.potential(context, context_time, source_time, label_ad)
        potential_target = self.potential(context, context_time, target_time, label_ad)
        return potential_target - potential_source

    def volume_potential_delta(
        self,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor,
        context_time: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar C7 potential difference, calibrated to local log-volume change."""
        _, source_volume_potential = self.potential.components(context, context_time, source_time, label_ad)
        _, target_volume_potential = self.potential.components(context, context_time, target_time, label_ad)
        return target_volume_potential - source_volume_potential

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None or context_time is None:
            raise ValueError("Exact volume-axis coboundary transport requires fixed subject context and context_time")
        return latent + self.coboundary_increment(source_time, target_time, label_ad, context, context_time)


class SoftStateDependentCoboundaryFlow(nn.Module):
    """State-dependent V5 flow with an auxiliary additive potential."""

    def __init__(self, latent_dim: int, width: int, residual_blocks: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.direct = DirectDiagnosisResidualCocycleFlow(latent_dim, width, residual_blocks)
        self.potential = DiagnosisResidualPotential(latent_dim, width, residual_blocks)

    def coboundary_increment(
        self,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor,
        context_time: torch.Tensor,
    ) -> torch.Tensor:
        potential_source = self.potential(context, context_time, source_time, label_ad)
        potential_target = self.potential(context, context_time, target_time, label_ad)
        return potential_target - potential_source

    def transport(
        self,
        latent: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        label_ad: torch.Tensor,
        context: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.direct.transport(latent, source_time, target_time, label_ad, context, context_time)


def build_flow(model_config: dict[str, Any], volume_axis: dict[str, Any] | None = None) -> nn.Module:
    """Build a direct flow, its time-conditioned C4 variant, or a coboundary variant."""
    variant = str(model_config.get("variant", "direct"))
    arguments = {
        "latent_dim": int(model_config["latent_dim"]),
        "width": int(model_config["width"]),
        "residual_blocks": int(model_config["residual_blocks"]),
    }
    if variant == "direct":
        return DirectDiagnosisResidualCocycleFlow(**arguments)
    if variant == "direct_time_film":
        return TimeEmbeddedDirectDiagnosisResidualCocycleFlow(
            **arguments,
            time_embedding_dim=int(model_config["time_embedding_dim"]),
            time_frequencies=model_config["time_frequencies"],
            modulation_max_scale=float(model_config["modulation_max_scale"]),
        )
    if variant == "coboundary_exact":
        return ExactAdditiveCoboundaryFlow(**arguments)
    if variant == "coboundary_soft":
        return SoftStateDependentCoboundaryFlow(**arguments)
    if variant == "coboundary_volume_axis":
        if not isinstance(volume_axis, dict):
            raise ValueError("coboundary_volume_axis requires train-only volume_axis statistics")
        coefficient = volume_axis.get("linear_log_volume_coefficient")
        if coefficient is None:
            raise KeyError("volume_axis.linear_log_volume_coefficient is required")
        return ExactVolumeAxisCoboundaryFlow(**arguments, volume_coefficient=coefficient)
    raise ValueError(f"Unknown model variant: {variant}")


class PcaGeometry(nn.Module):
    """Fixed train-only PCA decoder and differentiable closed-mesh volume."""

    def __init__(self, pca_model: dict[str, np.ndarray], score_mean: np.ndarray, score_std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean_flat", torch.from_numpy(pca_model["mean"].astype(np.float32)).view(1, -1))
        self.register_buffer("components", torch.from_numpy(pca_model["components"].astype(np.float32)))
        self.register_buffer("faces", torch.from_numpy(pca_model["faces"].astype(np.int64)))
        self.register_buffer("score_mean", torch.from_numpy(score_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("score_std", torch.from_numpy(np.maximum(score_std, 1.0e-6).astype(np.float32)).view(1, -1))

    def vertices(self, standardized_scores: torch.Tensor) -> torch.Tensor:
        raw_scores = standardized_scores * self.score_std + self.score_mean
        flat = raw_scores @ self.components + self.mean_flat
        return flat.reshape(standardized_scores.shape[0], -1, 3)

    def volume_from_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        faces = self.faces.to(device=vertices.device)
        v0 = vertices[:, faces[:, 0], :]
        v1 = vertices[:, faces[:, 1], :]
        v2 = vertices[:, faces[:, 2], :]
        signed = torch.sum(v0 * torch.cross(v1, v2, dim=2), dim=2).sum(dim=1) / 6.0
        return torch.clamp(torch.abs(signed), min=1.0e-8)

    def volume(self, standardized_scores: torch.Tensor) -> torch.Tensor:
        return self.volume_from_vertices(self.vertices(standardized_scores))


@dataclass(frozen=True)
class CocyclePair:
    source: int
    target: int
    intermediate: int
    subject: str
    diagnosis: str
    pair_type: str


class CocyclePairDataset(Dataset[CocyclePair]):
    def __init__(self, rows: list[CocyclePair]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CocyclePair:
        return self.rows[index]


def collate_pairs(rows: list[CocyclePair]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.tensor([row.source for row in rows], dtype=torch.long),
        "target": torch.tensor([row.target for row in rows], dtype=torch.long),
        "intermediate": torch.tensor([row.intermediate for row in rows], dtype=torch.long),
    }


def convert_pairs(rows: Iterable[PairRow], archive: dict[str, np.ndarray]) -> list[CocyclePair]:
    subjects = archive["visit_subject_ids"].astype(str)
    diagnoses = archive["visit_diagnoses"].astype(str)
    converted: list[CocyclePair] = []
    for row in rows:
        subject = str(subjects[row.source_index])
        diagnosis = str(diagnoses[row.source_index])
        if str(subjects[row.target_index]) != subject or str(diagnoses[row.target_index]) != diagnosis:
            raise ValueError("A training pair crosses subject or diagnosis boundaries")
        converted.append(CocyclePair(
            source=int(row.source_index),
            target=int(row.target_index),
            intermediate=int(row.intermediate_index),
            subject=subject,
            diagnosis=diagnosis,
            pair_type=str(row.pair_type),
        ))
    if not converted:
        raise ValueError("No pairs available")
    return converted


def values_on_device(archive: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    z = torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device)
    age = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
    context = torch.empty_like(z)
    context_age = torch.empty_like(age)
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    for subject_index in range(len(offsets) - 1):
        first = int(offsets[subject_index])
        last = int(offsets[subject_index + 1])
        context[first:last] = z[first].unsqueeze(0).expand(last - first, -1)
        context_age[first:last] = age[first]
    return {
        "z": z,
        "age": age,
        "years": torch.from_numpy(archive["visit_time_years_from_baseline"].astype(np.float32)).to(device),
        "label": torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device),
        # Fixed for every visit of a subject; required by exact coboundary algebra.
        "context": context,
        "context_age": context_age,
    }


def indexed(values: dict[str, torch.Tensor], raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source_index = raw["source"].to(values["z"].device)
    target_index = raw["target"].to(values["z"].device)
    intermediate_index = raw["intermediate"].to(values["z"].device)
    return {
        "source": values["z"][source_index],
        "target": values["z"][target_index],
        "source_age": values["age"][source_index],
        "target_age": values["age"][target_index],
        "source_years": values["years"][source_index],
        "target_years": values["years"][target_index],
        "label": values["label"][source_index],
        "context": values["context"][source_index],
        "context_age": values["context_age"][source_index],
        "intermediate_index": intermediate_index,
    }


def safe_median(values: list[np.ndarray], floor: float = 1.0e-6) -> float:
    if not values:
        return float(floor)
    merged = np.concatenate(values)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return float(floor)
    return float(max(np.median(merged), floor))


def line_slope(times: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    centered_time = times - times.mean()
    denominator = torch.sum(centered_time.square()).clamp_min(1.0e-8)
    return torch.sum(centered_time * (values - values.mean())) / denominator


@torch.no_grad()
def estimate_volume_axis(
    geometry: PcaGeometry,
    standardized_scores: torch.Tensor,
    batch_size: int = 512,
    ridge: float = 1.0e-4,
) -> dict[str, Any]:
    """Fit the train-only local linear log-volume axis in standardized PCA space."""
    if standardized_scores.ndim != 2 or standardized_scores.shape[0] < 2:
        raise ValueError("At least two train scans are required to estimate the volume axis")
    if ridge < 0.0:
        raise ValueError("volume-axis ridge must be non-negative")
    latent_dim = int(standardized_scores.shape[1])
    count = 0
    sum_z = np.zeros(latent_dim, dtype=np.float64)
    sum_y = 0.0
    sum_zz = np.zeros((latent_dim, latent_dim), dtype=np.float64)
    sum_zy = np.zeros(latent_dim, dtype=np.float64)
    sum_yy = 0.0
    for start in range(0, int(standardized_scores.shape[0]), int(batch_size)):
        z = standardized_scores[start : start + int(batch_size)]
        y = torch.log(geometry.volume(z))
        z_numpy = z.detach().cpu().numpy().astype(np.float64, copy=False)
        y_numpy = y.detach().cpu().numpy().astype(np.float64, copy=False)
        count += int(z_numpy.shape[0])
        sum_z += np.sum(z_numpy, axis=0)
        sum_y += float(np.sum(y_numpy))
        sum_zz += z_numpy.T @ z_numpy
        sum_zy += z_numpy.T @ y_numpy
        sum_yy += float(y_numpy @ y_numpy)
    if count < 2:
        raise ValueError("At least two train scans are required to estimate the volume axis")
    mean_z = sum_z / count
    mean_y = sum_y / count
    covariance = sum_zz / count - np.outer(mean_z, mean_z)
    cross_covariance = sum_zy / count - mean_z * mean_y
    coefficient = np.linalg.solve(covariance + float(ridge) * np.eye(latent_dim), cross_covariance)
    coefficient_norm = float(np.linalg.norm(coefficient))
    if not np.all(np.isfinite(coefficient)) or not math.isfinite(coefficient_norm) or coefficient_norm <= 1.0e-8:
        raise RuntimeError("Could not estimate a finite non-zero train-only volume axis")
    intercept = float(mean_y - mean_z @ coefficient)
    residual_sum_squares = (
        sum_yy
        - 2.0 * intercept * sum_y
        - 2.0 * coefficient @ sum_zy
        + count * intercept * intercept
        + 2.0 * intercept * coefficient @ sum_z
        + coefficient @ sum_zz @ coefficient
    )
    total_sum_squares = float(sum_yy - count * mean_y * mean_y)
    r_squared = 1.0 - residual_sum_squares / max(total_sum_squares, 1.0e-12)
    return {
        "definition": "train-only ridge fit: log(decoded_volume) = intercept + w^T standardized_pca",
        "linear_log_volume_coefficient": coefficient.tolist(),
        "linear_log_volume_intercept": intercept,
        "coefficient_l2_norm": coefficient_norm,
        "ridge": float(ridge),
        "train_r_squared": float(r_squared),
        "scans": int(count),
    }


@torch.no_grad()
def training_statistics(
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    rows: list[CocyclePair],
    archive: dict[str, np.ndarray],
    batch_size: int = 512,
    include_volume_axis: bool = False,
) -> dict[str, Any]:
    collected: dict[str, list[np.ndarray]] = {
        "pca": [], "coordinate": [], "euclidean": [], "volume_log": [], "rate": [], "displacement": [],
    }
    rates_by_subject: dict[str, dict[str, list[float]]] = {"CN": {}, "AD": {}}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        raw = collate_pairs(chunk)
        batch = indexed(values, raw)
        source_vertices = geometry.vertices(batch["source"])
        target_vertices = geometry.vertices(batch["target"])
        source_volume = geometry.volume_from_vertices(source_vertices)
        target_volume = geometry.volume_from_vertices(target_vertices)
        gap = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        log_delta = torch.log(target_volume) - torch.log(source_volume)
        collected["pca"].append(torch.mean((batch["target"] - batch["source"]).square(), dim=1).cpu().numpy())
        collected["coordinate"].append(torch.mean(torch.abs(target_vertices - source_vertices), dim=(1, 2)).cpu().numpy())
        collected["euclidean"].append(torch.linalg.vector_norm(target_vertices - source_vertices, dim=2).mean(dim=1).cpu().numpy())
        collected["volume_log"].append(torch.abs(log_delta).cpu().numpy())
        collected["rate"].append(torch.abs(log_delta / gap).cpu().numpy())
        collected["displacement"].append(torch.sqrt(torch.mean((batch["target"] - batch["source"]).square(), dim=1)).cpu().numpy())
        for offset, row in enumerate(chunk):
            rate = float((log_delta[offset] / gap[offset]).cpu())
            rates_by_subject[row.diagnosis].setdefault(row.subject, []).append(rate)

    offsets = archive["subject_visit_offsets"].astype(np.int64)
    slope_values: list[np.ndarray] = []
    for subject_index in range(len(archive["subject_ids"])):
        first, last = int(offsets[subject_index]), int(offsets[subject_index + 1])
        slope_values.append(np.asarray([
            float(line_slope(values["years"][first:last], torch.log(geometry.volume(values["z"][first:last]))).cpu())
        ], dtype=np.float64))
    group_targets: dict[str, float] = {}
    for diagnosis in ("CN", "AD"):
        subject_means = [float(np.mean(item)) for item in rates_by_subject[diagnosis].values()]
        if not subject_means:
            raise ValueError(f"No train rate targets for {diagnosis}")
        group_targets[diagnosis] = float(np.mean(subject_means))
    output = {
        "normalization_scales": {
            "pca": safe_median(collected["pca"]),
            "coordinate": safe_median(collected["coordinate"]),
            "euclidean": safe_median(collected["euclidean"]),
            "volume_log": safe_median(collected["volume_log"]),
            "rate": safe_median(collected["rate"]),
            "slope": safe_median([np.abs(value) for value in slope_values]),
            "displacement": safe_median(collected["displacement"]),
        },
        "group_log_volume_rate_targets": group_targets,
        "ad_minus_cn_log_volume_rate_target": group_targets["AD"] - group_targets["CN"],
        "subjects_by_diagnosis": {diagnosis: len(entries) for diagnosis, entries in rates_by_subject.items()},
        "statistics_source": "train split only; subject-balanced observed forward pairs",
    }
    if include_volume_axis:
        output["volume_axis"] = estimate_volume_axis(geometry, values["z"], batch_size=batch_size)
    return output


def balanced_pair_sampler(rows: list[CocyclePair], seed: int, epoch: int, samples: int) -> WeightedRandomSampler:
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        gap = "adjacent" if row.pair_type == "adjacent" else "nonadjacent"
        buckets.setdefault((row.diagnosis, gap, row.subject), []).append(index)
    strata: dict[tuple[str, str], list[str]] = {}
    for diagnosis, gap, subject in buckets:
        strata.setdefault((diagnosis, gap), []).append(subject)
    if set(strata) != {("CN", "adjacent"), ("CN", "nonadjacent"), ("AD", "adjacent"), ("AD", "nonadjacent")}:
        raise ValueError(f"Each diagnosis/gap stratum is required; found {sorted(strata)}")
    weights = np.zeros(len(rows), dtype=np.float64)
    for (diagnosis, gap), subjects in strata.items():
        stratum_weight = 1.0 / len(strata)
        for subject in subjects:
            indices = buckets[(diagnosis, gap, subject)]
            row_weight = stratum_weight / len(subjects) / len(indices)
            weights[np.asarray(indices, dtype=np.int64)] = row_weight
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 100_003 * int(epoch))
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=int(samples), replacement=True, generator=generator)


def balanced_sequence_starts(archive: dict[str, np.ndarray], seed: int, epoch: int) -> list[int]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    diagnoses = archive["subject_diagnoses"].astype(str)
    grouped = {diagnosis: [int(offsets[index]) for index, item in enumerate(diagnoses) if item == diagnosis] for diagnosis in ("CN", "AD")}
    if not grouped["CN"] or not grouped["AD"]:
        raise ValueError("Both CN and AD sequence subjects are required")
    generator = random.Random(int(seed) + 97_003 * int(epoch))
    for items in grouped.values():
        generator.shuffle(items)
    output: list[int] = []
    for index in range(max(len(grouped["CN"]), len(grouped["AD"]))):
        output.append(grouped["CN"][index % len(grouped["CN"])])
        output.append(grouped["AD"][index % len(grouped["AD"])])
    return output


def shape_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    geometry: PcaGeometry,
    scales: dict[str, float],
) -> dict[str, torch.Tensor]:
    pca = torch.mean((prediction - target).square()) / float(scales["pca"])
    predicted_vertices = geometry.vertices(prediction)
    target_vertices = geometry.vertices(target)
    coordinate = torch.mean(torch.abs(predicted_vertices - target_vertices)) / float(scales["coordinate"])
    euclidean = torch.linalg.vector_norm(predicted_vertices - target_vertices, dim=2).mean() / float(scales["euclidean"])
    return {"pca": pca, "vertex": 0.5 * (coordinate + euclidean), "coordinate": coordinate, "euclidean": euclidean}


def mean_scaled_mse(left: torch.Tensor, right: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.mean((left - right).square()) / float(scale)


def pair_terms(
    flow: nn.Module,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    statistics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    batch = indexed(values, raw)
    scales = statistics["normalization_scales"]
    prediction_forward = flow.transport(
        batch["source"], batch["source_age"], batch["target_age"], batch["label"], batch["context"], batch["context_age"]
    )
    prediction_backward = flow.transport(
        batch["target"], batch["target_age"], batch["source_age"], batch["label"], batch["context"], batch["context_age"]
    )
    forward = shape_terms(prediction_forward, batch["target"], geometry, scales)
    backward = shape_terms(prediction_backward, batch["source"], geometry, scales)

    valid_middle = batch["intermediate_index"] >= 0
    observed = prediction_forward.sum() * 0.0
    if bool(valid_middle.any()):
        mid_index = batch["intermediate_index"][valid_middle]
        middle_age = values["age"][mid_index]
        labels = batch["label"][valid_middle]
        middle_context = batch["context"][valid_middle]
        middle_context_age = batch["context_age"][valid_middle]
        forward_middle = flow.transport(
            batch["source"][valid_middle], batch["source_age"][valid_middle], middle_age, labels,
            middle_context, middle_context_age,
        )
        forward_composed = flow.transport(
            forward_middle, middle_age, batch["target_age"][valid_middle], labels,
            middle_context, middle_context_age,
        )
        backward_middle = flow.transport(
            batch["target"][valid_middle], batch["target_age"][valid_middle], middle_age, labels,
            middle_context, middle_context_age,
        )
        backward_composed = flow.transport(
            backward_middle, middle_age, batch["source_age"][valid_middle], labels,
            middle_context, middle_context_age,
        )
        observed = 0.5 * (
            mean_scaled_mse(prediction_forward[valid_middle], forward_composed, scales["pca"])
            + mean_scaled_mse(prediction_backward[valid_middle], backward_composed, scales["pca"])
        )

    ratio = torch.empty_like(batch["source_age"]).uniform_(0.2, 0.8)
    virtual_age = batch["source_age"] + ratio * (batch["target_age"] - batch["source_age"])
    forward_middle = flow.transport(
        batch["source"], batch["source_age"], virtual_age, batch["label"], batch["context"], batch["context_age"]
    )
    forward_composed = flow.transport(
        forward_middle, virtual_age, batch["target_age"], batch["label"], batch["context"], batch["context_age"]
    )
    backward_middle = flow.transport(
        batch["target"], batch["target_age"], virtual_age, batch["label"], batch["context"], batch["context_age"]
    )
    backward_composed = flow.transport(
        backward_middle, virtual_age, batch["source_age"], batch["label"], batch["context"], batch["context_age"]
    )
    virtual = 0.5 * (
        mean_scaled_mse(prediction_forward, forward_composed, scales["pca"])
        + mean_scaled_mse(prediction_backward, backward_composed, scales["pca"])
    )

    inverse_forward = flow.transport(
        prediction_forward, batch["target_age"], batch["source_age"], batch["label"], batch["context"], batch["context_age"]
    )
    inverse_backward = flow.transport(
        prediction_backward, batch["source_age"], batch["target_age"], batch["label"], batch["context"], batch["context_age"]
    )
    inverse = 0.5 * (
        mean_scaled_mse(inverse_forward, batch["source"], scales["pca"])
        + mean_scaled_mse(inverse_backward, batch["target"], scales["pca"])
    )

    coboundary = prediction_forward.sum() * 0.0
    if hasattr(flow, "coboundary_increment"):
        potential_forward = flow.coboundary_increment(
            batch["source_age"], batch["target_age"], batch["label"], batch["context"], batch["context_age"]
        )
        potential_backward = flow.coboundary_increment(
            batch["target_age"], batch["source_age"], batch["label"], batch["context"], batch["context_age"]
        )
        coboundary = 0.5 * (
            mean_scaled_mse(prediction_forward - batch["source"], potential_forward, scales["pca"])
            + mean_scaled_mse(prediction_backward - batch["target"], potential_backward, scales["pca"])
        )

    source_volume = geometry.volume(batch["source"])
    target_volume = geometry.volume(batch["target"])
    predicted_forward_volume = geometry.volume(prediction_forward)
    predicted_backward_volume = geometry.volume(prediction_backward)
    log_forward = torch.log(predicted_forward_volume) - torch.log(target_volume)
    log_backward = torch.log(predicted_backward_volume) - torch.log(source_volume)
    volume = 0.5 * (
        F.smooth_l1_loss(log_forward / float(scales["volume_log"]), torch.zeros_like(log_forward))
        + F.smooth_l1_loss(log_backward / float(scales["volume_log"]), torch.zeros_like(log_backward))
    )
    volume_potential = prediction_forward.sum() * 0.0
    if hasattr(flow, "volume_potential_delta"):
        observed_log_delta = torch.log(target_volume) - torch.log(source_volume)
        potential_forward = flow.volume_potential_delta(
            batch["source_age"], batch["target_age"], batch["label"], batch["context"], batch["context_age"]
        )
        potential_backward = flow.volume_potential_delta(
            batch["target_age"], batch["source_age"], batch["label"], batch["context"], batch["context_age"]
        )
        volume_potential = 0.5 * (
            F.smooth_l1_loss(
                (potential_forward - observed_log_delta) / float(scales["volume_log"]),
                torch.zeros_like(potential_forward),
            )
            + F.smooth_l1_loss(
                (potential_backward + observed_log_delta) / float(scales["volume_log"]),
                torch.zeros_like(potential_backward),
            )
        )
    years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
    rate_forward = (torch.log(predicted_forward_volume) - torch.log(source_volume)) / years
    observed_rate = (torch.log(target_volume) - torch.log(source_volume)) / years
    rate_backward = (torch.log(predicted_backward_volume) - torch.log(target_volume)) / (-years)
    rate = 0.5 * (
        F.smooth_l1_loss((rate_forward - observed_rate) / float(scales["rate"]), torch.zeros_like(rate_forward))
        + F.smooth_l1_loss((rate_backward - observed_rate) / float(scales["rate"]), torch.zeros_like(rate_backward))
    )

    labels = batch["label"] >= 0.5
    group_rate = prediction_forward.sum() * 0.0
    disease_gap = prediction_forward.sum() * 0.0
    if bool(labels.any()) and bool((~labels).any()):
        group_losses: list[torch.Tensor] = []
        for diagnosis, mask in (("CN", ~labels), ("AD", labels)):
            target = torch.tensor(float(statistics["group_log_volume_rate_targets"][diagnosis]), device=rate_forward.device)
            group_losses.append(F.smooth_l1_loss((rate_forward[mask].mean() - target) / float(scales["rate"]), torch.zeros((), device=rate_forward.device)))
        group_rate = torch.stack(group_losses).mean()
        target_gap = torch.tensor(float(statistics["ad_minus_cn_log_volume_rate_target"]), device=rate_forward.device)
        disease_gap = F.smooth_l1_loss(
            ((rate_forward[labels].mean() - rate_forward[~labels].mean()) - target_gap) / float(scales["rate"]),
            torch.zeros((), device=rate_forward.device),
        )
    return {
        "real_pca": 0.5 * (forward["pca"] + backward["pca"]),
        "real_vertex": 0.5 * (forward["vertex"] + backward["vertex"]),
        "real_coordinate": 0.5 * (forward["coordinate"] + backward["coordinate"]),
        "real_euclidean": 0.5 * (forward["euclidean"] + backward["euclidean"]),
        "observed_semigroup": observed,
        "virtual_semigroup": virtual,
        "inverse": inverse,
        "coboundary": coboundary,
        "volume": volume,
        "volume_potential": volume_potential,
        "rate": rate,
        "group_rate": group_rate,
        "disease_gap": disease_gap,
    }


def sequence_terms(
    flow: nn.Module,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    start: int,
    statistics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subject_index = int(np.searchsorted(offsets, int(start), side="right") - 1)
    end = int(offsets[subject_index + 1])
    z = values["z"][start:end]
    ages = values["age"][start:end]
    years = values["years"][start:end]
    label = values["label"][start : start + 1]
    context = values["context"][start : start + 1]
    context_age = values["context_age"][start : start + 1]
    if z.shape[0] < 2:
        raise ValueError("Sequence loss requires at least two visits")
    scales = statistics["normalization_scales"]

    source = z[:1]
    source_age = ages[:1]
    targets = z[1:]
    target_ages = ages[1:]
    count = targets.shape[0]
    direct_forward = flow.transport(
        source.expand(count, -1), source_age.expand(count), target_ages, label.expand(count),
        context.expand(count, -1), context_age.expand(count),
    )
    rollout_forward: list[torch.Tensor] = []
    current = source
    previous_age = source_age
    for index in range(1, z.shape[0]):
        current = flow.transport(current, previous_age, ages[index : index + 1], label, context, context_age)
        rollout_forward.append(current)
        previous_age = ages[index : index + 1]
    rollout_forward_tensor = torch.cat(rollout_forward, dim=0)

    reverse_targets = z[:-1].flip(0)
    reverse_ages = ages[:-1].flip(0)
    reverse_source = z[-1:]
    reverse_source_age = ages[-1:]
    direct_backward = flow.transport(
        reverse_source.expand(count, -1), reverse_source_age.expand(count), reverse_ages, label.expand(count),
        context.expand(count, -1), context_age.expand(count),
    )
    rollout_backward: list[torch.Tensor] = []
    current = reverse_source
    previous_age = reverse_source_age
    for index in range(z.shape[0] - 2, -1, -1):
        current = flow.transport(current, previous_age, ages[index : index + 1], label, context, context_age)
        rollout_backward.append(current)
        previous_age = ages[index : index + 1]
    rollout_backward_tensor = torch.cat(rollout_backward, dim=0)

    shape_values = [
        shape_terms(direct_forward, targets, geometry, scales),
        shape_terms(rollout_forward_tensor, targets, geometry, scales),
        shape_terms(direct_backward, reverse_targets, geometry, scales),
        shape_terms(rollout_backward_tensor, reverse_targets, geometry, scales),
    ]
    sequence_pca = torch.stack([item["pca"] for item in shape_values]).mean()
    sequence_vertex = torch.stack([item["vertex"] for item in shape_values]).mean()
    semigroup = 0.5 * (
        mean_scaled_mse(direct_forward, rollout_forward_tensor, scales["pca"])
        + mean_scaled_mse(direct_backward, rollout_backward_tensor, scales["pca"])
    )

    forward_volume = geometry.volume(torch.cat([source, direct_forward], dim=0))
    backward_volume = geometry.volume(torch.cat([reverse_source, direct_backward], dim=0))
    observed_volume = geometry.volume(z)
    slope_forward = line_slope(years, torch.log(forward_volume))
    slope_observed = line_slope(years, torch.log(observed_volume))
    slope_backward = line_slope(years.flip(0), torch.log(backward_volume))
    slope = 0.5 * (
        F.smooth_l1_loss((slope_forward - slope_observed) / float(scales["slope"]), torch.zeros_like(slope_forward))
        + F.smooth_l1_loss((slope_backward - slope_observed) / float(scales["slope"]), torch.zeros_like(slope_backward))
    )
    return {"sequence_pca": sequence_pca, "sequence_vertex": sequence_vertex, "sequence_semigroup": semigroup, "slope": slope}


def ramp(epoch: int, epochs: int) -> float:
    return 1.0 if epochs <= 0 else min(1.0, float(epoch) / float(epochs))


def total_loss(
    pair: dict[str, torch.Tensor],
    sequence: dict[str, torch.Tensor],
    config: dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = config["loss"]
    consistency_factor = ramp(epoch, int(config["training"]["consistency_ramp_epochs"]))
    anatomy_factor = ramp(epoch, int(config["training"]["anatomy_ramp_epochs"]))
    total = (
        float(loss_config["real_pca_weight"]) * pair["real_pca"]
        + float(loss_config["real_vertex_weight"]) * pair["real_vertex"]
        + consistency_factor * (
            float(loss_config["observed_semigroup_weight"]) * pair["observed_semigroup"]
            + float(loss_config["virtual_semigroup_weight"]) * pair["virtual_semigroup"]
            + float(loss_config["inverse_weight"]) * pair["inverse"]
            + float(loss_config.get("coboundary_weight", 0.0)) * pair["coboundary"]
            + float(loss_config["sequence_pca_weight"]) * sequence["sequence_pca"]
            + float(loss_config["sequence_vertex_weight"]) * sequence["sequence_vertex"]
            + float(loss_config["sequence_semigroup_weight"]) * sequence["sequence_semigroup"]
        )
        + anatomy_factor * (
            float(loss_config["volume_weight"]) * pair["volume"]
            + float(loss_config.get("volume_potential_weight", 0.0)) * pair["volume_potential"]
            + float(loss_config["rate_weight"]) * pair["rate"]
            + float(loss_config["slope_weight"]) * sequence["slope"]
            + float(loss_config["group_rate_weight"]) * pair["group_rate"]
            + float(loss_config["disease_gap_weight"]) * pair["disease_gap"]
        )
    )
    terms = pair | sequence
    terms["total"] = total
    terms["consistency_factor"] = torch.tensor(consistency_factor, device=total.device)
    terms["anatomy_factor"] = torch.tensor(anatomy_factor, device=total.device)
    return total, {name: float(value.detach().cpu()) for name, value in terms.items()}


def aggregate(values: dict[str, list[float]]) -> dict[str, float]:
    return {f"{name}_mean": float(np.mean(items)) if items else float("nan") for name, items in values.items()} | {"rows": float(len(next(iter(values.values()))) if values else 0)}


@torch.no_grad()
def evaluate_pairs(
    flow: nn.Module,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    rows: list[CocyclePair],
    batch_size: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {diagnosis: {name: [] for name in (
        "pca", "coordinate", "euclidean", "volume_relative", "rate",
        "predicted_signed_rate", "observed_signed_rate",
        "nochange_pca", "nochange_coordinate", "nochange_euclidean", "nochange_volume_relative", "nochange_rate"
    )} for diagnosis in ("CN", "AD", "overall")}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = indexed(values, collate_pairs(chunk))
        prediction = flow.transport(
            batch["source"], batch["source_age"], batch["target_age"], batch["label"],
            batch["context"], batch["context_age"],
        )
        source_vertices = geometry.vertices(batch["source"])
        target_vertices = geometry.vertices(batch["target"])
        predicted_vertices = geometry.vertices(prediction)
        source_volume = geometry.volume_from_vertices(source_vertices)
        target_volume = geometry.volume_from_vertices(target_vertices)
        predicted_volume = geometry.volume_from_vertices(predicted_vertices)
        years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        predicted_signed_rate = (torch.log(predicted_volume) - torch.log(source_volume)) / years
        observed_signed_rate = (torch.log(target_volume) - torch.log(source_volume)) / years
        metric_tensors = {
            "pca": torch.mean((prediction - batch["target"]).square(), dim=1),
            "coordinate": torch.mean(torch.abs(predicted_vertices - target_vertices), dim=(1, 2)),
            "euclidean": torch.linalg.vector_norm(predicted_vertices - target_vertices, dim=2).mean(dim=1),
            "volume_relative": torch.abs(predicted_volume - target_volume) / target_volume,
            "rate": torch.abs((torch.log(predicted_volume) - torch.log(target_volume)) / years),
            "predicted_signed_rate": predicted_signed_rate,
            "observed_signed_rate": observed_signed_rate,
            "nochange_pca": torch.mean((batch["source"] - batch["target"]).square(), dim=1),
            "nochange_coordinate": torch.mean(torch.abs(source_vertices - target_vertices), dim=(1, 2)),
            "nochange_euclidean": torch.linalg.vector_norm(source_vertices - target_vertices, dim=2).mean(dim=1),
            "nochange_volume_relative": torch.abs(source_volume - target_volume) / target_volume,
            "nochange_rate": torch.abs((torch.log(source_volume) - torch.log(target_volume)) / years),
        }
        for index, row in enumerate(chunk):
            buckets = (row.diagnosis, "overall")
            for bucket in buckets:
                for name, tensor in metric_tensors.items():
                    grouped[bucket][name].append(float(tensor[index].cpu()))
    return {"groups": {name: aggregate(values) for name, values in grouped.items()}}


def first_last_pairs(archive: dict[str, np.ndarray]) -> list[CocyclePair]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    diagnoses = archive["subject_diagnoses"].astype(str)
    subjects = archive["subject_ids"].astype(str)
    return [CocyclePair(
        source=int(offsets[index]), target=int(offsets[index + 1] - 1), intermediate=-1,
        subject=str(subjects[index]), diagnosis=str(diagnoses[index]), pair_type="first_last",
    ) for index in range(len(subjects))]


@torch.no_grad()
def cocycle_defects(
    flow: nn.Module,
    values: dict[str, torch.Tensor],
    rows: list[CocyclePair],
    statistics: dict[str, Any],
    batch_size: int,
) -> dict[str, float]:
    semi_values: list[float] = []
    inverse_values: list[float] = []
    scale = float(statistics["normalization_scales"]["displacement"])
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = indexed(values, collate_pairs(chunk))
        direct = flow.transport(
            batch["source"], batch["source_age"], batch["target_age"], batch["label"],
            batch["context"], batch["context_age"],
        )
        middle_age = 0.5 * (batch["source_age"] + batch["target_age"])
        middle = flow.transport(
            batch["source"], batch["source_age"], middle_age, batch["label"],
            batch["context"], batch["context_age"],
        )
        composed = flow.transport(
            middle, middle_age, batch["target_age"], batch["label"],
            batch["context"], batch["context_age"],
        )
        inverse = flow.transport(
            direct, batch["target_age"], batch["source_age"], batch["label"],
            batch["context"], batch["context_age"],
        )
        semi_values.extend((torch.sqrt(torch.mean((direct - composed).square(), dim=1)) / scale).cpu().tolist())
        inverse_values.extend((torch.sqrt(torch.mean((inverse - batch["source"]).square(), dim=1)) / scale).cpu().tolist())
    return {
        "relative_semigroup_defect_mean": float(np.mean(semi_values)),
        "relative_semigroup_defect_p95": float(np.quantile(semi_values, 0.95)),
        "relative_inverse_defect_mean": float(np.mean(inverse_values)),
        "relative_inverse_defect_p95": float(np.quantile(inverse_values, 0.95)),
    }


def validation_score(
    all_pairs: dict[str, Any],
    first_last: dict[str, Any],
    defects: dict[str, float],
    selection: dict[str, Any],
) -> tuple[float, bool, dict[str, float]]:
    macro_first_last: list[float] = []
    macro_first_last_volume: list[float] = []
    macro_first_last_rate: list[float] = []
    macro_group_trend: list[float] = []
    signed_rates: dict[str, tuple[float, float]] = {}
    feasible = True
    pca_allowance = 1.0 + float(selection["pca_nochange_tolerance"])
    coordinate_allowance = 1.0 + float(selection["coordinate_nochange_tolerance"])
    for diagnosis in ("CN", "AD"):
        values = first_last["groups"][diagnosis]
        coordinate_ratio = values["coordinate_mean"] / max(values["nochange_coordinate_mean"], 1.0e-8)
        euclidean_ratio = values["euclidean_mean"] / max(values["nochange_euclidean_mean"], 1.0e-8)
        pca_ratio = values["pca_mean"] / max(values["nochange_pca_mean"], 1.0e-8)
        first_last_volume_ratio = values["volume_relative_mean"] / max(values["nochange_volume_relative_mean"], 1.0e-8)
        first_last_rate_ratio = values["rate_mean"] / max(values["nochange_rate_mean"], 1.0e-8)
        predicted_signed_rate = float(values["predicted_signed_rate_mean"])
        observed_signed_rate = float(values["observed_signed_rate_mean"])
        group_trend_ratio = abs(predicted_signed_rate - observed_signed_rate) / max(values["nochange_rate_mean"], 1.0e-8)
        macro_first_last.append(0.5 * (coordinate_ratio + euclidean_ratio))
        macro_first_last_volume.append(first_last_volume_ratio)
        macro_first_last_rate.append(first_last_rate_ratio)
        macro_group_trend.append(group_trend_ratio)
        signed_rates[diagnosis] = (predicted_signed_rate, observed_signed_rate)
        feasible = feasible and all(math.isfinite(item) for item in (
            coordinate_ratio, euclidean_ratio, pca_ratio, first_last_volume_ratio,
            first_last_rate_ratio, group_trend_ratio,
        ))
        feasible = feasible and pca_ratio <= pca_allowance and coordinate_ratio <= coordinate_allowance
    all_values = all_pairs["groups"]["overall"]
    all_coordinate_ratio = all_values["coordinate_mean"] / max(all_values["nochange_coordinate_mean"], 1.0e-8)
    all_euclidean_ratio = all_values["euclidean_mean"] / max(all_values["nochange_euclidean_mean"], 1.0e-8)
    all_shape = 0.5 * (all_coordinate_ratio + all_euclidean_ratio)
    volume_ratio = all_values["volume_relative_mean"] / max(all_values["nochange_volume_relative_mean"], 1.0e-8)
    macro_volume = float(np.mean(macro_first_last_volume))
    macro_rate = float(np.mean(macro_first_last_rate))
    macro_trend = float(np.mean(macro_group_trend))
    score = (
        float(np.mean(macro_first_last))
        + float(selection["all_pair_shape_weight"]) * all_shape
        + float(selection["volume_tiebreak_weight"]) * volume_ratio
        + float(selection.get("first_last_volume_weight", 0.0)) * macro_volume
        + float(selection.get("first_last_rate_weight", 0.0)) * macro_rate
        + float(selection.get("group_trend_weight", 0.0)) * macro_trend
    )
    feasible = feasible and math.isfinite(score)
    feasible = feasible and defects["relative_semigroup_defect_mean"] <= float(selection["max_relative_semigroup_defect"])
    feasible = feasible and defects["relative_inverse_defect_mean"] <= float(selection["max_relative_inverse_defect"])
    ratios = {
        "macro_first_last_shape": float(np.mean(macro_first_last)),
        "all_pair_shape": float(all_shape),
        "all_pair_volume": float(volume_ratio),
        "macro_first_last_volume": macro_volume,
        "macro_first_last_rate": macro_rate,
        "macro_group_trend": macro_trend,
        "first_last_cn_predicted_signed_rate": signed_rates["CN"][0],
        "first_last_cn_observed_signed_rate": signed_rates["CN"][1],
        "first_last_ad_predicted_signed_rate": signed_rates["AD"][0],
        "first_last_ad_observed_signed_rate": signed_rates["AD"][1],
        "score": score,
    }
    return score, bool(feasible), ratios


def output_directory(config_path: Path, config: dict[str, Any], run_name: str) -> Path:
    return config_path.parent.parent / "training" / run_name


def validate_config(config_path: Path, config: dict[str, Any], structure: str, experiment: str) -> None:
    accepted_methods = {
        "direct_pca_cocycle_v5",
        "direct_time_embedded_pca_cocycle_v1",
        "exact_pca_coboundary_v1",
        "soft_pca_coboundary_v1",
        "exact_pca_coboundary_volume_axis_v1",
    }
    if config.get("method") not in accepted_methods:
        raise ValueError(f"Unexpected V5/coboundary method in {config_path}")
    if config.get("experiment") != experiment:
        raise ValueError(f"Config experiment must be {experiment}")
    expected_structure = "left_hippocampus" if structure == "hippocampus" else "left_lateral_ventricle"
    if config.get("structure") != expected_structure:
        raise ValueError(f"Config structure must be {expected_structure}")
    for section in ("input_config", "model", "training", "loss", "selection", "scientific_contract"):
        if section not in config:
            raise KeyError(f"Missing config section {section}")
    if int(config["model"]["latent_dim"]) != 150:
        raise ValueError("Cocycle-V5 requires PCA-150")
    expected_variants = {
        "c4_time_embedded": "direct_time_film",
        "c4_coboundary_exact": "coboundary_exact",
        "c4_coboundary_soft": "coboundary_soft",
        "c5_coboundary_exact_volume": "coboundary_exact",
        "c6_coboundary_exact_volume_selection": "coboundary_exact",
        "c7_coboundary_volume_axis": "coboundary_volume_axis",
    }
    expected_variant = expected_variants.get(experiment, "direct")
    if str(config["model"].get("variant", "direct")) != expected_variant:
        raise ValueError(f"Experiment {experiment} requires model.variant={expected_variant}")
    if experiment == "c4_time_embedded":
        time_dim = int(config["model"].get("time_embedding_dim", 0))
        frequencies = config["model"].get("time_frequencies")
        modulation_scale = float(config["model"].get("modulation_max_scale", 0.0))
        if time_dim <= 0 or not isinstance(frequencies, list) or not frequencies:
            raise ValueError("Time-embedded C4 requires a positive time_embedding_dim and non-empty time_frequencies")
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0 for value in frequencies):
            raise ValueError("time_frequencies must contain only positive finite numeric values")
        if not math.isfinite(modulation_scale) or modulation_scale <= 0.0:
            raise ValueError("Time-embedded C4 requires a positive finite modulation_max_scale")
    if experiment == "c4_coboundary_soft" and float(config["loss"].get("coboundary_weight", 0.0)) <= 0.0:
        raise ValueError("Soft coboundary experiment requires a positive coboundary_weight")
    if experiment in {"c6_coboundary_exact_volume_selection", "c7_coboundary_volume_axis"}:
        selection_weights = (
            float(config["selection"].get("first_last_volume_weight", 0.0)),
            float(config["selection"].get("first_last_rate_weight", 0.0)),
            float(config["selection"].get("group_trend_weight", 0.0)),
        )
        if any(weight <= 0.0 for weight in selection_weights):
            raise ValueError("C6/C7 requires positive first-last volume, rate, and group-trend selection weights")
    if experiment == "c7_coboundary_volume_axis" and float(config["loss"].get("volume_potential_weight", 0.0)) <= 0.0:
        raise ValueError("C7 requires a positive volume_potential_weight")
    if int(config["training"]["epochs"]) <= 0 or int(config["training"]["batch_size"]) <= 0:
        raise ValueError("Training epochs and batch size must be positive")


def checkpoint_payload(
    *,
    epoch: int,
    flow: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    config_path: Path,
    statistics: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "flow_state_dict": flow.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "config_path": str(config_path),
        "statistics": statistics,
        "validation": validation,
    }


def write_history(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config or default_config_path(args.structure, args.experiment))
    config = read_json(config_path)
    validate_config(config_path, config, args.structure, args.experiment)
    run_name = str(args.run_name or config["training"]["run_name"])
    validate_run_name(run_name)
    output_dir = output_directory(config_path, config, run_name)
    device = choose_device(args.device)
    seed = int(config["training"]["seed"])
    set_seed(seed)

    input_config = read_json(resolve_path(config["input_config"]))
    pca_model = validate_pca_model(input_config, 150)
    # Test inputs are deliberately not opened by this trainer.  They are used
    # only by the separate, read-only evaluator after checkpoint selection.
    archives = {split: load_archive(resolve_path(input_config["dataset"][f"{split}_sequences"]), split, 150) for split in ("train", "val")}
    pair_rows = {split: convert_pairs(load_pairs(resolve_path(input_config["dataset"][f"{split}_pairs"]), archives[split], split), archives[split]) for split in ("train", "val")}
    values = {split: values_on_device(archive, device) for split, archive in archives.items()}
    geometry = PcaGeometry(pca_model, archives["train"]["train_pca_mean_150"], archives["train"]["train_pca_std_150"]).to(device)
    geometry.eval()
    variant = str(config["model"].get("variant", "direct"))
    statistics = training_statistics(
        geometry,
        values["train"],
        pair_rows["train"],
        archives["train"],
        include_volume_axis=(variant == "coboundary_volume_axis"),
    )
    flow = build_flow(config["model"], statistics.get("volume_axis")).to(device)
    parameter_count = sum(parameter.numel() for parameter in flow.parameters())
    training = config["training"]
    print("=" * 96, flush=True)
    print(f"PCA transport | {args.structure} | {args.experiment} | variant={variant} | device={device} | parameters={parameter_count}", flush=True)
    print(f"Transport: {config['model']['transport']}; no ODE integration", flush=True)
    print(json.dumps({
        split: {"subjects": len(archives[split]["subject_ids"]), "visits": len(archives[split]["visit_scan_ids"]), "pairs": len(pair_rows[split])}
        for split in ("train", "val")
    }, sort_keys=True), flush=True)
    print("Train-only normalization:", json.dumps(statistics, sort_keys=True), flush=True)

    loader = DataLoader(
        CocyclePairDataset(pair_rows["train"]),
        batch_size=int(training["batch_size"]),
        sampler=balanced_pair_sampler(pair_rows["train"], seed, 0, int(training["samples_per_epoch"])),
        num_workers=0,
        collate_fn=collate_pairs,
        drop_last=False,
    )
    first_raw = next(iter(loader))
    pair = pair_terms(flow, geometry, values["train"], first_raw, statistics)
    zero = pair["real_pca"] * 0.0
    sequence = {"sequence_pca": zero, "sequence_vertex": zero, "sequence_semigroup": zero, "slope": zero}
    if any(float(config["loss"][key]) > 0.0 for key in ("sequence_pca_weight", "sequence_vertex_weight", "sequence_semigroup_weight", "slope_weight")):
        sequence = sequence_terms(flow, geometry, values["train"], archives["train"], balanced_sequence_starts(archives["train"], seed, 1)[0], statistics)
    probe_loss, probe_terms = total_loss(pair, sequence, config, 1)
    if not torch.isfinite(probe_loss):
        raise RuntimeError("Non-finite Cocycle-V5 dry-run loss")
    probe_loss.backward()
    gradient_norm = math.sqrt(sum(float(torch.sum(parameter.grad.detach().square()).cpu()) for parameter in flow.parameters() if parameter.grad is not None))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise RuntimeError("Invalid Cocycle-V5 dry-run gradients")
    flow.zero_grad(set_to_none=True)
    flow.eval()
    val_pairs = evaluate_pairs(flow, geometry, values["val"], pair_rows["val"], int(training["evaluation_batch_size"]))
    val_first_last = evaluate_pairs(flow, geometry, values["val"], first_last_pairs(archives["val"]), int(training["evaluation_batch_size"]))
    val_defects = cocycle_defects(flow, values["val"], pair_rows["val"], statistics, int(training["evaluation_batch_size"]))
    val_score, val_feasible, val_ratios = validation_score(val_pairs, val_first_last, val_defects, config["selection"])
    if args.dry_run:
        print("DRY RUN PASSED — finite gradients, forward/backward transport, and consistency validation; no files written.", flush=True)
        print(json.dumps({"loss": float(probe_loss.detach().cpu()), "gradient_l2_norm": gradient_norm, "terms": probe_terms, "val_score": val_score, "val_feasible": val_feasible, "val_ratios": val_ratios, "val_defects": val_defects}, indent=2, sort_keys=True), flush=True)
        return 0

    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output_dir}. Choose --run-name or use --resume.")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(f"--resume requires {checkpoint_dir / 'latest.pt'}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        uses_subject_context = variant.startswith("coboundary")
        atomic_json(output_dir / "resolved_config.json", config)
        atomic_json(output_dir / "training_statistics.json", statistics)
        atomic_json(output_dir / "run_contract.json", {
            "method": config["method"], "experiment": args.experiment, "structure": config["structure"],
            "model_variant": variant, "transport": config["model"]["transport"],
            "time_conditioning": config["model"].get("time_conditioning", "raw source/target age, signed/absolute interval, and midpoint"),
            "subject_context": "first longitudinal visit PCA state and normalized age, fixed within subject" if uses_subject_context else "not used by this direct transport",
            "ode_used": False, "attention_used": False, "cross_subject_operations": False, "test_loaded_during_training": False,
            "input_config": str(resolve_path(config["input_config"])), "input_config_sha256": sha256(resolve_path(config["input_config"])),
            "source_meshes_modified": False, "all_current_qc_passed_subjects_retained": True,
        })

    optimizer = torch.optim.AdamW(flow.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]), eta_min=float(training["minimum_learning_rate"]))
    history_path = output_dir / "history.jsonl"
    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    if args.resume:
        checkpoint = torch.load(checkpoint_dir / "latest.pt", map_location=device)
        flow.load_state_dict(checkpoint["flow_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        status = read_json(output_dir / "training_status.json")
        best_score = float(status["best_validation_score"])
        best_epoch = int(status["best_epoch"])
        stale = int(status["stale_epochs"])
    else:
        baseline_validation = {"score": val_score, "feasible": val_feasible, "ratios": val_ratios, "all_pairs": val_pairs, "first_last": val_first_last, "defects": val_defects, "checkpoint_role": "zero_velocity_no_change"}
        baseline_payload = checkpoint_payload(epoch=0, flow=flow, optimizer=optimizer, scheduler=scheduler, config=config, config_path=config_path, statistics=statistics, validation=baseline_validation)
        atomic_torch_save(checkpoint_dir / "epoch_0000_no_change.pt", baseline_payload)
        atomic_torch_save(checkpoint_dir / "best_shape.pt", baseline_payload)
        best_score, best_epoch = val_score, 0

    started = time.time()
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        flow.train()
        epoch_loader = DataLoader(
            CocyclePairDataset(pair_rows["train"]),
            batch_size=int(training["batch_size"]),
            sampler=balanced_pair_sampler(pair_rows["train"], seed, epoch, int(training["samples_per_epoch"])),
            num_workers=0,
            collate_fn=collate_pairs,
            drop_last=False,
        )
        sequence_starts = balanced_sequence_starts(archives["train"], seed, epoch)
        totals: dict[str, float] = {}
        batches = 0
        for step, raw in enumerate(epoch_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            pair = pair_terms(flow, geometry, values["train"], raw, statistics)
            zero = pair["real_pca"] * 0.0
            sequence = {"sequence_pca": zero, "sequence_vertex": zero, "sequence_semigroup": zero, "slope": zero}
            if any(float(config["loss"][key]) > 0.0 for key in ("sequence_pca_weight", "sequence_vertex_weight", "sequence_semigroup_weight", "slope_weight")):
                sequence = sequence_terms(flow, geometry, values["train"], archives["train"], sequence_starts[(step - 1) % len(sequence_starts)], statistics)
            loss, terms = total_loss(pair, sequence, config, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + value
            if step % int(training["progress_every_batches"]) == 0 or step == len(epoch_loader):
                print(f"epoch {epoch:03d}/{int(training['epochs'])} batch {step:03d}/{len(epoch_loader):03d} loss={terms['total']:.5f} real_v={terms['real_vertex']:.5f} semi={terms['virtual_semigroup']:.5f}", flush=True)
        scheduler.step()

        flow.eval()
        val_pairs = evaluate_pairs(flow, geometry, values["val"], pair_rows["val"], int(training["evaluation_batch_size"]))
        val_first_last = evaluate_pairs(flow, geometry, values["val"], first_last_pairs(archives["val"]), int(training["evaluation_batch_size"]))
        val_defects = cocycle_defects(flow, values["val"], pair_rows["val"], statistics, int(training["evaluation_batch_size"]))
        val_score, val_feasible, val_ratios = validation_score(val_pairs, val_first_last, val_defects, config["selection"])
        validation = {"score": val_score, "feasible": val_feasible, "ratios": val_ratios, "all_pairs": val_pairs, "first_last": val_first_last, "defects": val_defects}
        row = {
            "epoch": epoch, "elapsed_minutes": (time.time() - started) / 60.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]), "train_batches": batches,
            **{f"train_{name}": value / max(batches, 1) for name, value in totals.items()},
            "val_score": val_score, "val_feasible": val_feasible, **{f"val_{name}": value for name, value in val_ratios.items()},
            **{f"val_{name}": value for name, value in val_defects.items()},
        }
        write_history(history_path, row)
        payload = checkpoint_payload(epoch=epoch, flow=flow, optimizer=optimizer, scheduler=scheduler, config=config, config_path=config_path, statistics=statistics, validation=validation)
        atomic_torch_save(checkpoint_dir / "latest.pt", payload)
        if epoch % int(training["save_every_epochs"]) == 0:
            atomic_torch_save(checkpoint_dir / f"epoch_{epoch:04d}.pt", payload)
        if val_feasible and val_score < best_score - float(training["early_stopping_min_delta"]):
            best_score, best_epoch, stale = val_score, epoch, 0
            atomic_torch_save(checkpoint_dir / "best_shape.pt", payload)
        else:
            stale += 1
        atomic_json(output_dir / "training_status.json", {
            "status": "running", "epoch": epoch, "epochs_requested": int(training["epochs"]), "best_epoch": best_epoch,
            "best_validation_score": best_score, "stale_epochs": stale, "test_data_loaded": False,
        })
        print(f"epoch {epoch:03d}/{int(training['epochs'])} val_score={val_score:.6f} feasible={val_feasible} macro_firstlast={val_ratios['macro_first_last_shape']:.6f} semi={val_defects['relative_semigroup_defect_mean']:.5f} best={best_score:.6f}@{best_epoch}", flush=True)
        if stale >= int(training["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}: no feasible validation improvement for {stale} epochs.", flush=True)
            break

    selected = torch.load(checkpoint_dir / "best_shape.pt", map_location="cpu")
    atomic_json(output_dir / "final_report.json", {
        "status": "complete", "structure": config["structure"], "experiment": args.experiment, "run_name": run_name,
        "epochs_completed": epoch, "epochs_requested": int(training["epochs"]), "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(checkpoint_dir / "best_shape.pt"), "selected_validation": selected["validation"],
        "parameter_count": parameter_count, "test_loaded_during_training": False, "source_meshes_modified": False,
    })
    atomic_json(output_dir / "training_status.json", {
        "status": "complete", "epoch": epoch, "epochs_requested": int(training["epochs"]), "best_epoch": int(selected["epoch"]),
        "best_validation_score": float(selected["validation"]["score"]), "stale_epochs": stale, "test_data_loaded": False,
    })
    print(f"COMPLETE: selected epoch {int(selected['epoch'])}; checkpoint: {checkpoint_dir / 'best_shape.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
