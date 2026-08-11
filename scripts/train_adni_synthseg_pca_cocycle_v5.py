#!/usr/bin/env python3
"""Train the direct, non-ODE Cocycle-V5 model on strict ADNI SynthSeg PCA data.

The learned transport is deliberately the same direct-flow family used by the
earlier cocycle experiment:

    Phi(z, s, t, d) = z + (t - s) * phi_theta(z, s, t, d)

There is no ODE solver, recurrence hidden inside the transport, attention over
subjects, or cross-run state.  Semigroup and inverse properties are imposed by
losses on compositions of this direct map.  Every output directory belongs to
one (structure, experiment, run-name) tuple, so independent experiments can be
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
EXPERIMENTS = ("c1", "c2", "c3", "c4")


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
    ) -> torch.Tensor:
        source = source_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        target = target_time.reshape(-1, 1).to(dtype=latent.dtype, device=latent.device)
        return latent + (target - source) * self.average_velocity(latent, source, target, label_ad)


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
    return {
        "z": torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device),
        "age": torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device),
        "years": torch.from_numpy(archive["visit_time_years_from_baseline"].astype(np.float32)).to(device),
        "label": torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device),
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
def training_statistics(
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    rows: list[CocyclePair],
    archive: dict[str, np.ndarray],
    batch_size: int = 512,
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
    return {
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
    flow: DirectDiagnosisResidualCocycleFlow,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    statistics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    batch = indexed(values, raw)
    scales = statistics["normalization_scales"]
    prediction_forward = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
    prediction_backward = flow.transport(batch["target"], batch["target_age"], batch["source_age"], batch["label"])
    forward = shape_terms(prediction_forward, batch["target"], geometry, scales)
    backward = shape_terms(prediction_backward, batch["source"], geometry, scales)

    valid_middle = batch["intermediate_index"] >= 0
    observed = prediction_forward.sum() * 0.0
    if bool(valid_middle.any()):
        mid_index = batch["intermediate_index"][valid_middle]
        middle_age = values["age"][mid_index]
        labels = batch["label"][valid_middle]
        forward_middle = flow.transport(batch["source"][valid_middle], batch["source_age"][valid_middle], middle_age, labels)
        forward_composed = flow.transport(forward_middle, middle_age, batch["target_age"][valid_middle], labels)
        backward_middle = flow.transport(batch["target"][valid_middle], batch["target_age"][valid_middle], middle_age, labels)
        backward_composed = flow.transport(backward_middle, middle_age, batch["source_age"][valid_middle], labels)
        observed = 0.5 * (
            mean_scaled_mse(prediction_forward[valid_middle], forward_composed, scales["pca"])
            + mean_scaled_mse(prediction_backward[valid_middle], backward_composed, scales["pca"])
        )

    ratio = torch.empty_like(batch["source_age"]).uniform_(0.2, 0.8)
    virtual_age = batch["source_age"] + ratio * (batch["target_age"] - batch["source_age"])
    forward_middle = flow.transport(batch["source"], batch["source_age"], virtual_age, batch["label"])
    forward_composed = flow.transport(forward_middle, virtual_age, batch["target_age"], batch["label"])
    backward_middle = flow.transport(batch["target"], batch["target_age"], virtual_age, batch["label"])
    backward_composed = flow.transport(backward_middle, virtual_age, batch["source_age"], batch["label"])
    virtual = 0.5 * (
        mean_scaled_mse(prediction_forward, forward_composed, scales["pca"])
        + mean_scaled_mse(prediction_backward, backward_composed, scales["pca"])
    )

    inverse_forward = flow.transport(prediction_forward, batch["target_age"], batch["source_age"], batch["label"])
    inverse_backward = flow.transport(prediction_backward, batch["source_age"], batch["target_age"], batch["label"])
    inverse = 0.5 * (
        mean_scaled_mse(inverse_forward, batch["source"], scales["pca"])
        + mean_scaled_mse(inverse_backward, batch["target"], scales["pca"])
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
        "volume": volume,
        "rate": rate,
        "group_rate": group_rate,
        "disease_gap": disease_gap,
    }


def sequence_terms(
    flow: DirectDiagnosisResidualCocycleFlow,
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
    if z.shape[0] < 2:
        raise ValueError("Sequence loss requires at least two visits")
    scales = statistics["normalization_scales"]

    source = z[:1]
    source_age = ages[:1]
    targets = z[1:]
    target_ages = ages[1:]
    count = targets.shape[0]
    direct_forward = flow.transport(source.expand(count, -1), source_age.expand(count), target_ages, label.expand(count))
    rollout_forward: list[torch.Tensor] = []
    current = source
    previous_age = source_age
    for index in range(1, z.shape[0]):
        current = flow.transport(current, previous_age, ages[index : index + 1], label)
        rollout_forward.append(current)
        previous_age = ages[index : index + 1]
    rollout_forward_tensor = torch.cat(rollout_forward, dim=0)

    reverse_targets = z[:-1].flip(0)
    reverse_ages = ages[:-1].flip(0)
    reverse_source = z[-1:]
    reverse_source_age = ages[-1:]
    direct_backward = flow.transport(reverse_source.expand(count, -1), reverse_source_age.expand(count), reverse_ages, label.expand(count))
    rollout_backward: list[torch.Tensor] = []
    current = reverse_source
    previous_age = reverse_source_age
    for index in range(z.shape[0] - 2, -1, -1):
        current = flow.transport(current, previous_age, ages[index : index + 1], label)
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
            + float(loss_config["sequence_pca_weight"]) * sequence["sequence_pca"]
            + float(loss_config["sequence_vertex_weight"]) * sequence["sequence_vertex"]
            + float(loss_config["sequence_semigroup_weight"]) * sequence["sequence_semigroup"]
        )
        + anatomy_factor * (
            float(loss_config["volume_weight"]) * pair["volume"]
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
    flow: DirectDiagnosisResidualCocycleFlow,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    rows: list[CocyclePair],
    batch_size: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {diagnosis: {name: [] for name in (
        "pca", "coordinate", "euclidean", "volume_relative", "rate", "nochange_pca", "nochange_coordinate", "nochange_euclidean", "nochange_volume_relative", "nochange_rate"
    )} for diagnosis in ("CN", "AD", "overall")}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = indexed(values, collate_pairs(chunk))
        prediction = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        source_vertices = geometry.vertices(batch["source"])
        target_vertices = geometry.vertices(batch["target"])
        predicted_vertices = geometry.vertices(prediction)
        source_volume = geometry.volume_from_vertices(source_vertices)
        target_volume = geometry.volume_from_vertices(target_vertices)
        predicted_volume = geometry.volume_from_vertices(predicted_vertices)
        years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        metric_tensors = {
            "pca": torch.mean((prediction - batch["target"]).square(), dim=1),
            "coordinate": torch.mean(torch.abs(predicted_vertices - target_vertices), dim=(1, 2)),
            "euclidean": torch.linalg.vector_norm(predicted_vertices - target_vertices, dim=2).mean(dim=1),
            "volume_relative": torch.abs(predicted_volume - target_volume) / target_volume,
            "rate": torch.abs((torch.log(predicted_volume) - torch.log(target_volume)) / years),
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
    flow: DirectDiagnosisResidualCocycleFlow,
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
        direct = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        middle_age = 0.5 * (batch["source_age"] + batch["target_age"])
        middle = flow.transport(batch["source"], batch["source_age"], middle_age, batch["label"])
        composed = flow.transport(middle, middle_age, batch["target_age"], batch["label"])
        inverse = flow.transport(direct, batch["target_age"], batch["source_age"], batch["label"])
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
    feasible = True
    pca_allowance = 1.0 + float(selection["pca_nochange_tolerance"])
    coordinate_allowance = 1.0 + float(selection["coordinate_nochange_tolerance"])
    for diagnosis in ("CN", "AD"):
        values = first_last["groups"][diagnosis]
        coordinate_ratio = values["coordinate_mean"] / max(values["nochange_coordinate_mean"], 1.0e-8)
        euclidean_ratio = values["euclidean_mean"] / max(values["nochange_euclidean_mean"], 1.0e-8)
        pca_ratio = values["pca_mean"] / max(values["nochange_pca_mean"], 1.0e-8)
        macro_first_last.append(0.5 * (coordinate_ratio + euclidean_ratio))
        feasible = feasible and all(math.isfinite(item) for item in (coordinate_ratio, euclidean_ratio, pca_ratio))
        feasible = feasible and pca_ratio <= pca_allowance and coordinate_ratio <= coordinate_allowance
    all_values = all_pairs["groups"]["overall"]
    all_coordinate_ratio = all_values["coordinate_mean"] / max(all_values["nochange_coordinate_mean"], 1.0e-8)
    all_euclidean_ratio = all_values["euclidean_mean"] / max(all_values["nochange_euclidean_mean"], 1.0e-8)
    all_shape = 0.5 * (all_coordinate_ratio + all_euclidean_ratio)
    volume_ratio = all_values["volume_relative_mean"] / max(all_values["nochange_volume_relative_mean"], 1.0e-8)
    score = float(np.mean(macro_first_last)) + float(selection["all_pair_shape_weight"]) * all_shape + float(selection["volume_tiebreak_weight"]) * volume_ratio
    feasible = feasible and math.isfinite(score)
    feasible = feasible and defects["relative_semigroup_defect_mean"] <= float(selection["max_relative_semigroup_defect"])
    feasible = feasible and defects["relative_inverse_defect_mean"] <= float(selection["max_relative_inverse_defect"])
    ratios = {
        "macro_first_last_shape": float(np.mean(macro_first_last)),
        "all_pair_shape": float(all_shape),
        "all_pair_volume": float(volume_ratio),
        "score": score,
    }
    return score, bool(feasible), ratios


def output_directory(config_path: Path, config: dict[str, Any], run_name: str) -> Path:
    return config_path.parent.parent / "training" / run_name


def validate_config(config_path: Path, config: dict[str, Any], structure: str, experiment: str) -> None:
    if config.get("method") != "direct_pca_cocycle_v5":
        raise ValueError(f"Unexpected V5 method in {config_path}")
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
    if int(config["training"]["epochs"]) <= 0 or int(config["training"]["batch_size"]) <= 0:
        raise ValueError("Training epochs and batch size must be positive")


def checkpoint_payload(
    *,
    epoch: int,
    flow: DirectDiagnosisResidualCocycleFlow,
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
    statistics = training_statistics(geometry, values["train"], pair_rows["train"], archives["train"])
    flow = DirectDiagnosisResidualCocycleFlow(
        latent_dim=int(config["model"]["latent_dim"]),
        width=int(config["model"]["width"]),
        residual_blocks=int(config["model"]["residual_blocks"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in flow.parameters())
    training = config["training"]
    print("=" * 96, flush=True)
    print(f"Direct Cocycle-V5 | {args.structure} | {args.experiment} | device={device} | parameters={parameter_count}", flush=True)
    print("Transport: Phi(z,s,t,d) = z + (t-s) * phi(z,s,t,d); no ODE integration", flush=True)
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
        print("DRY RUN PASSED — finite direct-flow gradients, forward/backward transport, and semigroup validation; no files written.", flush=True)
        print(json.dumps({"loss": float(probe_loss.detach().cpu()), "gradient_l2_norm": gradient_norm, "terms": probe_terms, "val_score": val_score, "val_feasible": val_feasible, "val_ratios": val_ratios, "val_defects": val_defects}, indent=2, sort_keys=True), flush=True)
        return 0

    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output_dir}. Choose --run-name or use --resume.")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(f"--resume requires {checkpoint_dir / 'latest.pt'}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_json(output_dir / "resolved_config.json", config)
        atomic_json(output_dir / "training_statistics.json", statistics)
        atomic_json(output_dir / "run_contract.json", {
            "method": config["method"], "experiment": args.experiment, "structure": config["structure"], "transport": "Phi(z,s,t,d)=z+(t-s)*phi(z,s,t,d)",
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
