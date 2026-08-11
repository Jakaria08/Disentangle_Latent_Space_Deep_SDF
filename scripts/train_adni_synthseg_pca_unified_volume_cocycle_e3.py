#!/usr/bin/env python3
"""Train one unified, volume-aware PCA temporal flow (Experiment E3).

E3 deliberately contains one trainable model, one optimizer, and one
checkpoint family.  The diagnosis-conditioned direct flow predicts the final
PCA endpoint itself; there is no teacher, frozen V1, secondary speed head, or
post-hoc calibrator.  Applicable objectives from the earlier V1 and anchored
V4 experiments are applied directly to this one flow:

* observed forward/backward PCA and decoded-vertex endpoint supervision,
* decoded endpoint-volume and log-volume-rate supervision,
* subject and diagnosis-group volume-slope supervision,
* observed AD-minus-CN rate-gap supervision,
* forward/backward cycle, observed multi-visit composition, and identity, and
* velocity regularization.

Hippocampus and left lateral ventricle remain separate.  Inputs are strict
CN/AD subject-level splits.  The script refuses to overwrite an output and
never modifies meshes, PCA artifacts, QC files, or cohort files. ``--dry-run``
performs validation and a finite-gradient audit without writing an output.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_adni_synthseg_pca_cocycle_v4 import (
    BASE_ROOT,
    STRUCTURES,
    DirectAgeDiseaseTemporalFlow,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=STRUCTURES)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--run-name", default="pca150_unified_volume_cocycle_e3_seed42")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_config_path(structure: str) -> Path:
    return (
        BASE_ROOT
        / f"{structure}_pca_cocycle_v4"
        / "cocycle_v4"
        / "configs"
        / "unified_single_flow_e3_primary.json"
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def validate_run_name(value: str) -> None:
    if Path(value).name != value or value in {"", ".", ".."}:
        raise ValueError("--run-name must be one new directory-name component")


class PcaGeometry(nn.Module):
    """Fixed PCA decoder and differentiable closed-mesh volume evaluator."""

    def __init__(
        self,
        pca_model: dict[str, np.ndarray],
        score_mean: np.ndarray,
        score_std: np.ndarray,
    ) -> None:
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
class UnifiedPair:
    source: int
    target: int
    intermediate: int
    subject: str
    diagnosis: str
    pair_type: str
    gap_bin: str
    delta_years: float


class UnifiedPairDataset(Dataset[UnifiedPair]):
    def __init__(self, rows: list[UnifiedPair]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> UnifiedPair:
        return self.rows[index]


def convert_pairs(rows: list[PairRow], archive: dict[str, np.ndarray]) -> list[UnifiedPair]:
    subjects = archive["visit_subject_ids"].astype(str)
    output = [
        UnifiedPair(
            source=row.source_index,
            target=row.target_index,
            intermediate=row.intermediate_index,
            subject=str(subjects[row.source_index]),
            diagnosis=row.diagnosis,
            pair_type=row.pair_type,
            gap_bin="adjacent" if row.pair_type == "adjacent" else ("short" if row.delta_years <= 2.0 else "long"),
            delta_years=row.delta_years,
        )
        for row in rows
    ]
    if not output:
        raise ValueError("No eligible observed pairs")
    return output


def collate_pairs(rows: list[UnifiedPair]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.tensor([row.source for row in rows], dtype=torch.long),
        "target": torch.tensor([row.target for row in rows], dtype=torch.long),
        "intermediate": torch.tensor([row.intermediate for row in rows], dtype=torch.long),
        "is_ad": torch.tensor([row.diagnosis == "AD" for row in rows], dtype=torch.bool),
    }


def value_tensors(archive: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "z": torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device),
        "time": torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device),
        "years": torch.from_numpy(archive["visit_age_years"].astype(np.float32)).to(device),
        "label": torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device),
        "raw_volume": torch.from_numpy(archive["visit_volume_mm3"].astype(np.float32)).to(device),
    }


def indexed_batch(values: dict[str, torch.Tensor], raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source_index = raw["source"].to(values["z"].device)
    target_index = raw["target"].to(values["z"].device)
    intermediate_index = raw["intermediate"].to(values["z"].device)
    return {
        "source_index": source_index,
        "target_index": target_index,
        "intermediate_index": intermediate_index,
        "source": values["z"][source_index],
        "target": values["z"][target_index],
        "source_time": values["time"][source_index],
        "target_time": values["time"][target_index],
        "source_years": values["years"][source_index],
        "target_years": values["years"][target_index],
        "label": values["label"][source_index],
    }


def line_slope(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    centered_x = x - x.mean()
    denominator = torch.sum(centered_x.square()).clamp_min(1.0e-8)
    return torch.sum(centered_x * (y - y.mean())) / denominator


def sequence_starts(archive: dict[str, np.ndarray], diagnosis: str | None = None) -> list[int]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    diagnoses = archive["visit_diagnoses"].astype(str)
    return [
        int(offsets[index])
        for index in range(len(offsets) - 1)
        if offsets[index + 1] - offsets[index] >= 3
        and (diagnosis is None or diagnoses[offsets[index]] == diagnosis)
    ]


def subject_end(archive: dict[str, np.ndarray], start: int) -> int:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    return int(offsets[np.searchsorted(offsets, int(start), side="right")])


def balanced_sequence_order(archive: dict[str, np.ndarray], seed: int, epoch: int, maximum: int) -> list[int]:
    groups = {diagnosis: sequence_starts(archive, diagnosis) for diagnosis in ("CN", "AD")}
    if not all(groups.values()):
        raise ValueError(f"Both CN and AD require >=3-visit training sequences: { {k: len(v) for k, v in groups.items()} }")
    for index, diagnosis in enumerate(("CN", "AD")):
        random.Random(seed + epoch * 104_729 + index).shuffle(groups[diagnosis])
    requested = maximum if maximum > 0 else 2 * max(len(groups["CN"]), len(groups["AD"]))
    order: list[int] = []
    cursor = {"CN": 0, "AD": 0}
    while len(order) < requested:
        for diagnosis in ("CN", "AD"):
            if len(order) >= requested:
                break
            values = groups[diagnosis]
            order.append(values[cursor[diagnosis] % len(values)])
            cursor[diagnosis] += 1
    return order


def pair_sampler(rows: list[UnifiedPair], samples_per_epoch: int, seed: int, epoch: int) -> WeightedRandomSampler:
    subject_counts: dict[str, int] = {}
    stratum_counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        subject_counts[row.subject] = subject_counts.get(row.subject, 0) + 1
        key = (row.diagnosis, row.pair_type, row.gap_bin)
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
    weights = torch.tensor(
        [
            1.0 / math.sqrt(subject_counts[row.subject] * stratum_counts[(row.diagnosis, row.pair_type, row.gap_bin)])
            for row in rows
        ],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
    return WeightedRandomSampler(
        weights,
        num_samples=int(samples_per_epoch) if int(samples_per_epoch) > 0 else len(rows),
        replacement=True,
        generator=generator,
    )


@torch.no_grad()
def training_statistics(
    archive: dict[str, np.ndarray],
    values: dict[str, torch.Tensor],
    pairs: list[UnifiedPair],
    geometry: PcaGeometry,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(UnifiedPairDataset(pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in ("pca", "vertex", "volume", "rate")}
    rates_by_subject: dict[str, dict[str, list[float]]] = {"CN": {}, "AD": {}}
    for raw in loader:
        batch = indexed_batch(values, raw)
        source_vertices = geometry.vertices(batch["source"])
        target_vertices = geometry.vertices(batch["target"])
        source_volume = geometry.volume_from_vertices(source_vertices)
        target_volume = geometry.volume_from_vertices(target_vertices)
        gap = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        observed_rate = (torch.log(target_volume) - torch.log(source_volume)) / gap
        collected["pca"].append(torch.mean((batch["source"] - batch["target"]) ** 2, dim=1).cpu().numpy())
        collected["vertex"].append(torch.mean(torch.abs(source_vertices - target_vertices), dim=(1, 2)).cpu().numpy())
        collected["volume"].append((torch.abs(source_volume - target_volume) / target_volume.clamp_min(1.0e-8)).cpu().numpy())
        collected["rate"].append(torch.abs(observed_rate).cpu().numpy())
        for local_index, row_index in enumerate(raw["source"].tolist()):
            diagnosis = "AD" if bool(raw["is_ad"][local_index]) else "CN"
            subject = str(archive["visit_subject_ids"][row_index])
            rates_by_subject[diagnosis].setdefault(subject, []).append(float(observed_rate[local_index].cpu()))

    def median_scale(name: str) -> float:
        array = np.concatenate(collected[name]).astype(np.float64)
        return float(max(np.median(array[np.isfinite(array)]), 1.0e-6))

    group_targets: dict[str, float] = {}
    for diagnosis in ("CN", "AD"):
        subject_means = [float(np.mean(items)) for items in rates_by_subject[diagnosis].values()]
        if not subject_means:
            raise ValueError(f"No train-only rate target for {diagnosis}")
        group_targets[diagnosis] = float(np.mean(subject_means))

    scales = {
        "pca": median_scale("pca"),
        "vertex": median_scale("vertex"),
        "volume": median_scale("volume"),
        "rate": median_scale("rate"),
        "slope": median_scale("rate"),
        "consistency": median_scale("pca"),
    }
    return {
        "normalization_scales": scales,
        "group_log_volume_rate_targets": group_targets,
        "ad_minus_cn_log_volume_rate_target": group_targets["AD"] - group_targets["CN"],
        "pair_rows": len(pairs),
        "subjects_by_diagnosis": {diagnosis: len(values) for diagnosis, values in rates_by_subject.items()},
        "statistics_source": "train split only; observed forward pairs; subject-balanced group means",
    }


def subject_slope_loss(
    model: DirectAgeDiseaseTemporalFlow,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    start: int,
    slope_scale: float,
    huber_delta: float,
) -> torch.Tensor:
    end = subject_end(archive, start)
    target_index = torch.arange(start + 1, end, device=values["z"].device)
    count = int(target_index.numel())
    source = values["z"][start : start + 1].expand(count, -1)
    source_time = values["time"][start : start + 1].expand(count)
    source_years = values["years"][start : start + 1].expand(count)
    label = values["label"][start : start + 1].expand(count)
    predicted = model.transport(source, source_time, values["time"][target_index], label)
    elapsed = values["years"][target_index] - values["years"][start]
    predicted_slope = line_slope(elapsed, torch.log(geometry.volume(predicted)))
    observed_slope = line_slope(elapsed, torch.log(geometry.volume(values["z"][target_index])))
    normalized = (predicted_slope - observed_slope) / max(slope_scale, 1.0e-6)
    return F.huber_loss(normalized, torch.zeros_like(normalized), delta=huber_delta)


def batch_terms(
    model: DirectAgeDiseaseTemporalFlow,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    statistics: dict[str, Any],
    loss_config: dict[str, Any],
    auxiliary_ramp: float,
    sequence_archive: dict[str, np.ndarray] | None = None,
    sequence_start: int | None = None,
) -> dict[str, torch.Tensor]:
    batch = indexed_batch(values, raw)
    scales = statistics["normalization_scales"]
    predicted_target = model.transport(batch["source"], batch["source_time"], batch["target_time"], batch["label"])
    predicted_source = model.transport(batch["target"], batch["target_time"], batch["source_time"], batch["label"])

    pca_forward = torch.mean((predicted_target - batch["target"]) ** 2, dim=1).mean() / scales["pca"]
    pca_backward = torch.mean((predicted_source - batch["source"]) ** 2, dim=1).mean() / scales["pca"]
    source_vertices = geometry.vertices(batch["source"])
    target_vertices = geometry.vertices(batch["target"])
    predicted_target_vertices = geometry.vertices(predicted_target)
    predicted_source_vertices = geometry.vertices(predicted_source)
    vertex_forward = torch.mean(torch.abs(predicted_target_vertices - target_vertices), dim=(1, 2)).mean() / scales["vertex"]
    vertex_backward = torch.mean(torch.abs(predicted_source_vertices - source_vertices), dim=(1, 2)).mean() / scales["vertex"]

    source_volume = geometry.volume_from_vertices(source_vertices)
    target_volume = geometry.volume_from_vertices(target_vertices)
    predicted_target_volume = geometry.volume_from_vertices(predicted_target_vertices)
    predicted_source_volume = geometry.volume_from_vertices(predicted_source_vertices)
    volume_forward = torch.mean(torch.abs(predicted_target_volume - target_volume) / target_volume.clamp_min(1.0e-8)) / scales["volume"]
    volume_backward = torch.mean(torch.abs(predicted_source_volume - source_volume) / source_volume.clamp_min(1.0e-8)) / scales["volume"]

    gap = batch["target_years"] - batch["source_years"]
    safe_gap = gap.clamp_min(1.0e-6)
    observed_rate = (torch.log(target_volume) - torch.log(source_volume)) / safe_gap
    predicted_forward_rate = (torch.log(predicted_target_volume) - torch.log(source_volume)) / safe_gap
    predicted_backward_rate = (torch.log(predicted_source_volume) - torch.log(target_volume)) / (-safe_gap)
    rate_forward = F.huber_loss(
        (predicted_forward_rate - observed_rate) / scales["rate"],
        torch.zeros_like(observed_rate),
        delta=float(loss_config["huber_delta"]),
    )
    rate_backward = F.huber_loss(
        (predicted_backward_rate - observed_rate) / scales["rate"],
        torch.zeros_like(observed_rate),
        delta=float(loss_config["huber_delta"]),
    )

    labels = batch["label"] >= 0.5
    group_slope = torch.zeros((), device=batch["source"].device)
    for diagnosis, mask in (("CN", ~labels), ("AD", labels)):
        if bool(mask.any()):
            target = torch.tensor(
                float(statistics["group_log_volume_rate_targets"][diagnosis]),
                device=batch["source"].device,
                dtype=batch["source"].dtype,
            )
            group_slope = group_slope + ((predicted_forward_rate[mask].mean() - target) / scales["rate"]) ** 2
    if bool(labels.any()) and bool((~labels).any()):
        predicted_gap = predicted_forward_rate[labels].mean() - predicted_forward_rate[~labels].mean()
        target_gap = torch.tensor(
            float(statistics["ad_minus_cn_log_volume_rate_target"]),
            device=batch["source"].device,
            dtype=batch["source"].dtype,
        )
        disease_rate_gap = ((predicted_gap - target_gap) / scales["rate"]) ** 2
    else:
        disease_rate_gap = torch.zeros((), device=batch["source"].device)

    if sequence_archive is not None and sequence_start is not None:
        subject_slope = subject_slope_loss(
            model,
            geometry,
            values,
            sequence_archive,
            sequence_start,
            scales["slope"],
            float(loss_config["huber_delta"]),
        )
    else:
        subject_slope = torch.zeros((), device=batch["source"].device)

    cycle_source = model.transport(predicted_target, batch["target_time"], batch["source_time"], batch["label"])
    cycle_target = model.transport(predicted_source, batch["source_time"], batch["target_time"], batch["label"])
    cycle_forward = torch.mean((cycle_source - batch["source"]) ** 2) / scales["consistency"]
    cycle_backward = torch.mean((cycle_target - batch["target"]) ** 2) / scales["consistency"]

    valid_intermediate = batch["intermediate_index"] >= 0
    if bool(valid_intermediate.any()):
        middle_index = batch["intermediate_index"][valid_intermediate]
        middle_time = values["time"][middle_index]
        predicted_middle = model.transport(
            batch["source"][valid_intermediate],
            batch["source_time"][valid_intermediate],
            middle_time,
            batch["label"][valid_intermediate],
        )
        composed_target = model.transport(
            predicted_middle,
            middle_time,
            batch["target_time"][valid_intermediate],
            batch["label"][valid_intermediate],
        )
        composition = torch.mean((composed_target - predicted_target[valid_intermediate]) ** 2) / scales["consistency"]
    else:
        composition = torch.zeros((), device=batch["source"].device)

    identity = model.transport(batch["source"], batch["source_time"], batch["source_time"], batch["label"])
    zero_displacement = torch.mean((identity - batch["source"]) ** 2) / scales["consistency"]
    velocity_forward = model.velocity(batch["source"], batch["source_time"], batch["target_time"], batch["label"])
    velocity_backward = model.velocity(batch["target"], batch["target_time"], batch["source_time"], batch["label"])
    velocity = 0.5 * (torch.mean(velocity_forward.square()) + torch.mean(velocity_backward.square()))

    total = (
        float(loss_config["pca_forward_weight"]) * pca_forward
        + float(loss_config["pca_backward_weight"]) * pca_backward
        + float(loss_config["vertex_forward_weight"]) * vertex_forward
        + float(loss_config["vertex_backward_weight"]) * vertex_backward
        + auxiliary_ramp
        * (
            float(loss_config["volume_forward_weight"]) * volume_forward
            + float(loss_config["volume_backward_weight"]) * volume_backward
            + float(loss_config["rate_forward_weight"]) * rate_forward
            + float(loss_config["rate_backward_weight"]) * rate_backward
            + float(loss_config["subject_slope_weight"]) * subject_slope
            + float(loss_config["group_slope_weight"]) * group_slope
            + float(loss_config["disease_rate_gap_weight"]) * disease_rate_gap
        )
        + float(loss_config["cycle_forward_weight"]) * cycle_forward
        + float(loss_config["cycle_backward_weight"]) * cycle_backward
        + float(loss_config["composition_weight"]) * composition
        + float(loss_config["zero_displacement_weight"]) * zero_displacement
        + float(loss_config["velocity_weight"]) * velocity
    )
    return {
        "total": total,
        "pca_forward": pca_forward,
        "pca_backward": pca_backward,
        "vertex_forward": vertex_forward,
        "vertex_backward": vertex_backward,
        "volume_forward": volume_forward,
        "volume_backward": volume_backward,
        "rate_forward": rate_forward,
        "rate_backward": rate_backward,
        "subject_slope": subject_slope,
        "group_slope": group_slope,
        "disease_rate_gap": disease_rate_gap,
        "cycle_forward": cycle_forward,
        "cycle_backward": cycle_backward,
        "composition": composition,
        "zero_displacement": zero_displacement,
        "velocity": velocity,
    }


@torch.no_grad()
def evaluate_pairs(
    model: DirectAgeDiseaseTemporalFlow,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    pairs: list[UnifiedPair],
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(UnifiedPairDataset(pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    groups: dict[str, dict[str, list[float]]] = {"CN": {}, "AD": {}, "overall": {}}
    for raw in loader:
        batch = indexed_batch(values, raw)
        predicted_target = model.transport(batch["source"], batch["source_time"], batch["target_time"], batch["label"])
        predicted_source = model.transport(batch["target"], batch["target_time"], batch["source_time"], batch["label"])
        cycle = model.transport(predicted_target, batch["target_time"], batch["source_time"], batch["label"])
        source_vertices = geometry.vertices(batch["source"])
        target_vertices = geometry.vertices(batch["target"])
        predicted_target_vertices = geometry.vertices(predicted_target)
        predicted_source_vertices = geometry.vertices(predicted_source)
        source_volume = geometry.volume_from_vertices(source_vertices)
        target_volume = geometry.volume_from_vertices(target_vertices)
        predicted_target_volume = geometry.volume_from_vertices(predicted_target_vertices)
        predicted_source_volume = geometry.volume_from_vertices(predicted_source_vertices)
        gap = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        observed_rate = (torch.log(target_volume) - torch.log(source_volume)) / gap
        predicted_rate = (torch.log(predicted_target_volume) - torch.log(source_volume)) / gap
        backward_rate = (torch.log(predicted_source_volume) - torch.log(target_volume)) / (-gap)
        raw_target_volume = values["raw_volume"][batch["target_index"]]
        metrics = {
            "pca_mse": torch.mean((predicted_target - batch["target"]) ** 2, dim=1),
            "vertex_mae": torch.mean(torch.abs(predicted_target_vertices - target_vertices), dim=(1, 2)),
            "volume_relative": torch.abs(predicted_target_volume - target_volume) / target_volume.clamp_min(1.0e-8),
            "raw_mesh_volume_relative": torch.abs(predicted_target_volume - raw_target_volume) / raw_target_volume.clamp_min(1.0e-8),
            "log_volume_rate_abs_error": torch.abs(predicted_rate - observed_rate),
            "backward_pca_mse": torch.mean((predicted_source - batch["source"]) ** 2, dim=1),
            "backward_vertex_mae": torch.mean(torch.abs(predicted_source_vertices - source_vertices), dim=(1, 2)),
            "backward_volume_relative": torch.abs(predicted_source_volume - source_volume) / source_volume.clamp_min(1.0e-8),
            "backward_log_volume_rate_abs_error": torch.abs(backward_rate - observed_rate),
            "cycle_pca_mse": torch.mean((cycle - batch["source"]) ** 2, dim=1),
            "predicted_log_volume_rate": predicted_rate,
            "observed_log_volume_rate": observed_rate,
            "predicted_volume_change_mm3_per_year": (predicted_target_volume - source_volume) / gap,
            "observed_volume_change_mm3_per_year": (target_volume - source_volume) / gap,
            "no_change_pca_mse": torch.mean((batch["source"] - batch["target"]) ** 2, dim=1),
            "no_change_vertex_mae": torch.mean(torch.abs(source_vertices - target_vertices), dim=(1, 2)),
            "no_change_volume_relative": torch.abs(source_volume - target_volume) / target_volume.clamp_min(1.0e-8),
            "no_change_log_volume_rate_abs_error": torch.abs(observed_rate),
        }
        labels = batch["label"] >= 0.5
        for name, mask in (("AD", labels), ("CN", ~labels), ("overall", torch.ones_like(labels, dtype=torch.bool))):
            if not bool(mask.any()):
                continue
            bucket = groups[name]
            for metric, tensor in metrics.items():
                bucket.setdefault(metric, []).extend(tensor[mask].cpu().tolist())
    summary: dict[str, Any] = {"groups": {}}
    for name, bucket in groups.items():
        summary["groups"][name] = {"rows": len(bucket.get("pca_mse", []))}
        for metric, items in bucket.items():
            array = np.asarray(items, dtype=np.float64)
            summary["groups"][name][f"{metric}_mean"] = float(np.mean(array)) if array.size else float("nan")
            summary["groups"][name][f"{metric}_median"] = float(np.median(array)) if array.size else float("nan")
    ad = summary["groups"]["AD"]
    cn = summary["groups"]["CN"]
    summary["ad_minus_cn_predicted_log_volume_rate"] = ad["predicted_log_volume_rate_mean"] - cn["predicted_log_volume_rate_mean"]
    summary["ad_minus_cn_observed_log_volume_rate"] = ad["observed_log_volume_rate_mean"] - cn["observed_log_volume_rate_mean"]
    return summary


@torch.no_grad()
def evaluate_subject_slopes(
    model: DirectAgeDiseaseTemporalFlow,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for diagnosis in ("CN", "AD"):
        errors: list[float] = []
        predicted_values: list[float] = []
        observed_values: list[float] = []
        no_change_errors: list[float] = []
        for start in sequence_starts(archive, diagnosis):
            end = subject_end(archive, start)
            target_index = torch.arange(start + 1, end, device=values["z"].device)
            count = int(target_index.numel())
            prediction = model.transport(
                values["z"][start : start + 1].expand(count, -1),
                values["time"][start : start + 1].expand(count),
                values["time"][target_index],
                values["label"][start : start + 1].expand(count),
            )
            elapsed = values["years"][target_index] - values["years"][start]
            predicted_slope = float(line_slope(elapsed, torch.log(geometry.volume(prediction))).cpu())
            observed_slope = float(line_slope(elapsed, torch.log(geometry.volume(values["z"][target_index]))).cpu())
            errors.append(abs(predicted_slope - observed_slope))
            predicted_values.append(predicted_slope)
            observed_values.append(observed_slope)
            no_change_errors.append(abs(observed_slope))
        correlation = (
            float(np.corrcoef(predicted_values, observed_values)[0, 1])
            if len(errors) > 1 and np.std(predicted_values) > 1.0e-12 and np.std(observed_values) > 1.0e-12
            else float("nan")
        )
        output[diagnosis] = {
            "subjects": len(errors),
            "slope_abs_error_mean": float(np.mean(errors)) if errors else float("nan"),
            "no_change_slope_abs_error_mean": float(np.mean(no_change_errors)) if no_change_errors else float("nan"),
            "predicted_slope_mean": float(np.mean(predicted_values)) if predicted_values else float("nan"),
            "observed_slope_mean": float(np.mean(observed_values)) if observed_values else float("nan"),
            "slope_pearson": correlation,
        }
    return output


def validation_score(
    pair_metrics: dict[str, Any],
    slope_metrics: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[float, bool, dict[str, float]]:
    overall = pair_metrics["groups"]["overall"]
    epsilon = 1.0e-8
    ratios = {
        "pca": overall["pca_mse_mean"] / max(overall["no_change_pca_mse_mean"], epsilon),
        "vertex": overall["vertex_mae_mean"] / max(overall["no_change_vertex_mae_mean"], epsilon),
        "volume": overall["volume_relative_mean"] / max(overall["no_change_volume_relative_mean"], epsilon),
        "rate": overall["log_volume_rate_abs_error_mean"] / max(overall["no_change_log_volume_rate_abs_error_mean"], epsilon),
        "subject_slope": float(np.mean([
            slope_metrics[diagnosis]["slope_abs_error_mean"] / max(slope_metrics[diagnosis]["no_change_slope_abs_error_mean"], epsilon)
            for diagnosis in ("CN", "AD")
        ])),
    }
    score = (
        float(selection["pca_weight"]) * ratios["pca"]
        + float(selection["vertex_weight"]) * ratios["vertex"]
        + float(selection["volume_weight"]) * ratios["volume"]
        + float(selection["rate_weight"]) * ratios["rate"]
        + float(selection["subject_slope_weight"]) * ratios["subject_slope"]
    )
    tolerance = float(selection["shape_no_change_tolerance"])
    feasible = all(
        math.isfinite(float(pair_metrics["groups"][diagnosis][metric]))
        and float(pair_metrics["groups"][diagnosis][metric])
        <= float(pair_metrics["groups"][diagnosis][baseline]) * (1.0 + tolerance)
        for diagnosis in ("CN", "AD", "overall")
        for metric, baseline in (("pca_mse_mean", "no_change_pca_mse_mean"), ("vertex_mae_mean", "no_change_vertex_mae_mean"))
    )
    feasible = feasible and math.isfinite(score)
    return float(score), bool(feasible), ratios


@torch.no_grad()
def volume_reconstruction_audit(
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, float]:
    errors: list[np.ndarray] = []
    decoded_values: list[np.ndarray] = []
    for start in range(0, values["z"].shape[0], batch_size):
        decoded = geometry.volume(values["z"][start : start + batch_size])
        raw = values["raw_volume"][start : start + batch_size].clamp_min(1.0e-8)
        errors.append((torch.abs(decoded - raw) / raw).cpu().numpy())
        decoded_values.append(decoded.cpu().numpy())
    error = np.concatenate(errors)
    decoded = np.concatenate(decoded_values)
    if not np.isfinite(error).all() or np.any(decoded <= 0.0):
        raise ValueError("Non-finite or non-positive decoded PCA volume")
    return {
        "visits": int(error.size),
        "relative_error_mean": float(np.mean(error)),
        "relative_error_median": float(np.median(error)),
        "relative_error_p95": float(np.quantile(error, 0.95)),
        "decoded_volume_min_mm3": float(np.min(decoded)),
        "decoded_volume_max_mm3": float(np.max(decoded)),
    }


def checkpoint_payload(
    model: DirectAgeDiseaseTemporalFlow,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    input_config: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "input_config": input_config,
        "training_statistics": statistics,
        "model_contract": {
            "trainable_models": 1,
            "model_type": "DirectAgeDiseaseTemporalFlow",
            "teacher_model": False,
            "frozen_v1": False,
            "secondary_speed_head": False,
            "optimizer_count": 1,
        },
    }


def validate_config(config_path: Path, config: dict[str, Any], structure: str) -> tuple[Path, dict[str, Any]]:
    expected_structure = {"hippocampus": "left_hippocampus", "lateral_ventricle": "left_lateral_ventricle"}[structure]
    if config.get("structure") != expected_structure:
        raise ValueError(f"Structure mismatch in {config_path}")
    if config.get("method") != "unified_single_trainable_volume_aware_temporal_flow_e3":
        raise ValueError(f"Not an E3 single-flow configuration: {config_path}")
    contract = config.get("model_contract", {})
    if contract.get("trainable_models") != 1 or contract.get("secondary_speed_head") is not False:
        raise ValueError("E3 must contain exactly one trainable model and no speed head")
    input_path = resolve_path(config["input_config"])
    input_config = read_json(input_path)
    if input_config.get("structure") != expected_structure:
        raise ValueError("E3 input config structure mismatch")
    if int(input_config["representation"]["components"]) != 150:
        raise ValueError("E3 requires PCA-150")
    return input_path, input_config


def main() -> int:
    args = parse_args()
    validate_run_name(args.run_name)
    config_path = args.config or default_config_path(args.structure)
    config = read_json(config_path)
    input_config_path, input_config = validate_config(config_path, config, args.structure)
    device = choose_device(args.device)
    training = config["training"]
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else training["batch_size"])
    seed = int(args.seed if args.seed is not None else training["seed"])
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive")
    set_seed(seed)

    archives = {
        split: load_archive(Path(input_config["dataset"][f"{split}_sequences"]), split, 150)
        for split in ("train", "val", "test")
    }
    pairs = {
        split: convert_pairs(
            load_pairs(Path(input_config["dataset"][f"{split}_pairs"]), archives[split], split),
            archives[split],
        )
        for split in ("train", "val", "test")
    }
    pca_model = validate_pca_model(input_config, 150)
    values = {split: value_tensors(archive, device) for split, archive in archives.items()}
    geometry = PcaGeometry(
        pca_model,
        archives["train"]["train_pca_mean_150"],
        archives["train"]["train_pca_std_150"],
    ).to(device)
    statistics = training_statistics(archives["train"], values["train"], pairs["train"], geometry, batch_size)
    reconstruction_audit = {
        split: volume_reconstruction_audit(geometry, values[split], batch_size)
        for split in ("train", "val", "test")
    }
    model = DirectAgeDiseaseTemporalFlow(
        latent_dim=150,
        hidden_dims=config["model"]["hidden_dims"],
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)

    print("=" * 96, flush=True)
    print(f"E3 unified single flow | {args.structure} | device={device} | PCA-150", flush=True)
    print("Trainable models=1; optimizers=1; teacher=no; frozen V1=no; speed head=no", flush=True)
    print("Strict cohort=CN/AD only; backward pairs=derived from observed forward endpoints", flush=True)
    for split in ("train", "val", "test"):
        print(
            f"  {split}: pairs={len(pairs[split])} CN={sum(row.diagnosis == 'CN' for row in pairs[split])} "
            f"AD={sum(row.diagnosis == 'AD' for row in pairs[split])} "
            f"adjacent={sum(row.pair_type == 'adjacent' for row in pairs[split])} "
            f"nonadjacent={sum(row.pair_type != 'adjacent' for row in pairs[split])}",
            flush=True,
        )
    print("Train-only group log-volume-rate targets:", statistics["group_log_volume_rate_targets"], flush=True)
    print("PCA-150 volume reconstruction audit:", json.dumps(reconstruction_audit, sort_keys=True), flush=True)

    if args.dry_run:
        sequence_order = balanced_sequence_order(archives["train"], seed, 1, int(training["sequence_subjects_per_epoch"]))
        loader = DataLoader(
            UnifiedPairDataset(pairs["train"]),
            batch_size=batch_size,
            sampler=pair_sampler(pairs["train"], max(batch_size, 2 * batch_size), seed, 1),
            collate_fn=collate_pairs,
            num_workers=args.num_workers,
        )
        raw = next(iter(loader))
        terms = batch_terms(
            model,
            geometry,
            values["train"],
            raw,
            statistics,
            config["loss"],
            auxiliary_ramp=1.0,
            sequence_archive=archives["train"],
            sequence_start=sequence_order[0],
        )
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise RuntimeError("Non-finite E3 dry-run loss")
        terms["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not gradients or not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("E3 gradient audit failed")
        gradient_norm = float(torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)).cpu())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError("E3 gradient norm is not positive and finite")
        model.eval()
        val_pair_metrics = evaluate_pairs(model, geometry, values["val"], pairs["val"], batch_size)
        val_slope_metrics = evaluate_subject_slopes(model, geometry, values["val"], archives["val"])
        val_score, val_feasible, val_ratios = validation_score(
            val_pair_metrics,
            val_slope_metrics,
            config["selection"],
        )
        if not math.isfinite(val_score) or not val_feasible:
            raise RuntimeError("E3 no-change validation/checkpoint-selection audit failed")
        print("DRY RUN PASSED — one model, one loss graph, finite backward gradients, no files written.", flush=True)
        print(
            json.dumps(
                {name: float(value.detach().cpu()) for name, value in terms.items()}
                | {
                    "gradient_l2_norm": gradient_norm,
                    "epoch_zero_validation_score": val_score,
                    "epoch_zero_validation_feasible": val_feasible,
                    "epoch_zero_ratios_to_no_change": val_ratios,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    output_dir = config_path.parents[1] / "training" / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite E3 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config_used.json")
    shutil.copy2(input_config_path, output_dir / "input_config_used.json")
    atomic_json(output_dir / "training_statistics.json", statistics)
    atomic_json(output_dir / "pca_volume_reconstruction_audit.json", reconstruction_audit)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training["minimum_learning_rate"]),
    )
    validation_frequency = int(training["validation_frequency"])
    patience = int(training["early_stopping_patience"])
    ramp_epochs = int(training["auxiliary_ramp_epochs"])
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    started = time.time()

    epoch_zero_pairs = evaluate_pairs(model, geometry, values["val"], pairs["val"], batch_size)
    epoch_zero_slopes = evaluate_subject_slopes(model, geometry, values["val"], archives["val"])
    epoch_zero_score, epoch_zero_feasible, epoch_zero_ratios = validation_score(epoch_zero_pairs, epoch_zero_slopes, config["selection"])
    epoch_zero_metrics = {
        "val_pairs": epoch_zero_pairs,
        "val_subject_slopes": epoch_zero_slopes,
        "val_score": epoch_zero_score,
        "val_feasible": epoch_zero_feasible,
        "val_ratios_to_no_change": epoch_zero_ratios,
        "checkpoint_role": "zero_velocity_no_change_fallback",
    }
    atomic_torch_save(
        output_dir / "checkpoints" / "epoch_0000_no_change.pt",
        checkpoint_payload(model, optimizer, 0, epoch_zero_metrics, config, input_config, statistics),
    )
    if epoch_zero_feasible:
        best_score = epoch_zero_score
        atomic_torch_save(
            output_dir / "checkpoints" / "best_feasible_composite.pt",
            checkpoint_payload(model, optimizer, 0, epoch_zero_metrics, config, input_config, statistics),
        )

    history_path = output_dir / "history.jsonl"
    completed_epoch = 0
    stopped_early = False
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, epochs + 1):
            completed_epoch = epoch
            model.train()
            sequence_order = balanced_sequence_order(
                archives["train"], seed, epoch, int(training["sequence_subjects_per_epoch"])
            )
            loader = DataLoader(
                UnifiedPairDataset(pairs["train"]),
                batch_size=batch_size,
                sampler=pair_sampler(pairs["train"], int(training["samples_per_epoch"]), seed, epoch),
                collate_fn=collate_pairs,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            auxiliary_ramp = min(1.0, epoch / max(ramp_epochs, 1))
            sums: dict[str, float] = {}
            trained_rows = 0
            for step, raw in enumerate(loader):
                terms = batch_terms(
                    model,
                    geometry,
                    values["train"],
                    raw,
                    statistics,
                    config["loss"],
                    auxiliary_ramp,
                    archives["train"],
                    sequence_order[step % len(sequence_order)],
                )
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                optimizer.step()
                rows = int(raw["source"].shape[0])
                trained_rows += rows
                for name, value in terms.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * rows
            scheduler.step()
            record: dict[str, Any] = {
                "epoch": epoch,
                "elapsed_minutes": (time.time() - started) / 60.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "auxiliary_ramp": auxiliary_ramp,
                **{f"train_{name}": value / max(trained_rows, 1) for name, value in sums.items()},
            }
            validation_due = epoch == 1 or epoch % validation_frequency == 0 or epoch == epochs
            if validation_due:
                model.eval()
                val_pairs = evaluate_pairs(model, geometry, values["val"], pairs["val"], batch_size)
                val_slopes = evaluate_subject_slopes(model, geometry, values["val"], archives["val"])
                score, is_feasible, ratios = validation_score(val_pairs, val_slopes, config["selection"])
                record.update({
                    "val_score": score,
                    "val_feasible": is_feasible,
                    "val_ratio_pca": ratios["pca"],
                    "val_ratio_vertex": ratios["vertex"],
                    "val_ratio_volume": ratios["volume"],
                    "val_ratio_rate": ratios["rate"],
                    "val_ratio_subject_slope": ratios["subject_slope"],
                })
                metrics = {
                    "val_pairs": val_pairs,
                    "val_subject_slopes": val_slopes,
                    "val_score": score,
                    "val_feasible": is_feasible,
                    "val_ratios_to_no_change": ratios,
                }
                payload = checkpoint_payload(model, optimizer, epoch, metrics, config, input_config, statistics)
                atomic_torch_save(output_dir / "checkpoints" / "latest.pt", payload)
                if is_feasible and score < best_score - float(training["early_stopping_min_delta"]):
                    best_score, best_epoch, stale = score, epoch, 0
                    atomic_torch_save(output_dir / "checkpoints" / "best_feasible_composite.pt", payload)
                else:
                    stale += validation_frequency
            history.write(json.dumps(record, sort_keys=True) + "\n")
            history.flush()
            progress = f"epoch {epoch:03d}/{epochs} train={record['train_total']:.6f} ramp={auxiliary_ramp:.2f}"
            if validation_due:
                progress += f" val={record['val_score']:.5f} feasible={record['val_feasible']} best={best_score:.5f}"
            print(progress, flush=True)
            if validation_due and stale >= patience:
                stopped_early = True
                print(f"Early stopping: no feasible composite improvement for {stale} epochs.", flush=True)
                break

    selected_path = output_dir / "checkpoints" / "best_feasible_composite.pt"
    if not selected_path.is_file():
        raise RuntimeError("No feasible E3 checkpoint, including the no-change fallback")
    selected = torch.load(selected_path, map_location=device)
    model.load_state_dict(selected["model_state_dict"])
    model.eval()
    test_pairs = evaluate_pairs(model, geometry, values["test"], pairs["test"], batch_size)
    test_slopes = evaluate_subject_slopes(model, geometry, values["test"], archives["test"])
    final_report = {
        "status": "complete",
        "experiment": "E3 unified single-model volume-aware temporal flow",
        "structure": config["structure"],
        "selected_checkpoint": str(selected_path),
        "selected_epoch": int(selected["epoch"]),
        "selected_is_no_change_fallback": int(selected["epoch"]) == 0,
        "best_validation_score": float(selected["metrics"]["val_score"]),
        "epochs_requested": epochs,
        "epochs_completed": completed_epoch,
        "stopped_early": stopped_early,
        "test_pairs": test_pairs,
        "test_subject_slopes": test_slopes,
        "model_contract": {
            "trainable_models": 1,
            "optimizer_count": 1,
            "teacher_model": False,
            "frozen_model": False,
            "secondary_speed_head": False,
        },
        "strict_no_mci": True,
        "source_meshes_modified": False,
        "pca_refitted": False,
    }
    atomic_json(output_dir / "final_report.json", final_report)
    atomic_json(output_dir / "run_contract.json", {
        "status": "complete",
        "config": str(config_path),
        "input_config": str(input_config_path),
        "output": str(output_dir),
        "seed": seed,
        **final_report["model_contract"],
        "strict_no_mci": True,
        "source_meshes_modified": False,
    })
    print("=" * 96, flush=True)
    print(f"E3 training complete; selected epoch={final_report['selected_epoch']}; output={output_dir}", flush=True)
    print(json.dumps(final_report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
