#!/usr/bin/env python3
"""Run a structure-specific PCA V1 -> frozen-V1 anchored V4 experiment.

One invocation trains the real-pair, bidirectional PCA flow (V1), freezes its
best validation checkpoint, then calibrates only the *AD* displacement speed
(V4).  CN predictions are mathematically identical to V1 in the V4 phase.

Hippocampus and left lateral ventricle are intentionally separate experiments:
they have separate PCA bases, QC cohorts, volume trajectories, and outputs.
The script never reads MCI data, never changes meshes/PCA inputs, and refuses
to overwrite a prior phase directory.  ``--dry-run`` validates both stages and
executes one no-gradient V4 loss batch without creating an output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import sys
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
    BASE_ROOT,
    STRUCTURES,
    DirectAgeDiseaseTemporalFlow,
    PairRow,
    atomic_json,
    atomic_torch_save,
    choose_device,
    collate_pairs,
    load_archive,
    load_pairs,
    read_json,
    set_seed,
    validate_pca_model,
)


SCRIPT_PATH = Path(__file__).resolve()
V1_SCRIPT = SCRIPT_PATH.with_name("train_adni_synthseg_pca_cocycle_v4.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=STRUCTURES)
    parser.add_argument("--phase", choices=("all", "v1", "v4"), default="all")
    parser.add_argument("--run-name", default="pca150_v1_anchored_ad_speed_v4_seed42")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, or cuda:N")
    parser.add_argument("--v1-config", type=Path, default=None)
    parser.add_argument("--v4-config", type=Path, default=None)
    parser.add_argument("--v1-epochs", type=int, default=None)
    parser.add_argument("--v4-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_v1_config(structure: str) -> Path:
    return BASE_ROOT / f"{structure}_pca_cocycle_v4" / "cocycle_v4" / "configs" / "cocycle_v4_primary.json"


def default_v4_config(structure: str) -> Path:
    return BASE_ROOT / f"{structure}_pca_cocycle_v4" / "cocycle_v4" / "configs" / "v1_anchored_ad_speed_v4_primary.json"


def require_single_name(value: str, flag: str) -> None:
    if Path(value).name != value or value in {"", ".", ".."}:
        raise ValueError(f"{flag} must be one directory-name component")


def phase_directories(v1_config: Path, run_name: str) -> tuple[Path, Path, Path]:
    training_root = v1_config.parents[1] / "training"
    return (
        training_root / f"{run_name}_v1",
        training_root / f"{run_name}_v4",
        training_root / f"{run_name}_pipeline.json",
    )


def assert_new(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing run artifact(s): {joined}")


def load_checked_configs(args: argparse.Namespace) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    v1_path = args.v1_config or default_v1_config(args.structure)
    v4_path = args.v4_config or default_v4_config(args.structure)
    v1 = read_json(v1_path)
    v4 = read_json(v4_path)
    expected_structure = {"hippocampus": "left_hippocampus", "lateral_ventricle": "left_lateral_ventricle"}[args.structure]
    for path, config in ((v1_path, v1), (v4_path, v4)):
        if config.get("structure") != expected_structure:
            raise ValueError(f"Structure mismatch in {path}")
        if int(config.get("representation", {}).get("components", config.get("components", -1))) != 150:
            raise ValueError(f"Only PCA-150 is supported: {path}")
    if v4.get("method") != "frozen_v1_anchored_ad_speed_calibration":
        raise ValueError(f"Not an anchored-V4 configuration: {v4_path}")
    return v1_path, v1, v4_path, v4


def run_v1(args: argparse.Namespace, v1_path: Path, *, dry_run: bool) -> None:
    command = [
        sys.executable, str(V1_SCRIPT), "--structure", args.structure,
        "--config", str(v1_path), "--device", args.device,
        "--num-workers", str(args.num_workers),
    ]
    if dry_run:
        command.append("--dry-run")
    else:
        command.extend(["--run-name", f"{args.run_name}_v1"])
        if args.v1_epochs is not None:
            command.extend(["--epochs", str(args.v1_epochs)])
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
    print("=" * 88, flush=True)
    print("PHASE 1/2: V1 direct bidirectional real-pair Cocycle", flush=True)
    print("Command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def mesh_volume_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    faces = faces.to(device=vertices.device, dtype=torch.long)
    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    signed = torch.sum(v0 * torch.cross(v1, v2, dim=2), dim=2).sum(dim=1) / 6.0
    return torch.clamp(torch.abs(signed), min=1.0e-8)


def decode_pca_torch(
    standardized: torch.Tensor,
    score_mean: torch.Tensor,
    score_std: torch.Tensor,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    raw = standardized * score_std + score_mean
    flat = raw @ components + mean_flat
    return flat.reshape(raw.shape[0], -1, 3)


class AdSpeedHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        widths = [int(input_dim), *[int(width) for width in hidden_dims], 1]
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        output_layer = next(layer for layer in reversed(self.net) if isinstance(layer, nn.Linear))
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class V1AnchoredAdSpeedCalibrator(nn.Module):
    """Frozen V1 transport with a positive AD-only displacement multiplier."""

    def __init__(
        self,
        *,
        base_flow: DirectAgeDiseaseTemporalFlow,
        model_config: dict[str, Any],
        feature_stats: dict[str, np.ndarray],
        pca_model: dict[str, np.ndarray],
        train_score_mean: np.ndarray,
        train_score_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.base_flow = base_flow
        for parameter in self.base_flow.parameters():
            parameter.requires_grad_(False)
        self.base_flow.eval()
        self.feature_pcs = int(model_config["feature_pcs"])
        self.minimum_speed = float(model_config["minimum_speed"])
        self.maximum_speed = float(model_config["maximum_speed"])
        self.individual_log_span = float(model_config["individual_log_span"])
        if not 0.0 < self.minimum_speed < 1.0 < self.maximum_speed:
            raise ValueError("V4 speed limits must enclose one")
        initial_speed = float(model_config["initial_ad_speed"])
        if not self.minimum_speed < initial_speed < self.maximum_speed:
            raise ValueError("V4 initial AD speed must lie inside speed limits")
        self.head = AdSpeedHead(self.feature_pcs + 6, model_config["hidden_dims"], float(model_config["dropout"]))
        fraction = (initial_speed - self.minimum_speed) / (self.maximum_speed - self.minimum_speed)
        self.global_speed_logit = nn.Parameter(torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=torch.float32))
        self.register_buffer("coefficient_mean", torch.from_numpy(feature_stats["coefficient_mean"]).float().view(1, -1))
        self.register_buffer("coefficient_std", torch.from_numpy(feature_stats["coefficient_std"]).float().clamp_min(1.0e-6).view(1, -1))
        self.register_buffer("scalar_mean", torch.from_numpy(feature_stats["scalar_mean"]).float().view(1, -1))
        self.register_buffer("scalar_std", torch.from_numpy(feature_stats["scalar_std"]).float().clamp_min(1.0e-6).view(1, -1))
        self.register_buffer("score_mean", torch.from_numpy(train_score_mean).float().view(1, -1))
        self.register_buffer("score_std", torch.from_numpy(train_score_std).float().clamp_min(1.0e-6).view(1, -1))
        self.register_buffer("mean_flat", torch.from_numpy(pca_model["mean"]).float().view(1, -1))
        self.register_buffer("components", torch.from_numpy(pca_model["components"]).float())
        self.register_buffer("faces", torch.from_numpy(pca_model["faces"]).long())

    @property
    def global_speed(self) -> torch.Tensor:
        return self.minimum_speed + (self.maximum_speed - self.minimum_speed) * torch.sigmoid(self.global_speed_logit)

    def set_global_speed(self, speed: float) -> None:
        if not self.minimum_speed < float(speed) < self.maximum_speed:
            raise ValueError("Requested speed lies outside configured open bounds")
        fraction = (float(speed) - self.minimum_speed) / (self.maximum_speed - self.minimum_speed)
        with torch.no_grad():
            self.global_speed_logit.copy_(torch.tensor(math.log(fraction / (1.0 - fraction)), device=self.global_speed_logit.device))

    def vertices(self, standardized: torch.Tensor) -> torch.Tensor:
        return decode_pca_torch(standardized, self.score_mean, self.score_std, self.mean_flat, self.components)

    def volume(self, standardized: torch.Tensor) -> torch.Tensor:
        return mesh_volume_torch(self.vertices(standardized), self.faces)

    def _features(
        self,
        source: torch.Tensor,
        base: torch.Tensor,
        source_norm: torch.Tensor,
        target_norm: torch.Tensor,
        source_years: torch.Tensor,
        target_years: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            latent = (source[:, : self.feature_pcs] - self.coefficient_mean[:, : self.feature_pcs]) / self.coefficient_std[:, : self.feature_pcs]
            source_volume = self.volume(source)
            base_volume = self.volume(base)
            gap = target_years - source_years
            safe_gap = torch.where(gap.abs() < 1.0e-6, torch.full_like(gap, 1.0e-6), gap)
            scalars = torch.stack((
                source_norm,
                target_norm,
                torch.log1p(safe_gap.abs()),
                torch.log(source_volume),
                (torch.log(base_volume) - torch.log(source_volume)) / safe_gap,
                torch.linalg.norm(base - source, dim=1) / safe_gap.abs().clamp_min(1.0e-6),
            ), dim=1)
            scalars = (scalars - self.scalar_mean) / self.scalar_std
        return torch.cat((latent, scalars), dim=1)

    def transport(
        self,
        source: torch.Tensor,
        source_norm: torch.Tensor,
        target_norm: torch.Tensor,
        source_years: torch.Tensor,
        target_years: torch.Tensor,
        label_ad: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.base_flow.eval()
        with torch.no_grad():
            base = self.base_flow.transport(source, source_norm, target_norm, label_ad)
        features = self._features(source, base, source_norm, target_norm, source_years, target_years)
        residual = self.individual_log_span * torch.tanh(self.head(features))
        ad_speed = torch.clamp(self.global_speed * torch.exp(residual), self.minimum_speed, self.maximum_speed)
        speed = torch.where(label_ad >= 0.5, ad_speed, torch.ones_like(ad_speed))
        prediction = source + speed.unsqueeze(1) * (base - source)
        return prediction, {"base": base, "speed": speed, "individual_log_speed": residual}

    def train(self, mode: bool = True) -> "V1AnchoredAdSpeedCalibrator":
        super().train(mode)
        self.base_flow.eval()
        return self


@dataclass(frozen=True)
class V4Pair:
    source: int
    target: int
    diagnosis: str
    subject: str


class V4PairDataset(Dataset[V4Pair]):
    def __init__(self, pairs: list[V4Pair]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> V4Pair:
        return self.pairs[index]


def collate_v4(rows: list[V4Pair]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.tensor([row.source for row in rows], dtype=torch.long),
        "target": torch.tensor([row.target for row in rows], dtype=torch.long),
        "is_ad": torch.tensor([row.diagnosis == "AD" for row in rows], dtype=torch.bool),
    }


def v4_pairs(rows: list[PairRow], archive: dict[str, np.ndarray], diagnosis: str | None = None) -> list[V4Pair]:
    subjects = archive["visit_subject_ids"].astype(str)
    output = []
    for row in rows:
        if diagnosis is not None and row.diagnosis != diagnosis:
            continue
        output.append(V4Pair(row.source_index, row.target_index, row.diagnosis, str(subjects[row.source_index])))
    if not output:
        raise ValueError(f"No {diagnosis or 'eligible'} observed pairs")
    return output


def value_tensors(archive: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "z": torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device),
        "norm": torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device),
        "years": torch.from_numpy(archive["visit_age_years"].astype(np.float32)).to(device),
        "label": torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device),
    }


def tensor_batch(values: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = batch["source"].to(values["z"].device)
    target = batch["target"].to(values["z"].device)
    return {
        "source": values["z"][source], "target": values["z"][target],
        "source_norm": values["norm"][source], "target_norm": values["norm"][target],
        "source_years": values["years"][source], "target_years": values["years"][target],
        "label": values["label"][source],
    }


def line_slope(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean()
    return torch.sum(centered * (y - y.mean())) / torch.sum(centered.square()).clamp_min(1.0e-8)


def ad_sequence_starts(archive: dict[str, np.ndarray]) -> list[int]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    diagnoses = archive["visit_diagnoses"].astype(str)
    return [
        int(offsets[index]) for index in range(len(offsets) - 1)
        if offsets[index + 1] - offsets[index] >= 3 and diagnoses[offsets[index]] == "AD"
    ]


def sequence_slope_loss(
    model: V1AnchoredAdSpeedCalibrator,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    start: int,
    scale: float,
    huber_delta: float,
) -> torch.Tensor:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    end = int(offsets[np.searchsorted(offsets, start, side="right")])
    target_indices = torch.arange(start + 1, end, device=values["z"].device)
    count = int(target_indices.numel())
    source = values["z"][start : start + 1].expand(count, -1)
    prediction, _ = model.transport(
        source,
        values["norm"][start : start + 1].expand(count), values["norm"][target_indices],
        values["years"][start : start + 1].expand(count), values["years"][target_indices],
        values["label"][start : start + 1].expand(count),
    )
    times = values["years"][target_indices] - values["years"][start]
    observed = line_slope(times, torch.log(model.volume(values["z"][target_indices])))
    predicted = line_slope(times, torch.log(model.volume(prediction)))
    return F.huber_loss((predicted - observed) / max(scale, 1.0e-6), torch.zeros_like(predicted), delta=huber_delta)


def base_statistics(
    model: V1AnchoredAdSpeedCalibrator,
    values: dict[str, torch.Tensor],
    pairs: list[V4Pair],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    loader = DataLoader(V4PairDataset(pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_v4)
    coefficients: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    pca_errors: list[np.ndarray] = []
    vertex_errors: list[np.ndarray] = []
    rate_errors: list[np.ndarray] = []
    with torch.no_grad():
        for raw in loader:
            batch = tensor_batch(values, raw)
            base = model.base_flow.transport(batch["source"], batch["source_norm"], batch["target_norm"], batch["label"])
            source_volume = model.volume(batch["source"])
            target_volume = model.volume(batch["target"])
            base_volume = model.volume(base)
            gap = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
            feature_scalars = torch.stack((
                batch["source_norm"], batch["target_norm"], torch.log1p(gap), torch.log(source_volume),
                (torch.log(base_volume) - torch.log(source_volume)) / gap,
                torch.linalg.norm(base - batch["source"], dim=1) / gap,
            ), dim=1)
            coefficients.append(batch["source"].cpu().numpy())
            scalars.append(feature_scalars.cpu().numpy())
            pca_errors.append(torch.mean((base - batch["target"]) ** 2, dim=1).cpu().numpy())
            vertex_errors.append(torch.mean(torch.abs(model.vertices(base) - model.vertices(batch["target"])), dim=(1, 2)).cpu().numpy())
            rate_errors.append(torch.abs((torch.log(base_volume) - torch.log(target_volume)) / gap).cpu().numpy())
    def positive_median(chunks: list[np.ndarray]) -> float:
        values_flat = np.concatenate(chunks) if chunks else np.asarray([1.0], dtype=np.float32)
        return float(max(np.median(values_flat), 1.0e-6))
    coefficient = np.concatenate(coefficients, axis=0)
    scalar = np.concatenate(scalars, axis=0)
    return {
        "coefficient_mean": coefficient.mean(axis=0).astype(np.float32),
        "coefficient_std": np.maximum(coefficient.std(axis=0), 1.0e-6).astype(np.float32),
        "scalar_mean": scalar.mean(axis=0).astype(np.float32),
        "scalar_std": np.maximum(scalar.std(axis=0), 1.0e-6).astype(np.float32),
    }, {
        "pca": positive_median(pca_errors),
        "vertex": positive_median(vertex_errors),
        "rate": positive_median(rate_errors),
        "slope": positive_median(rate_errors),
        "consistency": positive_median(pca_errors),
    }


def loss_terms(
    model: V1AnchoredAdSpeedCalibrator,
    values: dict[str, torch.Tensor],
    raw_batch: dict[str, torch.Tensor],
    scales: dict[str, float],
    loss_config: dict[str, Any],
    sequence_archive: dict[str, np.ndarray] | None = None,
    sequence_start: int | None = None,
) -> dict[str, torch.Tensor]:
    batch = tensor_batch(values, raw_batch)
    predicted, diagnostics = model.transport(**{
        "source": batch["source"], "source_norm": batch["source_norm"], "target_norm": batch["target_norm"],
        "source_years": batch["source_years"], "target_years": batch["target_years"], "label_ad": batch["label"],
    })
    base = diagnostics["base"]
    pca_per = torch.mean((predicted - batch["target"]) ** 2, dim=1)
    base_pca_per = torch.mean((base - batch["target"]) ** 2, dim=1)
    vertex_per = torch.mean(torch.abs(model.vertices(predicted) - model.vertices(batch["target"])), dim=(1, 2))
    base_vertex_per = torch.mean(torch.abs(model.vertices(base) - model.vertices(batch["target"])), dim=(1, 2))
    source_volume = model.volume(batch["source"])
    target_volume = model.volume(batch["target"])
    predicted_volume = model.volume(predicted)
    gap = batch["target_years"] - batch["source_years"]
    safe_gap = torch.where(gap.abs() < 1.0e-6, torch.full_like(gap, 1.0e-6), gap)
    observed_rate = (torch.log(target_volume) - torch.log(source_volume)) / safe_gap
    predicted_rate = (torch.log(predicted_volume) - torch.log(source_volume)) / safe_gap
    pca = pca_per.mean() / scales["pca"]
    vertex = vertex_per.mean() / scales["vertex"]
    pair_rate = F.huber_loss((predicted_rate - observed_rate) / scales["rate"], torch.zeros_like(predicted_rate), delta=float(loss_config["huber_delta"]))
    group_slope = F.mse_loss(predicted_rate.mean() / scales["rate"], observed_rate.mean() / scales["rate"])
    if sequence_archive is not None and sequence_start is not None:
        subject_slope = sequence_slope_loss(model, values, sequence_archive, sequence_start, scales["slope"], float(loss_config["huber_delta"]))
    else:
        subject_slope = torch.zeros((), device=predicted.device)
    midpoint_ratio = torch.rand_like(safe_gap)
    midpoint_years = batch["source_years"] + midpoint_ratio * safe_gap
    midpoint_norm = batch["source_norm"] + midpoint_ratio * (batch["target_norm"] - batch["source_norm"])
    midpoint, _ = model.transport(batch["source"], batch["source_norm"], midpoint_norm, batch["source_years"], midpoint_years, batch["label"])
    composed, _ = model.transport(midpoint, midpoint_norm, batch["target_norm"], midpoint_years, batch["target_years"], batch["label"])
    cocycle = torch.mean((composed - predicted.detach()) ** 2) / scales["consistency"]
    inverse, _ = model.transport(predicted, batch["target_norm"], batch["source_norm"], batch["target_years"], batch["source_years"], batch["label"])
    inverse_loss = torch.mean((inverse - batch["source"]) ** 2) / scales["consistency"]
    tolerance = float(loss_config["no_degradation_tolerance"])
    no_degradation = torch.mean(F.relu((pca_per - base_pca_per * (1.0 + tolerance)) / scales["pca"]))
    no_degradation = no_degradation + torch.mean(F.relu((vertex_per - base_vertex_per * (1.0 + tolerance)) / scales["vertex"]))
    speed_regularization = torch.mean(diagnostics["individual_log_speed"] ** 2)
    total = (
        float(loss_config["pca_weight"]) * pca + float(loss_config["vertex_weight"]) * vertex
        + float(loss_config["pair_rate_weight"]) * pair_rate + float(loss_config["subject_slope_weight"]) * subject_slope
        + float(loss_config["group_slope_weight"]) * group_slope + float(loss_config["cocycle_weight"]) * cocycle
        + float(loss_config["inverse_weight"]) * inverse_loss + float(loss_config["no_degradation_weight"]) * no_degradation
        + float(loss_config["speed_regularization_weight"]) * speed_regularization
    )
    return {
        "total": total, "pca": pca, "vertex": vertex, "pair_rate": pair_rate, "subject_slope": subject_slope,
        "group_slope": group_slope, "cocycle": cocycle, "inverse": inverse_loss,
        "no_degradation": no_degradation, "speed_regularization": speed_regularization,
    }


@torch.no_grad()
def evaluate(
    model: V1AnchoredAdSpeedCalibrator,
    values: dict[str, torch.Tensor],
    pairs: list[V4Pair],
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(V4PairDataset(pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_v4)
    groups: dict[str, dict[str, list[float]]] = {"CN": {}, "AD": {}, "overall": {}}
    cn_delta: list[float] = []
    for raw in loader:
        batch = tensor_batch(values, raw)
        predicted, diagnostics = model.transport(batch["source"], batch["source_norm"], batch["target_norm"], batch["source_years"], batch["target_years"], batch["label"])
        base = diagnostics["base"]
        source_volume = model.volume(batch["source"])
        target_volume = model.volume(batch["target"])
        predicted_volume = model.volume(predicted)
        gap = (batch["target_years"] - batch["source_years"]).abs().clamp_min(1.0e-6)
        metrics = {
            "pca": torch.mean((predicted - batch["target"]) ** 2, dim=1),
            "vertex": torch.mean(torch.abs(model.vertices(predicted) - model.vertices(batch["target"])), dim=(1, 2)),
            "volume_relative": torch.abs(predicted_volume - target_volume) / target_volume.clamp_min(1.0e-8),
            "rate": torch.abs((torch.log(predicted_volume) - torch.log(target_volume)) / gap),
            "speed": diagnostics["speed"],
        }
        labels = batch["label"] >= 0.5
        for name, mask in (("AD", labels), ("CN", ~labels), ("overall", torch.ones_like(labels, dtype=torch.bool))):
            if bool(mask.any()):
                for metric, tensor in metrics.items():
                    groups[name].setdefault(metric, []).extend(tensor[mask].detach().cpu().tolist())
        if bool((~labels).any()):
            cn_delta.extend(torch.max(torch.abs(predicted[~labels] - base[~labels]), dim=1).values.cpu().tolist())
    return {
        "groups": {
            name: {f"{metric}_mean": float(np.mean(values)) if values else float("nan") for metric, values in bucket.items()} | {"rows": len(bucket.get("pca", []))}
            for name, bucket in groups.items()
        },
        "cn_max_abs_delta_from_v1": float(max(cn_delta, default=0.0)),
    }


@torch.no_grad()
def ad_slope_error(
    model: V1AnchoredAdSpeedCalibrator,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
) -> dict[str, float]:
    errors: list[float] = []
    for start in ad_sequence_starts(archive):
        offsets = archive["subject_visit_offsets"].astype(np.int64)
        end = int(offsets[np.searchsorted(offsets, start, side="right")])
        targets = torch.arange(start + 1, end, device=values["z"].device)
        count = int(targets.numel())
        prediction, _ = model.transport(
            values["z"][start : start + 1].expand(count, -1), values["norm"][start : start + 1].expand(count), values["norm"][targets],
            values["years"][start : start + 1].expand(count), values["years"][targets], values["label"][start : start + 1].expand(count),
        )
        times = values["years"][targets] - values["years"][start]
        errors.append(float(torch.abs(line_slope(times, torch.log(model.volume(prediction))) - line_slope(times, torch.log(model.volume(values["z"][targets])))).cpu()))
    return {"subjects": len(errors), "ad_subject_slope_abs_error": float(np.mean(errors)) if errors else float("nan")}


def first_last_pairs(pairs: list[V4Pair], archive: dict[str, np.ndarray]) -> list[V4Pair]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    endpoint = {int(offsets[index]): int(offsets[index + 1] - 1) for index in range(len(offsets) - 1)}
    result = [pair for pair in pairs if endpoint.get(pair.source) == pair.target]
    if not result:
        raise ValueError("No first-to-last pairs available for V4 feasibility evaluation")
    return result


def feasible(
    current: dict[str, Any],
    base: dict[str, Any],
    current_first_last: dict[str, Any],
    base_first_last: dict[str, Any],
    selection: dict[str, Any],
) -> bool:
    if current["cn_max_abs_delta_from_v1"] > 2.0e-6:
        return False
    for group, allowance in (("AD", float(selection["endpoint_tolerance"])), ("overall", float(selection["endpoint_tolerance"]))):
        for metric in ("pca_mean", "vertex_mean"):
            old = float(base["groups"][group].get(metric, float("nan")))
            new = float(current["groups"][group].get(metric, float("nan")))
            if not math.isfinite(old) or not math.isfinite(new) or new > old * (1.0 + allowance):
                return False
    for group in ("AD", "overall"):
        for metric in ("pca_mean", "vertex_mean"):
            old = float(base_first_last["groups"][group].get(metric, float("nan")))
            new = float(current_first_last["groups"][group].get(metric, float("nan")))
            if not math.isfinite(old) or not math.isfinite(new) or new > old * (1.0 + float(selection["first_last_tolerance"])):
                return False
    return True


def state_snapshot(model: V1AnchoredAdSpeedCalibrator) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.base_flow.state_dict().items()}


def assert_frozen(model: V1AnchoredAdSpeedCalibrator, before: dict[str, torch.Tensor]) -> None:
    for name, value in before.items():
        if not torch.equal(value, model.base_flow.state_dict()[name].detach().cpu()):
            raise RuntimeError(f"Frozen V1 parameter changed: {name}")


def make_base_flow(v1_config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> DirectAgeDiseaseTemporalFlow:
    model = DirectAgeDiseaseTemporalFlow(
        latent_dim=150,
        hidden_dims=v1_config["model"]["hidden_dims"],
        dropout=float(v1_config["model"].get("dropout", 0.0)),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def v4_checkpoint_payload(
    model: V1AnchoredAdSpeedCalibrator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    v1_checkpoint: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics, "v1_checkpoint": str(v1_checkpoint), "config": config,
        "model_config": {
            "feature_pcs": model.feature_pcs, "minimum_speed": model.minimum_speed, "maximum_speed": model.maximum_speed,
            "individual_log_span": model.individual_log_span,
        },
    }


def train_v4(
    args: argparse.Namespace,
    v1_path: Path,
    v1_config: dict[str, Any],
    v4_path: Path,
    v4_config: dict[str, Any],
    v1_dir: Path,
    v4_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    device = choose_device(args.device)
    seed = int(args.seed if args.seed is not None else v4_config["training"]["seed"])
    set_seed(seed)
    archives = {split: load_archive(Path(v1_config["dataset"][f"{split}_sequences"]), split, 150) for split in ("train", "val", "test")}
    raw_pairs = {split: load_pairs(Path(v1_config["dataset"][f"{split}_pairs"]), archives[split], split) for split in ("train", "val", "test")}
    pca_model = validate_pca_model(v1_config, 150)
    train_values, val_values, test_values = (value_tensors(archives[split], device) for split in ("train", "val", "test"))
    if dry_run:
        base = DirectAgeDiseaseTemporalFlow(150, v1_config["model"]["hidden_dims"], float(v1_config["model"].get("dropout", 0.0))).to(device)
    else:
        v1_checkpoint = v1_dir / "checkpoint_best.pt"
        if not v1_checkpoint.is_file() or not (v1_dir / "final_report.json").is_file():
            raise FileNotFoundError(f"Completed V1 checkpoint/report required before V4: {v1_dir}")
        base = make_base_flow(v1_config, v1_checkpoint, device)
    train_ad_pairs = v4_pairs(raw_pairs["train"], archives["train"], diagnosis="AD")
    val_all_pairs = v4_pairs(raw_pairs["val"], archives["val"])
    test_all_pairs = v4_pairs(raw_pairs["test"], archives["test"])
    temporary_model = V1AnchoredAdSpeedCalibrator(
        base_flow=base, model_config=v4_config["model"],
        feature_stats={"coefficient_mean": np.zeros(150, np.float32), "coefficient_std": np.ones(150, np.float32), "scalar_mean": np.zeros(6, np.float32), "scalar_std": np.ones(6, np.float32)},
        pca_model=pca_model, train_score_mean=archives["train"]["train_pca_mean_150"], train_score_std=archives["train"]["train_pca_std_150"],
    ).to(device)
    feature_stats, scales = base_statistics(temporary_model, train_values, train_ad_pairs, int(v4_config["training"]["batch_size"]))
    model = V1AnchoredAdSpeedCalibrator(
        base_flow=base, model_config=v4_config["model"], feature_stats=feature_stats, pca_model=pca_model,
        train_score_mean=archives["train"]["train_pca_mean_150"], train_score_std=archives["train"]["train_pca_std_150"],
    ).to(device)
    starts = ad_sequence_starts(archives["train"])
    if not starts:
        raise ValueError("V4 requires at least one QC-approved AD subject with at least three visits")
    first_last_val = first_last_pairs(val_all_pairs, archives["val"])
    batch_size = int(args.batch_size if args.batch_size is not None else v4_config["training"]["batch_size"])
    if dry_run:
        print("=" * 88, flush=True)
        print("PHASE 2/2 DRY RUN: frozen-V1 anchored AD-speed V4", flush=True)
        loader = DataLoader(V4PairDataset(train_ad_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_v4)
        with torch.no_grad():
            terms = loss_terms(model, train_values, next(iter(loader)), scales, v4_config["loss"], archives["train"], starts[0])
        print("V4 DRY RUN PASSED — no optimisation and no files written.", flush=True)
        print(json.dumps({name: float(value.detach().cpu()) for name, value in terms.items()}, indent=2), flush=True)
        return None

    assert_new((v4_dir,))
    v4_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(v4_path, v4_dir / "config_used.json")
    atomic_json(v4_dir / "feature_statistics.json", {"loss_scales": scales, "feature_source": "V1 train AD pairs only", "source_meshes_modified": False})
    np.savez_compressed(v4_dir / "feature_statistics.npz", **feature_stats)
    base_before = state_snapshot(model)
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [model.global_speed_logit], "lr": float(v4_config["training"]["learning_rate_global"]), "weight_decay": 0.0},
        {"params": head_params, "lr": float(v4_config["training"]["learning_rate_head"]), "weight_decay": float(v4_config["training"]["weight_decay"])},
    ])
    epochs = int(args.v4_epochs if args.v4_epochs is not None else v4_config["training"]["epochs"])
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("V4 epochs and batch-size must be positive")
    base_val = evaluate(model, val_values, val_all_pairs, batch_size)
    base_first_last = evaluate(model, val_values, first_last_val, batch_size)
    v1_checkpoint = v1_dir / "checkpoint_best.pt"
    # A mathematical V1 identity fallback is always saved.  It prevents a
    # failed calibration from silently replacing V1 with a worse model.
    identity_model = copy.deepcopy(model)
    identity_model.set_global_speed(1.0)
    identity_val = evaluate(identity_model, val_values, val_all_pairs, batch_size)
    identity_first_last = evaluate(identity_model, val_values, first_last_val, batch_size)
    identity_slope = ad_slope_error(identity_model, val_values, archives["val"])
    identity_metrics = {"val": identity_val, "val_first_last": identity_first_last, "val_ad_slope": identity_slope, "feasible": True, "global_ad_speed": 1.0, "checkpoint_role": "exact_v1_identity"}
    identity_metrics["volume_score"] = float(identity_val["groups"]["AD"]["rate_mean"]) + float(identity_slope["ad_subject_slope_abs_error"]) + float(v4_config["selection"]["volume_relative_error_weight"]) * float(identity_val["groups"]["AD"]["volume_relative_mean"])
    checkpoint_dir = v4_dir / "checkpoints"
    atomic_torch_save(checkpoint_dir / "v1_identity.pt", v4_checkpoint_payload(identity_model, optimizer, 0, identity_metrics, v1_checkpoint, v4_config))
    best_volume = identity_metrics["volume_score"]
    atomic_torch_save(checkpoint_dir / "best_feasible_volume.pt", v4_checkpoint_payload(identity_model, optimizer, 0, identity_metrics, v1_checkpoint, v4_config))
    history_path = v4_dir / "history.jsonl"
    selection = v4_config["selection"]
    training = v4_config["training"]
    stopped_early = False
    stale = 0
    started = time.time()
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, epochs + 1):
            model.train()
            for parameter in head_params:
                parameter.requires_grad_(epoch > int(training["warmup_global_epochs"]))
            counts: dict[str, float] = {}
            rows = 0
            subjects = list(starts)
            random.Random(seed + epoch * 100_003).shuffle(subjects)
            limit = int(training["sequence_subjects_per_epoch"])
            if limit > 0:
                subjects = subjects[:limit]
            subject_counts: dict[str, int] = {}
            for pair in train_ad_pairs:
                subject_counts[pair.subject] = subject_counts.get(pair.subject, 0) + 1
            generator = torch.Generator().manual_seed(seed + epoch * 10_007)
            sampler = WeightedRandomSampler(torch.tensor([1.0 / subject_counts[pair.subject] for pair in train_ad_pairs], dtype=torch.double), num_samples=int(training["samples_per_epoch"]), replacement=True, generator=generator)
            loader = DataLoader(V4PairDataset(train_ad_pairs), batch_size=batch_size, sampler=sampler, collate_fn=collate_v4, num_workers=args.num_workers)
            for step, raw in enumerate(loader):
                terms = loss_terms(model, train_values, raw, scales, v4_config["loss"], archives["train"], subjects[step % len(subjects)])
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                optimizer.step()
                batch_rows = int(raw["source"].shape[0])
                rows += batch_rows
                for name, value in terms.items():
                    counts[name] = counts.get(name, 0.0) + float(value.detach().cpu()) * batch_rows
            assert_frozen(model, base_before)
            model.eval()
            val = evaluate(model, val_values, val_all_pairs, batch_size)
            val_first = evaluate(model, val_values, first_last_val, batch_size)
            val_slope = ad_slope_error(model, val_values, archives["val"])
            is_feasible = feasible(val, base_val, val_first, base_first_last, selection)
            volume_score = float(val["groups"]["AD"]["rate_mean"]) + float(val_slope["ad_subject_slope_abs_error"]) + float(selection["volume_relative_error_weight"]) * float(val["groups"]["AD"]["volume_relative_mean"])
            record = {
                "epoch": epoch, "elapsed_minutes": (time.time() - started) / 60.0,
                "global_ad_speed": float(model.global_speed.detach().cpu()), "feasible": is_feasible,
                "val_volume_score": volume_score, "val_cn_max_abs_delta_from_v1": val["cn_max_abs_delta_from_v1"],
                "val_ad_pca": val["groups"]["AD"]["pca_mean"], "val_ad_vertex": val["groups"]["AD"]["vertex_mean"],
                "val_ad_volume_relative": val["groups"]["AD"]["volume_relative_mean"], "val_ad_rate": val["groups"]["AD"]["rate_mean"],
                "val_ad_subject_slope": val_slope["ad_subject_slope_abs_error"],
                **{f"train_{name}": total / max(rows, 1) for name, total in counts.items()},
            }
            history.write(json.dumps(record, sort_keys=True) + "\n")
            history.flush()
            metrics = {"val": val, "val_first_last": val_first, "val_ad_slope": val_slope, "feasible": is_feasible, "volume_score": volume_score, "global_ad_speed": record["global_ad_speed"]}
            payload = v4_checkpoint_payload(model, optimizer, epoch, metrics, v1_checkpoint, v4_config)
            atomic_torch_save(checkpoint_dir / "latest.pt", payload)
            if is_feasible and volume_score < best_volume - float(training["early_stopping_min_delta"]):
                best_volume, stale = volume_score, 0
                atomic_torch_save(checkpoint_dir / "best_feasible_volume.pt", payload)
            else:
                stale += 1
            print(f"V4 epoch {epoch:03d}/{epochs} train={record['train_total']:.6f} speed={record['global_ad_speed']:.4f} feasible={is_feasible} val_volume={volume_score:.6f} best={best_volume:.6f}", flush=True)
            if stale >= int(training["early_stopping_patience"]):
                stopped_early = True
                print(f"V4 early stopping after {stale} non-improving epochs.", flush=True)
                break
    chosen = torch.load(checkpoint_dir / "best_feasible_volume.pt", map_location=device)
    model.load_state_dict(chosen["model_state_dict"])
    model.eval()
    test = evaluate(model, test_values, test_all_pairs, batch_size)
    final = {
        "status": "complete", "structure": v4_config["structure"], "method": v4_config["method"],
        "v1_checkpoint": str(v1_checkpoint), "selected_v4_checkpoint": str(checkpoint_dir / "best_feasible_volume.pt"),
        "selected_epoch": int(chosen["epoch"]), "selected_global_ad_speed": float(chosen["metrics"]["global_ad_speed"]),
        "selected_is_exact_v1_identity": int(chosen["epoch"]) == 0, "v1_frozen_verified": True,
        "test": test, "source_meshes_modified": False, "pca_refitted": False,
        "epochs_requested": epochs, "epochs_completed": epoch, "stopped_early": stopped_early,
    }
    atomic_json(v4_dir / "final_report.json", final)
    print("=" * 88, flush=True)
    print(f"V4 complete. selected epoch={final['selected_epoch']}; output: {v4_dir}", flush=True)
    return final


def main() -> int:
    args = parse_args()
    require_single_name(args.run_name, "--run-name")
    v1_path, v1_config, v4_path, v4_config = load_checked_configs(args)
    v1_dir, v4_dir, pipeline_path = phase_directories(v1_path, args.run_name)
    print("=" * 88, flush=True)
    print(f"PCA V1 -> anchored AD-speed V4 | structure={args.structure} | phase={args.phase}", flush=True)
    print("Strict cohort: CN/AD only; MCI is rejected by the input validators.", flush=True)
    print("CN is exactly frozen to V1 in V4; V1 itself is frozen before V4 starts.", flush=True)
    if args.dry_run:
        run_v1(args, v1_path, dry_run=True)
        train_v4(args, v1_path, v1_config, v4_path, v4_config, v1_dir, v4_dir, dry_run=True)
        return 0
    if args.phase == "all":
        assert_new((v1_dir, v4_dir, pipeline_path))
        run_v1(args, v1_path, dry_run=False)
        final = train_v4(args, v1_path, v1_config, v4_path, v4_config, v1_dir, v4_dir, dry_run=False)
    elif args.phase == "v1":
        assert_new((v1_dir, pipeline_path))
        run_v1(args, v1_path, dry_run=False)
        final = None
    else:
        if not v1_dir.is_dir():
            raise FileNotFoundError(f"V4 cannot run before V1: {v1_dir}")
        assert_new((v4_dir, pipeline_path))
        final = train_v4(args, v1_path, v1_config, v4_path, v4_config, v1_dir, v4_dir, dry_run=False)
    if args.phase in {"all", "v4"}:
        atomic_json(pipeline_path, {
            "status": "complete", "structure": v4_config["structure"], "method": v4_config["method"],
            "v1_run": str(v1_dir), "v4_run": str(v4_dir), "v4_final_report": final,
            "source_meshes_modified": False, "strict_no_mci": True,
        })
        print(f"Pipeline complete: {pipeline_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
