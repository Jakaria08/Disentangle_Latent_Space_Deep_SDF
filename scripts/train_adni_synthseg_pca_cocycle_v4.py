#!/usr/bin/env python3
"""Train one independent SynthSeg PCA Cocycle-v4 model.

This is a PCA-space adaptation of the earlier ADNI v4 temporal-flow setup.
It keeps the important v4 directionality:

* real observed forward pair supervision (source -> target),
* real observed backward pair supervision (target -> source),
* forward and backward cocycle consistency, and
* exact identity at zero elapsed time.

The input bundle deliberately stores each observed pair once in chronological
order.  The reverse direction is derived from the same two observations at
training time; no duplicate or synthetic ``backward_pairs.csv`` is required.

The script is structure-selectable, never combines hippocampus with lateral
ventricle, refuses invalid/MCI-contaminated inputs, and refuses to overwrite a
previous training directory.  ``--dry-run`` only validates data and evaluates
one batch; it creates no output and performs no optimisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
STRUCTURES = ("hippocampus", "lateral_ventricle")
REQUIRED_PAIR_FIELDS = {
    "split", "diagnosis", "label_ad", "subject_id", "source_index", "target_index",
    "intermediate_index", "source_scan_id", "target_scan_id", "source_visit_order",
    "target_visit_order", "pair_type", "delta_years",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=STRUCTURES)
    parser.add_argument("--config", type=Path, default=None, help="Defaults to this structure's cocycle_v4_primary.json.")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, or cuda:N")
    parser.add_argument("--run-name", default="v4_real_pair_forward_backward", help="New directory below cocycle_v4/training/.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs; useful for a short pilot.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Override config random seed.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and one loss batch only; no training or files written.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def choose_device(requested: str) -> torch.device:
    request = requested.strip().lower()
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is not available.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class PairRow:
    source_index: int
    target_index: int
    intermediate_index: int
    pair_type: str
    diagnosis: str
    delta_years: float


class PairDataset(Dataset[PairRow]):
    def __init__(self, rows: list[PairRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> PairRow:
        return self.rows[index]


def collate_pairs(rows: list[PairRow]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.tensor([row.source_index for row in rows], dtype=torch.long),
        "target": torch.tensor([row.target_index for row in rows], dtype=torch.long),
        "intermediate": torch.tensor([row.intermediate_index for row in rows], dtype=torch.long),
        "is_adjacent": torch.tensor([row.pair_type == "adjacent" for row in rows], dtype=torch.bool),
        "is_ad": torch.tensor([row.diagnosis == "AD" for row in rows], dtype=torch.bool),
    }


class DirectAgeDiseaseTemporalFlow(nn.Module):
    """The prior v4 Euler temporal-flow form in standardized PCA space."""

    def __init__(self, latent_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        widths = [int(latent_dim) + 4, *[int(width) for width in hidden_dims], int(latent_dim)]
        if len(widths) < 3 or any(width <= 0 for width in widths):
            raise ValueError(f"Invalid model widths: {widths}")
        layers: list[nn.Module] = []
        for layer_index, (in_width, out_width) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(in_width, out_width)
            layers.append(linear)
            if layer_index < len(widths) - 2:
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)
        last_linear = next(module for module in reversed(self.net) if isinstance(module, nn.Linear))
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def velocity(self, z: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, diagnosis: torch.Tensor) -> torch.Tensor:
        source_time = source_time.reshape(-1, 1)
        target_time = target_time.reshape(-1, 1)
        diagnosis = diagnosis.reshape(-1, 1)
        delta = target_time - source_time
        return self.net(torch.cat([z, source_time, target_time, delta, diagnosis], dim=1))

    def transport(self, z: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor, diagnosis: torch.Tensor) -> torch.Tensor:
        delta = (target_time - source_time).reshape(-1, 1)
        return z + delta * self.velocity(z, source_time, target_time, diagnosis)


def default_config_path(structure: str) -> Path:
    return BASE_ROOT / f"{structure}_pca_cocycle_v4" / "cocycle_v4" / "configs" / "cocycle_v4_primary.json"


def load_archive(path: Path, split: str, components: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} sequence archive: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    required = {
        "subject_ids", "subject_splits", "subject_diagnoses", "subject_label_ad", "subject_visit_offsets",
        "visit_scan_ids", "visit_subject_ids", "visit_splits", "visit_diagnoses", "visit_label_ad",
        "visit_orders", "visit_time_years_from_baseline", "visit_age_norm_train",
        "visit_pca_150", "visit_pca_standardized_150", "visit_volume_mm3",
        "train_pca_mean_150", "train_pca_std_150",
    }
    missing = sorted(required.difference(archive))
    if missing:
        raise KeyError(f"{path} is missing fields: {missing}")
    if any(value.dtype == object for value in archive.values()):
        raise ValueError(f"Pickle-dependent array found in {path}")
    visits = len(archive["visit_scan_ids"])
    subjects = len(archive["subject_ids"])
    if archive["visit_pca_150"].shape != (visits, components):
        raise ValueError(f"Unexpected raw PCA shape in {path}: {archive['visit_pca_150'].shape}")
    if archive["visit_pca_standardized_150"].shape != (visits, components):
        raise ValueError(f"Unexpected standardized PCA shape in {path}: {archive['visit_pca_standardized_150'].shape}")
    if not np.isfinite(archive["visit_pca_150"]).all() or not np.isfinite(archive["visit_pca_standardized_150"]).all():
        raise ValueError(f"Non-finite PCA score in {path}")
    if set(archive["visit_splits"].astype(str)) != {split}:
        raise ValueError(f"Split contamination in {path}")
    if set(archive["visit_diagnoses"].astype(str)).difference({"CN", "AD"}):
        raise ValueError(f"Only strict CN/AD visits are permitted in {path}")
    labels = archive["visit_label_ad"].astype(np.int64)
    diagnosis_is_ad = archive["visit_diagnoses"].astype(str) == "AD"
    if not np.array_equal(labels, diagnosis_is_ad.astype(np.int64)):
        raise ValueError(f"Diagnosis label mismatch in {path}")
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    if offsets.shape != (subjects + 1,) or offsets[0] != 0 or offsets[-1] != visits:
        raise ValueError(f"Invalid subject offsets in {path}")
    for subject_index in range(subjects):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        if end - start < 2:
            raise ValueError(f"Subject {subject_index} has fewer than two visits in {path}")
        if np.any(np.diff(archive["visit_time_years_from_baseline"][start:end]) <= 0):
            raise ValueError(f"Non-increasing follow-up time in {path}")
        if np.unique(archive["visit_diagnoses"][start:end].astype(str)).size != 1:
            raise ValueError(f"Diagnosis changes within a subject in {path}")
    return archive


def load_pairs(path: Path, archive: dict[str, np.ndarray], split: str) -> list[PairRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} pair file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_PAIR_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Pair schema mismatch in {path}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"No pairs in {path}")
    visits = len(archive["visit_scan_ids"])
    subject_ids = archive["visit_subject_ids"].astype(str)
    scans = archive["visit_scan_ids"].astype(str)
    diagnoses = archive["visit_diagnoses"].astype(str)
    labels = archive["visit_label_ad"].astype(np.int64)
    orders = archive["visit_orders"].astype(np.int64)
    age = archive["visit_age_norm_train"].astype(np.float64)
    rows: list[PairRow] = []
    seen: set[tuple[int, int]] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        try:
            source = int(row["source_index"])
            target = int(row["target_index"])
            intermediate = int(row["intermediate_index"])
            label = int(row["label_ad"])
            delta_years = float(row["delta_years"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bad numeric value in {path}:{row_number}") from exc
        if not (0 <= source < visits and 0 <= target < visits and (intermediate == -1 or 0 <= intermediate < visits)):
            raise ValueError(f"Out-of-range pair index in {path}:{row_number}")
        if row["split"] != split or row["pair_type"] not in {"adjacent", "nonadjacent"}:
            raise ValueError(f"Invalid split or pair type in {path}:{row_number}")
        if not (subject_ids[source] == subject_ids[target] == row["subject_id"]):
            raise ValueError(f"Cross-subject pair in {path}:{row_number}")
        if not (scans[source] == row["source_scan_id"] and scans[target] == row["target_scan_id"]):
            raise ValueError(f"Scan identifier mismatch in {path}:{row_number}")
        if not (diagnoses[source] == diagnoses[target] == row["diagnosis"] and labels[source] == labels[target] == label):
            raise ValueError(f"Diagnosis mismatch in {path}:{row_number}")
        if not (orders[target] > orders[source] and age[target] > age[source] and delta_years > 0.0):
            raise ValueError(f"Pair is not strictly forward in {path}:{row_number}")
        if intermediate >= 0 and not (subject_ids[intermediate] == subject_ids[source] and orders[source] < orders[intermediate] < orders[target]):
            raise ValueError(f"Invalid intermediate visit in {path}:{row_number}")
        key = (source, target)
        if key in seen:
            raise ValueError(f"Duplicate pair in {path}:{row_number}")
        seen.add(key)
        rows.append(PairRow(source, target, intermediate, row["pair_type"], row["diagnosis"], delta_years))
    return rows


def validate_pca_model(config: dict[str, Any], components: int) -> dict[str, np.ndarray]:
    model_dir = Path(config["representation"]["pca_model_dir"])
    filenames = {"mean": "mean.npy", "components": "components_150.npy", "faces": "faces.npy"}
    arrays: dict[str, np.ndarray] = {}
    for key, filename in filenames.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing PCA model artifact: {path}")
        arrays[key] = np.load(path, allow_pickle=False)
    if arrays["components"].shape[0] != components or arrays["mean"].ndim != 1:
        raise ValueError("PCA model does not have the required PCA-150 shape")
    if arrays["components"].shape[1] != arrays["mean"].shape[0] or arrays["mean"].size % 3 != 0:
        raise ValueError("PCA model vertex shape mismatch")
    if arrays["faces"].ndim != 2 or arrays["faces"].shape[1] != 3:
        raise ValueError("PCA model face array is invalid")
    return arrays


def make_tensors(archive: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "z": torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device),
        "time": torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device),
        "diagnosis": torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device),
    }


def batch_terms(
    model: DirectAgeDiseaseTemporalFlow,
    values: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    source = batch["source"].to(values["z"].device)
    target = batch["target"].to(values["z"].device)
    z_source, z_target = values["z"][source], values["z"][target]
    t_source, t_target = values["time"][source], values["time"][target]
    diagnosis = values["diagnosis"][source]

    predicted_target = model.transport(z_source, t_source, t_target, diagnosis)
    predicted_source = model.transport(z_target, t_target, t_source, diagnosis)
    forward_target = torch.mean((predicted_target - z_target) ** 2)
    backward_target = torch.mean((predicted_source - z_source) ** 2)

    cycle_source = model.transport(predicted_target, t_target, t_source, diagnosis)
    cycle_target = model.transport(predicted_source, t_source, t_target, diagnosis)
    forward_cocycle = torch.mean((cycle_source - z_source) ** 2)
    backward_cocycle = torch.mean((cycle_target - z_target) ** 2)

    # The flow form z + (t-s)*v makes this exactly zero, preserving the v4
    # zero-displacement condition by construction rather than by a noisy penalty.
    identity = model.transport(z_source, t_source, t_source, diagnosis)
    zero_displacement = torch.mean((identity - z_source) ** 2)

    velocity = model.velocity(z_source, t_source, t_target, diagnosis)
    speed_guard = torch.mean(velocity ** 2)
    total = (
        weights["real_pair_forward_weight"] * forward_target
        + weights["real_pair_backward_weight"] * backward_target
        + weights["cocycle_forward_weight"] * forward_cocycle
        + weights["cocycle_backward_weight"] * backward_cocycle
        + weights["zero_displacement_weight"] * zero_displacement
        + weights["speed_guard_weight"] * speed_guard
    )
    return {
        "total": total,
        "forward_target": forward_target,
        "backward_target": backward_target,
        "forward_cocycle": forward_cocycle,
        "backward_cocycle": backward_cocycle,
        "zero_displacement": zero_displacement,
        "speed_guard": speed_guard,
    }


def mean_terms(model: DirectAgeDiseaseTemporalFlow, values: dict[str, torch.Tensor], loader: DataLoader, weights: dict[str, float]) -> dict[str, float]:
    sums: dict[str, float] = {}
    batches = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            terms = batch_terms(model, values, batch, weights)
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
    if batches == 0:
        raise RuntimeError("Evaluation loader had no batches")
    return {key: value / batches for key, value in sums.items()}


def mesh_volumes(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[:, faces[:, 0]], vertices[:, faces[:, 1]], vertices[:, faces[:, 2]]
    return np.abs(np.einsum("bfi,bfi->bf", v0, np.cross(v1, v2)).sum(axis=1) / 6.0)


def decoded_metrics(
    model: DirectAgeDiseaseTemporalFlow,
    archive: dict[str, np.ndarray],
    pairs: list[PairRow],
    pca_model: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Decode only at final evaluation, never inside the optimisation loop."""
    values = make_tensors(archive, device)
    components = pca_model["components"].astype(np.float32)
    mean = pca_model["mean"].astype(np.float32)
    faces = pca_model["faces"].astype(np.int64)
    pca_mean = archive["train_pca_mean_150"].astype(np.float32)
    pca_std = archive["train_pca_std_150"].astype(np.float32)
    output: dict[str, list[float]] = {
        "vertex_mae_mm": [], "volume_relative_error_pct": [],
        "standardized_pca_mse": [], "delta_years": [], "diagnosis": [],
        "predicted_volume_change_mm3_per_year": [], "observed_volume_change_mm3_per_year": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            source_index = torch.tensor([row.source_index for row in chunk], dtype=torch.long, device=device)
            target_index = torch.tensor([row.target_index for row in chunk], dtype=torch.long, device=device)
            predicted = model.transport(
                values["z"][source_index], values["time"][source_index], values["time"][target_index], values["diagnosis"][source_index]
            ).cpu().numpy()
            target_standardized = archive["visit_pca_standardized_150"][np.asarray([row.target_index for row in chunk])]
            predicted_raw = predicted * pca_std[None, :] + pca_mean[None, :]
            target_raw = archive["visit_pca_150"][np.asarray([row.target_index for row in chunk])]
            source_raw = archive["visit_pca_150"][np.asarray([row.source_index for row in chunk])]
            predicted_vertices = (mean[None, :] + predicted_raw @ components).reshape(len(chunk), -1, 3)
            target_vertices = (mean[None, :] + target_raw @ components).reshape(len(chunk), -1, 3)
            source_vertices = (mean[None, :] + source_raw @ components).reshape(len(chunk), -1, 3)
            vertex_mae = np.mean(np.abs(predicted_vertices - target_vertices), axis=(1, 2))
            target_volume = mesh_volumes(target_vertices, faces)
            predicted_volume = mesh_volumes(predicted_vertices, faces)
            source_volume = mesh_volumes(source_vertices, faces)
            volume_error = 100.0 * np.abs(predicted_volume - target_volume) / np.maximum(target_volume, 1.0e-8)
            delta_years = np.asarray([row.delta_years for row in chunk], dtype=np.float64)
            output["vertex_mae_mm"].extend(vertex_mae.astype(float))
            output["volume_relative_error_pct"].extend(volume_error.astype(float))
            output["standardized_pca_mse"].extend(np.mean((predicted - target_standardized) ** 2, axis=1).astype(float))
            output["delta_years"].extend(delta_years.astype(float))
            output["diagnosis"].extend(row.diagnosis for row in chunk)
            output["predicted_volume_change_mm3_per_year"].extend(((predicted_volume - source_volume) / delta_years).astype(float))
            output["observed_volume_change_mm3_per_year"].extend(((target_volume - source_volume) / delta_years).astype(float))
    return {
        "pairs": len(pairs),
        "standardized_pca_mse_mean": float(np.mean(output["standardized_pca_mse"])),
        "decoded_vertex_mae_mm_mean": float(np.mean(output["vertex_mae_mm"])),
        "decoded_vertex_mae_mm_p95": float(np.quantile(output["vertex_mae_mm"], 0.95)),
        "decoded_volume_relative_error_pct_mean": float(np.mean(output["volume_relative_error_pct"])),
        "decoded_volume_relative_error_pct_p95": float(np.quantile(output["volume_relative_error_pct"], 0.95)),
        "by_gap": {
            "short_le_1y": grouped_metric(output, lambda gap: gap <= 1.0),
            "long_gt_1y": grouped_metric(output, lambda gap: gap > 1.0),
        },
        "volume_speed_mm3_per_year": diagnosis_speed_summary(output),
    }


def grouped_metric(values: dict[str, list[float]], select: Any) -> dict[str, Any]:
    indices = [index for index, gap in enumerate(values["delta_years"]) if select(gap)]
    if not indices:
        return {"pairs": 0, "standardized_pca_mse_mean": None, "decoded_vertex_mae_mm_mean": None, "decoded_volume_relative_error_pct_mean": None}
    return {
        "pairs": len(indices),
        "standardized_pca_mse_mean": float(np.mean([values["standardized_pca_mse"][index] for index in indices])),
        "decoded_vertex_mae_mm_mean": float(np.mean([values["vertex_mae_mm"][index] for index in indices])),
        "decoded_volume_relative_error_pct_mean": float(np.mean([values["volume_relative_error_pct"][index] for index in indices])),
    }


def diagnosis_speed_summary(values: dict[str, list[Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for diagnosis in ("CN", "AD"):
        indices = [index for index, label in enumerate(values["diagnosis"]) if label == diagnosis]
        result[diagnosis] = {
            "pairs": len(indices),
            "predicted_volume_change_mm3_per_year_mean": (
                float(np.mean([values["predicted_volume_change_mm3_per_year"][index] for index in indices])) if indices else None
            ),
            "observed_volume_change_mm3_per_year_mean": (
                float(np.mean([values["observed_volume_change_mm3_per_year"][index] for index in indices])) if indices else None
            ),
        }
    if result["CN"]["pairs"] and result["AD"]["pairs"]:
        result["AD_minus_CN_predicted_volume_change_mm3_per_year"] = (
            result["AD"]["predicted_volume_change_mm3_per_year_mean"]
            - result["CN"]["predicted_volume_change_mm3_per_year_mean"]
        )
        result["AD_minus_CN_observed_volume_change_mm3_per_year"] = (
            result["AD"]["observed_volume_change_mm3_per_year_mean"]
            - result["CN"]["observed_volume_change_mm3_per_year_mean"]
        )
    return result


def pair_sampler(rows: list[PairRow], seed: int) -> WeightedRandomSampler:
    counts = {kind: sum(row.pair_type == kind for row in rows) for kind in ("adjacent", "nonadjacent")}
    if not all(counts.values()):
        raise ValueError(f"Both adjacent and nonadjacent pairs are required for prior-v4 mixed sampling: {counts}")
    # Equivalent to the previous mixed_adjacent_far setting with 0.5 adjacent
    # probability, while retaining every QC-approved observed pair as eligible.
    weights = torch.tensor([0.5 / counts[row.pair_type] for row in rows], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True, generator=generator)


def structure_report(rows: list[PairRow]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "adjacent": sum(row.pair_type == "adjacent" for row in rows),
        "nonadjacent": sum(row.pair_type == "nonadjacent" for row in rows),
        "CN": sum(row.diagnosis == "CN" for row in rows),
        "AD": sum(row.diagnosis == "AD" for row in rows),
        "min_delta_years": float(min(row.delta_years for row in rows)),
        "max_delta_years": float(max(row.delta_years for row in rows)),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config or default_config_path(args.structure)
    config = read_json(config_path)
    if config.get("structure") not in {"left_hippocampus", "left_lateral_ventricle"}:
        raise ValueError(f"Not a recognized separate-structure config: {config_path}")
    if args.structure not in str(config.get("name", "")):
        raise ValueError("--structure does not match config name")
    if config.get("model", {}).get("legacy_checkpoint_used") is not False:
        raise ValueError("This PCA Cocycle-v4 runner must not use a legacy checkpoint")
    representation = config["representation"]
    components = int(representation["components"])
    if components != 150:
        raise ValueError(f"Expected PCA-150, got PCA-{components}")
    loss_config = config["loss"]
    required_weights = {
        "real_pair_forward_weight", "real_pair_backward_weight", "cocycle_forward_weight",
        "cocycle_backward_weight", "zero_displacement_weight", "speed_guard_weight",
    }
    missing_weights = sorted(required_weights.difference(loss_config))
    if missing_weights:
        raise KeyError(f"Config is not the prior-v4-compatible loss contract; missing {missing_weights}")
    weights = {key: float(loss_config[key]) for key in required_weights}
    if weights["real_pair_forward_weight"] <= 0.0 or weights["real_pair_backward_weight"] <= 0.0:
        raise ValueError("Both prior-v4 real pair directions must have positive weights")
    dataset_config = config["dataset"]
    archives = {split: load_archive(Path(dataset_config[f"{split}_sequences"]), split, components) for split in ("train", "val", "test")}
    pairs = {split: load_pairs(Path(dataset_config[f"{split}_pairs"]), archives[split], split) for split in ("train", "val", "test")}
    pca_model = validate_pca_model(config, components)
    device = choose_device(args.device)
    training = config["training"]
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else training["batch_size"])
    seed = int(args.seed if args.seed is not None else training["seed"])
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    set_seed(seed)
    model = DirectAgeDiseaseTemporalFlow(
        latent_dim=components,
        hidden_dims=config["model"]["hidden_dims"],
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    tensors = {split: make_tensors(archive, device) for split, archive in archives.items()}
    train_dataset = PairDataset(pairs["train"])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=pair_sampler(pairs["train"], seed),
        num_workers=args.num_workers, collate_fn=collate_pairs, pin_memory=device.type == "cuda",
    )
    eval_loaders = {
        split: DataLoader(PairDataset(pairs[split]), batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                          collate_fn=collate_pairs, pin_memory=device.type == "cuda")
        for split in ("val", "test")
    }

    print("=" * 88)
    print(f"PCA Cocycle-v4 | {args.structure} | device={device} | PCA-{components}")
    print("Prior-v4 directions: real forward=yes, real backward=yes, cocycle forward=yes, cocycle backward=yes")
    for split in ("train", "val", "test"):
        print(f"  {split}: {structure_report(pairs[split])}")
    print(f"  config: {config_path}", flush=True)
    if args.dry_run:
        first_batch = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            terms = batch_terms(model, tensors["train"], first_batch, weights)
        print("DRY RUN PASSED — no optimisation and no files written.")
        print(json.dumps({key: float(value.cpu()) for key, value in terms.items()}, indent=2), flush=True)
        return 0

    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be a single new directory name")
    output_dir = config_path.parents[1] / "training" / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite existing training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config_used.json")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    best_val = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_path = output_dir / "history.jsonl"
    validation_frequency = int(training["validation_frequency"])
    patience = int(training["early_stopping_patience"])
    start_time = time.time()
    run_contract = {
        "status": "training_started",
        "structure": config["structure"],
        "config": str(config_path),
        "source_meshes_modified": False,
        "pca_refitted": False,
        "legacy_checkpoint_used": False,
        "backward_pair_file": "not required; reverse transport derives from the observed forward rows",
        "prior_v4_losses": config.get("prior_v4_losses"),
        "device": str(device),
        "seed": seed,
        "epochs_requested": epochs,
    }
    atomic_json(output_dir / "run_contract.json", run_contract)
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, epochs + 1):
            model.train()
            totals: dict[str, float] = {}
            batches = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                terms = batch_terms(model, tensors["train"], batch, weights)
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                optimizer.step()
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                batches += 1
            train_means = {f"train_{key}": value / batches for key, value in totals.items()}
            record: dict[str, Any] = {"epoch": epoch, "elapsed_minutes": (time.time() - start_time) / 60.0, **train_means}
            validation_due = epoch == 1 or epoch % validation_frequency == 0 or epoch == epochs
            if validation_due:
                val_means = mean_terms(model, tensors["val"], eval_loaders["val"], weights)
                record.update({f"val_{key}": value for key, value in val_means.items()})
                monitored = val_means["forward_target"] + val_means["backward_target"]
                record["monitor_real_pair_bidir_mse"] = monitored
                if monitored < best_val:
                    best_val, best_epoch, epochs_without_improvement = monitored, epoch, 0
                    atomic_torch_save(output_dir / "checkpoint_best.pt", {
                        "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_real_pair_bidir_mse": best_val, "config": config, "structure": args.structure,
                    })
                else:
                    epochs_without_improvement += validation_frequency
            history.write(json.dumps(record, sort_keys=True) + "\n")
            history.flush()
            print(
                f"epoch {epoch:03d}/{epochs} train={record['train_total']:.6f} "
                + (f"val_bidir={record['monitor_real_pair_bidir_mse']:.6f} best={best_val:.6f}" if validation_due else ""),
                flush=True,
            )
            if validation_due and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; no validation improvement for {epochs_without_improvement} epochs.", flush=True)
                break
    checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_means = mean_terms(model, tensors["test"], eval_loaders["test"], weights)
    val_decoded = decoded_metrics(model, archives["val"], pairs["val"], pca_model, device, batch_size)
    test_decoded = decoded_metrics(model, archives["test"], pairs["test"], pca_model, device, batch_size)
    final = {
        "status": "complete",
        "structure": config["structure"],
        "best_epoch": best_epoch,
        "best_val_real_pair_bidir_mse": best_val,
        "epochs_completed": epoch,
        "elapsed_minutes": (time.time() - start_time) / 60.0,
        "test_losses": test_means,
        "decoded_metrics": {"val": val_decoded, "test": test_decoded},
        "source_meshes_modified": False,
        "pca_refitted": False,
        "brainode_trained": False,
        "backward_transport_used": True,
    }
    atomic_json(output_dir / "final_report.json", final)
    run_contract["status"] = "complete"
    run_contract["best_epoch"] = best_epoch
    atomic_json(output_dir / "run_contract.json", run_contract)
    print("=" * 88)
    print(f"Training complete. Best epoch {best_epoch}; output: {output_dir}")
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
