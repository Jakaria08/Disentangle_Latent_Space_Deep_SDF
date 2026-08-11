#!/usr/bin/env python3
"""Independent read-only validation for the strict SynthSeg longitudinal models.

The validator never changes a checkpoint or source artifact.  It verifies
cohort/checkpoint contracts and evaluates all methods with one PCA decoder and
one metric implementation.  E3's selected checkpoint is primary; its latest
checkpoint is reported only as an explicitly exploratory sensitivity result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_adni_synthseg_pca_cocycle_v4 import (
    BASE_ROOT,
    DirectAgeDiseaseTemporalFlow,
    PairRow,
    choose_device,
    load_archive,
    load_pairs,
    read_json,
    validate_pca_model,
)
from train_adni_synthseg_pca_v1_anchored_speed_v4 import V1AnchoredAdSpeedCalibrator


STRUCTURE_RUNS = {
    "hippocampus": {
        "label": "Left hippocampus",
        "folder": "hippocampus_pca_cocycle_v4",
        "v1": "hippocampus_v1_anchor_v4_s42_v1",
        "v4": "hippocampus_v1_anchor_v4_s42_v4",
        "e3": "hippocampus_unified_volume_e3_s42",
    },
    "lateral_ventricle": {
        "label": "Left lateral ventricle",
        "folder": "lateral_ventricle_pca_cocycle_v4",
        "v1": "lv_v1_anchor_v4_s42_v1",
        "v4": "lv_v1_anchor_v4_s42_v4",
        "e3": "lv_unified_volume_e3_s42",
    },
}

METHOD_ORDER = ["no_change", "v1", "anchored_v4", "e3_selected", "e3_latest_sensitivity"]
METHOD_LABELS = {
    "no_change": "No change",
    "v1": "V1",
    "anchored_v4": "Anchored V4",
    "e3_selected": "E3 selected",
    "e3_latest_sensitivity": "E3 latest (exploratory)",
}
PRIMARY_METHODS = {"no_change", "v1", "anchored_v4", "e3_selected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_ROOT / "longitudinal_model_validation" / "current_models_s42",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def top_change_dice(left: np.ndarray, right: np.ndarray, fraction: float = 0.2) -> float:
    count = max(1, int(round(left.size * fraction)))
    left_ids = set(np.argpartition(np.abs(left), -count)[-count:].tolist())
    right_ids = set(np.argpartition(np.abs(right), -count)[-count:].tolist())
    return float(2.0 * len(left_ids & right_ids) / max(len(left_ids) + len(right_ids), 1))


class Geometry:
    def __init__(
        self,
        pca_model: dict[str, np.ndarray],
        score_mean: np.ndarray,
        score_std: np.ndarray,
    ) -> None:
        self.mean = pca_model["mean"].astype(np.float64)
        self.components = pca_model["components"].astype(np.float64)
        self.faces = pca_model["faces"].astype(np.int64)
        self.score_mean = score_mean.astype(np.float64)
        self.score_std = np.maximum(score_std.astype(np.float64), 1.0e-6)

    def vertices(self, standardized: np.ndarray) -> np.ndarray:
        standardized = np.asarray(standardized, dtype=np.float64)
        raw = standardized * self.score_std + self.score_mean
        return (raw @ self.components + self.mean).reshape(standardized.shape[0], -1, 3)

    def volume(self, vertices: np.ndarray) -> np.ndarray:
        v0 = vertices[:, self.faces[:, 0], :]
        v1 = vertices[:, self.faces[:, 1], :]
        v2 = vertices[:, self.faces[:, 2], :]
        signed = np.einsum("bfi,bfi->bf", v0, np.cross(v1, v2)).sum(axis=1) / 6.0
        return np.maximum(np.abs(signed), 1.0e-8)

    def vertex_normals(self, vertices: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(vertices, dtype=np.float64)
        v0 = vertices[self.faces[:, 0]]
        v1 = vertices[self.faces[:, 1]]
        v2 = vertices[self.faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        for corner in range(3):
            np.add.at(normals, self.faces[:, corner], face_normals)
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)


class Adapter:
    def __init__(self, name: str, model: torch.nn.Module | None, device: torch.device, checkpoint: Path | None) -> None:
        self.name = name
        self.model = model
        self.device = device
        self.checkpoint = checkpoint
        if self.model is not None:
            self.model.eval()

    @torch.no_grad()
    def transport(
        self,
        z: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        source_years: torch.Tensor,
        target_years: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            return z
        if isinstance(self.model, V1AnchoredAdSpeedCalibrator):
            prediction, _ = self.model.transport(z, source_time, target_time, source_years, target_years, label)
            return prediction
        return self.model.transport(z, source_time, target_time, label)


def finite_state_dict(state: dict[str, torch.Tensor], label: str) -> None:
    if not state:
        raise ValueError(f"Empty checkpoint state: {label}")
    for name, value in state.items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor {label}:{name}")


def load_direct(
    checkpoint: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[DirectAgeDiseaseTemporalFlow, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model_state_dict"]
    finite_state_dict(state, str(checkpoint))
    model = DirectAgeDiseaseTemporalFlow(
        latent_dim=150,
        hidden_dims=config["model"]["hidden_dims"],
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload


def validate_cohorts(archives: dict[str, dict[str, np.ndarray]], structure: str) -> dict[str, Any]:
    subject_sets: dict[str, set[str]] = {}
    scan_sets: dict[str, set[str]] = {}
    split_report: dict[str, Any] = {}
    for split, archive in archives.items():
        subjects = archive["subject_ids"].astype(str)
        scans = archive["visit_scan_ids"].astype(str)
        diagnoses = archive["visit_diagnoses"].astype(str)
        labels = archive["visit_label_ad"].astype(np.int64)
        if set(diagnoses) - {"CN", "AD"}:
            raise ValueError(f"MCI/nonstable diagnosis in {structure}/{split}")
        if not np.array_equal(labels, (diagnoses == "AD").astype(np.int64)):
            raise ValueError(f"Diagnosis-label mismatch in {structure}/{split}")
        if len(set(subjects)) != len(subjects) or len(set(scans)) != len(scans):
            raise ValueError(f"Duplicate subject or scan in {structure}/{split}")
        subject_sets[split] = set(subjects)
        scan_sets[split] = set(scans)
        split_report[split] = {
            "subjects": len(subjects),
            "visits": len(scans),
            "CN_visits": int(np.sum(diagnoses == "CN")),
            "AD_visits": int(np.sum(diagnoses == "AD")),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if subject_sets[left] & subject_sets[right] or scan_sets[left] & scan_sets[right]:
            raise ValueError(f"Split leakage in {structure}: {left}/{right}")
    return {"status": "pass", "splits": split_report, "strict_no_mci": True, "subject_and_scan_disjoint": True}


def load_structure(
    structure: str,
    specification: dict[str, str],
    device: torch.device,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, list[PairRow]],
    Geometry,
    dict[str, Adapter],
    dict[str, Any],
]:
    root = BASE_ROOT / specification["folder"] / "cocycle_v4"
    training_root = root / "training"
    v1_dir = training_root / specification["v1"]
    v4_dir = training_root / specification["v4"]
    e3_dir = training_root / specification["e3"]
    for path in (v1_dir, v4_dir, e3_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    v1_config = read_json(v1_dir / "config_used.json")
    e3_config = read_json(e3_dir / "config_used.json")
    v4_config = read_json(v4_dir / "config_used.json")
    archives = {
        split: load_archive(Path(v1_config["dataset"][f"{split}_sequences"]), split, 150)
        for split in ("train", "val", "test")
    }
    pairs = {
        split: load_pairs(Path(v1_config["dataset"][f"{split}_pairs"]), archives[split], split)
        for split in ("train", "val", "test")
    }
    cohort_report = validate_cohorts(archives, structure)
    pca_model = validate_pca_model(v1_config, 150)
    geometry = Geometry(
        pca_model,
        archives["train"]["train_pca_mean_150"],
        archives["train"]["train_pca_std_150"],
    )

    v1_checkpoint = v1_dir / "checkpoint_best.pt"
    v1_model, v1_payload = load_direct(v1_checkpoint, v1_config, device)
    with np.load(v4_dir / "feature_statistics.npz", allow_pickle=False) as loaded:
        feature_stats = {name: loaded[name] for name in loaded.files}
    v4_base, _ = load_direct(v1_checkpoint, v1_config, device)
    v4_model = V1AnchoredAdSpeedCalibrator(
        base_flow=v4_base,
        model_config=v4_config["model"],
        feature_stats=feature_stats,
        pca_model=pca_model,
        train_score_mean=archives["train"]["train_pca_mean_150"],
        train_score_std=archives["train"]["train_pca_std_150"],
    ).to(device)
    v4_checkpoint = v4_dir / "checkpoints" / "best_feasible_volume.pt"
    v4_payload = torch.load(v4_checkpoint, map_location=device)
    finite_state_dict(v4_payload["model_state_dict"], str(v4_checkpoint))
    v4_model.load_state_dict(v4_payload["model_state_dict"])
    v4_model.eval()

    # Verify that the anchored checkpoint contains the exact selected V1.
    for name, tensor in v1_payload["model_state_dict"].items():
        anchored_name = f"base_flow.{name}"
        anchored = v4_payload["model_state_dict"].get(anchored_name)
        if anchored is None or not torch.equal(tensor.detach().cpu(), anchored.detach().cpu()):
            raise ValueError(f"Frozen V1 mismatch in {structure}: {name}")

    e3_selected_checkpoint = e3_dir / "checkpoints" / "best_feasible_composite.pt"
    e3_latest_checkpoint = e3_dir / "checkpoints" / "latest.pt"
    e3_selected_model, e3_selected_payload = load_direct(e3_selected_checkpoint, e3_config, device)
    e3_latest_model, e3_latest_payload = load_direct(e3_latest_checkpoint, e3_config, device)
    if e3_config["model_contract"].get("trainable_models") != 1 or e3_config["model_contract"].get("secondary_speed_head") is not False:
        raise ValueError(f"E3 one-model contract failed for {structure}")

    adapters = {
        "no_change": Adapter("no_change", None, device, None),
        "v1": Adapter("v1", v1_model, device, v1_checkpoint),
        "anchored_v4": Adapter("anchored_v4", v4_model, device, v4_checkpoint),
        "e3_selected": Adapter("e3_selected", e3_selected_model, device, e3_selected_checkpoint),
        "e3_latest_sensitivity": Adapter("e3_latest_sensitivity", e3_latest_model, device, e3_latest_checkpoint),
    }
    checkpoint_report = {
        "status": "pass",
        "frozen_v1_exact_match": True,
        "e3_single_model_contract": True,
        "checkpoints": {
            name: ({"path": str(adapter.checkpoint), "sha256": file_sha256(adapter.checkpoint)} if adapter.checkpoint else {"path": None})
            for name, adapter in adapters.items()
        },
        "epochs": {
            "v1": int(v1_payload["epoch"]),
            "anchored_v4": int(v4_payload["epoch"]),
            "e3_selected": int(e3_selected_payload["epoch"]),
            "e3_latest_sensitivity": int(e3_latest_payload["epoch"]),
        },
        "e3_latest_is_exploratory_not_primary": True,
    }
    return archives, pairs, geometry, adapters, {"cohort": cohort_report, "checkpoints": checkpoint_report}


def pair_metadata(archive: dict[str, np.ndarray], pair: PairRow, split: str) -> dict[str, Any]:
    source, target = pair.source_index, pair.target_index
    return {
        "split": split,
        "diagnosis": pair.diagnosis,
        "subject_id": str(archive["visit_subject_ids"][source]),
        "source_scan_id": str(archive["visit_scan_ids"][source]),
        "target_scan_id": str(archive["visit_scan_ids"][target]),
        "source_visit_order": int(archive["visit_orders"][source]),
        "target_visit_order": int(archive["visit_orders"][target]),
        "pair_type": pair.pair_type,
        "gap_bin": "adjacent" if pair.pair_type == "adjacent" else ("short" if pair.delta_years <= 2.0 else "long"),
        "gap_years": float(pair.delta_years),
    }


def evaluate_method_pairs(
    structure: str,
    method: str,
    adapter: Adapter,
    archive: dict[str, np.ndarray],
    pairs: list[PairRow],
    geometry: Geometry,
    split: str,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    z_all = torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device)
    time_all = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
    years_all = torch.from_numpy(archive["visit_age_years"].astype(np.float32)).to(device)
    label_all = torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device)
    raw_volume = archive["visit_volume_mm3"].astype(np.float64)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(pairs), batch_size):
        current = pairs[start : start + batch_size]
        source_index = torch.tensor([pair.source_index for pair in current], dtype=torch.long, device=device)
        target_index = torch.tensor([pair.target_index for pair in current], dtype=torch.long, device=device)
        source = z_all[source_index]
        target = z_all[target_index]
        source_time = time_all[source_index]
        target_time = time_all[target_index]
        source_years = years_all[source_index]
        target_years = years_all[target_index]
        label = label_all[source_index]
        predicted = adapter.transport(source, source_time, target_time, source_years, target_years, label)
        backward = adapter.transport(target, target_time, source_time, target_years, source_years, label)
        cycle = adapter.transport(predicted, target_time, source_time, target_years, source_years, label)
        composition = torch.full((len(current),), float("nan"), device=device)
        valid_positions = [position for position, pair in enumerate(current) if pair.intermediate_index >= 0]
        if valid_positions:
            positions = torch.tensor(valid_positions, dtype=torch.long, device=device)
            middle_index = torch.tensor([current[position].intermediate_index for position in valid_positions], dtype=torch.long, device=device)
            middle_time = time_all[middle_index]
            middle_years = years_all[middle_index]
            predicted_middle = adapter.transport(
                source[positions], source_time[positions], middle_time, source_years[positions], middle_years, label[positions]
            )
            composed = adapter.transport(
                predicted_middle, middle_time, target_time[positions], middle_years, target_years[positions], label[positions]
            )
            composition[positions] = torch.mean((composed - predicted[positions]) ** 2, dim=1)
        source_np = source.cpu().numpy()
        target_np = target.cpu().numpy()
        predicted_np = predicted.cpu().numpy()
        backward_np = backward.cpu().numpy()
        cycle_np = cycle.cpu().numpy()
        source_vertices = geometry.vertices(source_np)
        target_vertices = geometry.vertices(target_np)
        predicted_vertices = geometry.vertices(predicted_np)
        backward_vertices = geometry.vertices(backward_np)
        source_volume = geometry.volume(source_vertices)
        target_volume = geometry.volume(target_vertices)
        predicted_volume = geometry.volume(predicted_vertices)
        backward_volume = geometry.volume(backward_vertices)
        gaps = np.asarray([pair.delta_years for pair in current], dtype=np.float64)
        observed_rate = (np.log(target_volume) - np.log(source_volume)) / gaps
        predicted_rate = (np.log(predicted_volume) - np.log(source_volume)) / gaps
        backward_rate = (np.log(backward_volume) - np.log(target_volume)) / (-gaps)
        for position, pair in enumerate(current):
            euclidean = np.linalg.norm(predicted_vertices[position] - target_vertices[position], axis=1)
            backward_euclidean = np.linalg.norm(backward_vertices[position] - source_vertices[position], axis=1)
            normals = geometry.vertex_normals(source_vertices[position])
            observed_local = np.sum((target_vertices[position] - source_vertices[position]) * normals, axis=1) / gaps[position]
            predicted_local = np.sum((predicted_vertices[position] - source_vertices[position]) * normals, axis=1) / gaps[position]
            meta = pair_metadata(archive, pair, split)
            row = {
                "structure": structure,
                "method": method,
                "method_role": "primary" if method in PRIMARY_METHODS else "exploratory_sensitivity",
                **meta,
                "pca_mse": float(np.mean((predicted_np[position] - target_np[position]) ** 2)),
                "vertex_coordinate_mae_mm": float(np.mean(np.abs(predicted_vertices[position] - target_vertices[position]))),
                "vertex_euclidean_mean_mm": float(np.mean(euclidean)),
                "vertex_euclidean_rmse_mm": float(np.sqrt(np.mean(euclidean**2))),
                "vertex_hd95_mm": float(np.quantile(euclidean, 0.95)),
                "volume_relative_error": float(abs(predicted_volume[position] - target_volume[position]) / target_volume[position]),
                "raw_mesh_volume_relative_error": float(abs(predicted_volume[position] - raw_volume[pair.target_index]) / max(raw_volume[pair.target_index], 1.0e-8)),
                "log_volume_rate_abs_error": float(abs(predicted_rate[position] - observed_rate[position])),
                "predicted_log_volume_rate": float(predicted_rate[position]),
                "observed_log_volume_rate": float(observed_rate[position]),
                "predicted_volume_change_mm3_per_year": float((predicted_volume[position] - source_volume[position]) / gaps[position]),
                "observed_volume_change_mm3_per_year": float((target_volume[position] - source_volume[position]) / gaps[position]),
                "backward_pca_mse": float(np.mean((backward_np[position] - source_np[position]) ** 2)),
                "backward_vertex_euclidean_mean_mm": float(np.mean(backward_euclidean)),
                "backward_volume_relative_error": float(abs(backward_volume[position] - source_volume[position]) / source_volume[position]),
                "backward_log_volume_rate_abs_error": float(abs(backward_rate[position] - observed_rate[position])),
                "cycle_pca_mse": float(np.mean((cycle_np[position] - source_np[position]) ** 2)),
                "composition_pca_mse": float(composition[position].cpu()),
                "local_normal_rate_mae_mm_per_year": float(np.mean(np.abs(predicted_local - observed_local))),
                "local_normal_rate_pearson": pearson(predicted_local, observed_local),
                "local_normal_top20_dice": top_change_dice(predicted_local, observed_local),
            }
            rows.append(row)
    return rows


PAIR_ERROR_METRICS = [
    "pca_mse",
    "vertex_coordinate_mae_mm",
    "vertex_euclidean_mean_mm",
    "vertex_euclidean_rmse_mm",
    "vertex_hd95_mm",
    "volume_relative_error",
    "raw_mesh_volume_relative_error",
    "log_volume_rate_abs_error",
    "backward_pca_mse",
    "backward_vertex_euclidean_mean_mm",
    "backward_volume_relative_error",
    "backward_log_volume_rate_abs_error",
    "cycle_pca_mse",
    "composition_pca_mse",
    "local_normal_rate_mae_mm_per_year",
]
PAIR_HIGHER_METRICS = ["local_normal_rate_pearson", "local_normal_top20_dice"]


def summarize_pair_rows(rows: list[dict[str, Any]], grouping: str = "diagnosis") -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if grouping == "diagnosis":
            keys = (row["structure"], row["method"], row["method_role"], row["split"], row["diagnosis"])
            buckets[keys].append(row)
            all_key = (row["structure"], row["method"], row["method_role"], row["split"], "ALL")
            buckets[all_key].append(row)
        else:
            keys = (row["structure"], row["method"], row["method_role"], row["split"], row["diagnosis"], row["gap_bin"])
            buckets[keys].append(row)
    output: list[dict[str, Any]] = []
    for key, current in sorted(buckets.items()):
        if grouping == "diagnosis":
            structure, method, role, split, diagnosis = key
            result = {"structure": structure, "method": method, "method_role": role, "split": split, "diagnosis": diagnosis, "rows": len(current)}
        else:
            structure, method, role, split, diagnosis, gap_bin = key
            result = {"structure": structure, "method": method, "method_role": role, "split": split, "diagnosis": diagnosis, "gap_bin": gap_bin, "rows": len(current)}
        for metric in PAIR_ERROR_METRICS + PAIR_HIGHER_METRICS + [
            "predicted_log_volume_rate", "observed_log_volume_rate",
            "predicted_volume_change_mm3_per_year", "observed_volume_change_mm3_per_year",
        ]:
            result[f"{metric}_mean"] = finite_mean(float(row[metric]) for row in current)
            result[f"{metric}_median"] = finite_median(float(row[metric]) for row in current)
        output.append(result)
    return output


def evaluate_trajectories(
    structure: str,
    method: str,
    adapter: Adapter,
    archive: dict[str, np.ndarray],
    geometry: Geometry,
    split: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    z = torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device)
    time = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
    years = torch.from_numpy(archive["visit_age_years"].astype(np.float32)).to(device)
    label = torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device)
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    trajectory_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    for subject_index in range(len(offsets) - 1):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        if end - start < 2:
            continue
        targets = torch.arange(start + 1, end, dtype=torch.long, device=device)
        count = int(targets.numel())
        prediction = adapter.transport(
            z[start : start + 1].expand(count, -1),
            time[start : start + 1].expand(count), time[targets],
            years[start : start + 1].expand(count), years[targets],
            label[start : start + 1].expand(count),
        )
        source_vertices = geometry.vertices(z[start : start + 1].cpu().numpy())
        target_vertices = geometry.vertices(z[targets].cpu().numpy())
        predicted_vertices = geometry.vertices(prediction.cpu().numpy())
        source_volume = float(geometry.volume(source_vertices)[0])
        observed_volume = geometry.volume(target_vertices)
        predicted_volume = geometry.volume(predicted_vertices)
        elapsed = archive["visit_age_years"][start + 1 : end].astype(np.float64) - float(archive["visit_age_years"][start])
        observed_slope = float(np.polyfit(elapsed, np.log(observed_volume), 1)[0]) if len(elapsed) >= 2 else float((np.log(observed_volume[0]) - math.log(source_volume)) / elapsed[0])
        predicted_slope = float(np.polyfit(elapsed, np.log(predicted_volume), 1)[0]) if len(elapsed) >= 2 else float((np.log(predicted_volume[0]) - math.log(source_volume)) / elapsed[0])
        diagnosis = str(archive["visit_diagnoses"][start])
        subject_id = str(archive["visit_subject_ids"][start])
        slope_rows.append({
            "structure": structure, "method": method,
            "method_role": "primary" if method in PRIMARY_METHODS else "exploratory_sensitivity",
            "split": split, "diagnosis": diagnosis, "subject_id": subject_id,
            "visits": end - start, "observed_log_volume_slope": observed_slope,
            "predicted_log_volume_slope": predicted_slope,
            "slope_abs_error": abs(predicted_slope - observed_slope),
            "no_change_slope_abs_error": abs(observed_slope),
        })
        for index, target in enumerate(range(start + 1, end)):
            trajectory_rows.append({
                "structure": structure, "method": method,
                "method_role": "primary" if method in PRIMARY_METHODS else "exploratory_sensitivity",
                "split": split, "diagnosis": diagnosis, "subject_id": subject_id,
                "target_scan_id": str(archive["visit_scan_ids"][target]),
                "elapsed_years": float(elapsed[index]), "source_volume_mm3": source_volume,
                "observed_volume_mm3": float(observed_volume[index]),
                "predicted_volume_mm3": float(predicted_volume[index]),
            })
    return trajectory_rows, slope_rows


def summarize_slopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["structure"], row["method"], row["method_role"], row["split"], row["diagnosis"])].append(row)
    output = []
    for (structure, method, role, split, diagnosis), current in sorted(buckets.items()):
        observed = np.asarray([row["observed_log_volume_slope"] for row in current])
        predicted = np.asarray([row["predicted_log_volume_slope"] for row in current])
        output.append({
            "structure": structure, "method": method, "method_role": role,
            "split": split, "diagnosis": diagnosis, "subjects": len(current),
            "slope_abs_error_mean": float(np.mean(np.abs(predicted - observed))),
            "slope_abs_error_median": float(np.median(np.abs(predicted - observed))),
            "observed_slope_mean": float(np.mean(observed)),
            "predicted_slope_mean": float(np.mean(predicted)),
            "slope_pearson": pearson(predicted, observed),
        })
    return output


def bootstrap_comparisons(rows: list[dict[str, Any]], replicates: int, seed: int = 42) -> list[dict[str, Any]]:
    comparisons = [
        ("anchored_v4", "v1", "primary"),
        ("e3_selected", "v1", "primary"),
        ("e3_latest_sensitivity", "e3_selected", "exploratory_sensitivity"),
    ]
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    test_rows = [row for row in rows if row["split"] == "test"]
    for structure in STRUCTURE_RUNS:
        for diagnosis in ("CN", "AD", "ALL"):
            subset = [row for row in test_rows if row["structure"] == structure and (diagnosis == "ALL" or row["diagnosis"] == diagnosis)]
            for current_method, baseline_method, role in comparisons:
                for metric in PAIR_ERROR_METRICS + PAIR_HIGHER_METRICS:
                    subject_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
                    for row in subset:
                        if row["method"] in {current_method, baseline_method} and math.isfinite(float(row[metric])):
                            subject_values[row["subject_id"]][row["method"]].append(float(row[metric]))
                    paired = []
                    for subject, methods in subject_values.items():
                        if current_method in methods and baseline_method in methods:
                            current = float(np.mean(methods[current_method]))
                            baseline = float(np.mean(methods[baseline_method]))
                            improvement = current - baseline if metric in PAIR_HIGHER_METRICS else baseline - current
                            paired.append(improvement)
                    if not paired:
                        continue
                    array = np.asarray(paired, dtype=np.float64)
                    boot = np.asarray([
                        np.mean(rng.choice(array, size=array.size, replace=True))
                        for _ in range(max(replicates, 1))
                    ])
                    output.append({
                        "structure": structure, "diagnosis": diagnosis,
                        "current_method": current_method, "baseline_method": baseline_method,
                        "comparison_role": role, "metric": metric, "subjects": len(array),
                        "improvement_mean": float(np.mean(array)),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "fraction_subjects_better": float(np.mean(array > 0.0)),
                        "higher_improvement_is_better": True,
                    })
    return output


def plot_metric_bars(summary: list[dict[str, Any]], output: Path) -> None:
    metrics = ["vertex_coordinate_mae_mm_mean", "volume_relative_error_mean", "log_volume_rate_abs_error_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for row_index, structure in enumerate(STRUCTURE_RUNS):
        for column, metric in enumerate(metrics):
            ax = axes[row_index, column]
            current = [row for row in summary if row["structure"] == structure and row["split"] == "test" and row["diagnosis"] == "ALL"]
            mapping = {row["method"]: row for row in current}
            methods = [method for method in METHOD_ORDER if method in mapping]
            values = [mapping[method][metric] for method in methods]
            ax.bar(range(len(methods)), values, color=["#999999", "#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(methods)])
            ax.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods], rotation=30, ha="right")
            ax.set_title(f"{STRUCTURE_RUNS[structure]['label']}\n{metric.replace('_mean', '')}")
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_volume_rates(summary: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, structure in zip(axes, STRUCTURE_RUNS):
        current = [row for row in summary if row["structure"] == structure and row["split"] == "test" and row["diagnosis"] in {"CN", "AD"}]
        mapping = {(row["method"], row["diagnosis"]): row for row in current}
        methods = [method for method in METHOD_ORDER if (method, "CN") in mapping and method != "no_change"]
        x = np.arange(len(methods))
        width = 0.18
        for offset, diagnosis, kind, color in ((-1.5, "CN", "predicted", "#4C78A8"), (-0.5, "CN", "observed", "#9ECAE9"), (0.5, "AD", "predicted", "#E45756"), (1.5, "AD", "observed", "#F7B6B2")):
            field = f"{kind}_volume_change_mm3_per_year_mean"
            ax.bar(x + offset * width, [mapping[(method, diagnosis)][field] for method in methods], width, label=f"{diagnosis} {kind}", color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=25, ha="right")
        ax.set_title(STRUCTURE_RUNS[structure]["label"])
        ax.set_ylabel("Volume change (mm³/year)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_e3_histories(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (structure, spec) in zip(axes, STRUCTURE_RUNS.items()):
        path = BASE_ROOT / spec["folder"] / "cocycle_v4" / "training" / spec["e3"] / "history.jsonl"
        history = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        validation = [row for row in history if "val_score" in row]
        epochs = [row["epoch"] for row in validation]
        scores = [row["val_score"] for row in validation]
        colors = ["#54A24B" if row["val_feasible"] else "#E45756" for row in validation]
        ax.plot(epochs, scores, color="#4C78A8", linewidth=1.5)
        ax.scatter(epochs, scores, c=colors, s=35)
        ax.set_title(f"{spec['label']} E3 validation\ngreen=feasible, red=rejected")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation composite (lower better)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_markdown(
    report: dict[str, Any],
    summary: list[dict[str, Any]],
    slope_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Independent longitudinal model validation",
        "",
        "This report uses one evaluator for no-change, V1, anchored V4, E3 selected, and E3 latest sensitivity. BrainODE is not yet trained.",
        "",
        "## Contract status",
        "",
        f"- Overall: **{report['status']}**",
        "- Strict CN/AD and disjoint subject/scan splits: passed for both structures.",
        "- Anchored V4 contains an exact frozen copy of its selected V1: passed.",
        "- E3 selected is primary; E3 latest is exploratory and must not replace it based on test results.",
        "",
        "## Locked test overview",
        "",
        "| Structure | Method | Diagnosis | Vertex MAE (mm) | Volume rel. error | Log-rate error | Predicted mm³/year | Observed mm³/year |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for structure in STRUCTURE_RUNS:
        for method in METHOD_ORDER:
            for diagnosis in ("CN", "AD"):
                match = next((row for row in summary if row["structure"] == structure and row["method"] == method and row["split"] == "test" and row["diagnosis"] == diagnosis), None)
                if match is None:
                    continue
                lines.append(
                    f"| {STRUCTURE_RUNS[structure]['label']} | {METHOD_LABELS[method]} | {diagnosis} | "
                    f"{match['vertex_coordinate_mae_mm_mean']:.4f} | {match['volume_relative_error_mean']:.4f} | "
                    f"{match['log_volume_rate_abs_error_mean']:.4f} | {match['predicted_volume_change_mm3_per_year_mean']:.1f} | "
                    f"{match['observed_volume_change_mm3_per_year_mean']:.1f} |"
                )
    lines.extend([
        "",
        "## Interpretation constraints",
        "",
        "- Primary claims must use `method_role=primary` rows only.",
        "- The E3 latest checkpoint is a validation sensitivity analysis; its test metrics are exploratory.",
        "- All predictions and target vertices use the same structure-specific PCA-150 decoder. Raw-mesh volume is also reported separately.",
        "- Subject-bootstrap intervals are in `paired_subject_bootstrap.csv`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("Batch size and bootstrap replicates must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation output: {args.output_dir}")
    device = choose_device(args.device)
    print(f"Independent validation | device={device} | output={args.output_dir}", flush=True)
    all_pair_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []
    all_slope_rows: list[dict[str, Any]] = []
    contract_report: dict[str, Any] = {}

    for structure, specification in STRUCTURE_RUNS.items():
        print(f"Loading and validating {specification['label']}...", flush=True)
        archives, pairs, geometry, adapters, structure_report = load_structure(structure, specification, device)
        contract_report[structure] = structure_report
        for method in METHOD_ORDER:
            adapter = adapters[method]
            for split in ("val", "test"):
                print(f"  scoring {method:24s} {split} pairs={len(pairs[split])}", flush=True)
                all_pair_rows.extend(evaluate_method_pairs(
                    structure, method, adapter, archives[split], pairs[split], geometry,
                    split, device, args.batch_size,
                ))
                trajectories, slopes = evaluate_trajectories(
                    structure, method, adapter, archives[split], geometry, split, device,
                )
                all_trajectory_rows.extend(trajectories)
                all_slope_rows.extend(slopes)

    pair_summary = summarize_pair_rows(all_pair_rows, "diagnosis")
    gap_summary = summarize_pair_rows(all_pair_rows, "gap")
    slope_summary = summarize_slopes(all_slope_rows)
    bootstrap = bootstrap_comparisons(all_pair_rows, args.bootstrap_replicates)
    report = {
        "status": "pass",
        "device": str(device),
        "structures": contract_report,
        "methods": {
            method: {"label": METHOD_LABELS[method], "role": "primary" if method in PRIMARY_METHODS else "exploratory_sensitivity"}
            for method in METHOD_ORDER
        },
        "pair_rows": len(all_pair_rows),
        "trajectory_rows": len(all_trajectory_rows),
        "subject_slope_rows": len(all_slope_rows),
        "bootstrap_replicates": args.bootstrap_replicates,
        "brainode_included": False,
        "source_artifacts_modified": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    write_csv(tables / "per_pair_metrics.csv", all_pair_rows)
    write_csv(tables / "pair_summary.csv", pair_summary)
    write_csv(tables / "gap_summary.csv", gap_summary)
    write_csv(tables / "trajectory_predictions.csv", all_trajectory_rows)
    write_csv(tables / "subject_slopes.csv", all_slope_rows)
    write_csv(tables / "subject_slope_summary.csv", slope_summary)
    write_csv(tables / "paired_subject_bootstrap.csv", bootstrap)
    write_json(args.output_dir / "validation_report.json", report)
    figures.mkdir(parents=True, exist_ok=True)
    plot_metric_bars(pair_summary, figures / "test_metric_comparison.png")
    plot_volume_rates(pair_summary, figures / "test_volume_change_rates.png")
    plot_e3_histories(figures / "e3_validation_checkpoint_tradeoff.png")
    (args.output_dir / "README.md").write_text(build_markdown(report, pair_summary, slope_summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Validation complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
