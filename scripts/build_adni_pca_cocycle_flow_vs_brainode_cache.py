#!/usr/bin/env python3
"""Build a load-only current ADNI PCA Cocycle Flow versus BrainODE cache.

The cache is intentionally structure-specific and never changes a mesh,
checkpoint, PCA model, or input archive.  It evaluates the locked current
PCA Cocycle Flow and BrainODE checkpoints on identical forward
pairs, then stores all tables and registered-mesh visualisation payloads that
the companion notebook needs.  Test is the primary split; train+val+test is
saved only as a descriptive aggregate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from train_adni_synthseg_pca_brainode_paper_core import (
    PaperCoreODEFuncWithAttention,
    integrate as integrate_brainode,
)
from train_adni_synthseg_pca_cocycle_v4 import (
    BASE_ROOT,
    STRUCTURES,
    choose_device,
    load_archive,
    load_pairs,
    read_json,
    validate_pca_model,
)
from train_adni_synthseg_pca_cocycle_v5 import (
    DirectDiagnosisResidualCocycleFlow,
    resolve_path,
)


SCHEMA_VERSION = 1
REPORT_ROOT = BASE_ROOT / "longitudinal_model_validation" / "pca_cocycle_flow_vs_paper_core_brainode"
MODEL_LABELS = {
    "brainode": "BrainODE",
    "cocycle_flow": "PCA Cocycle Flow",
}
METRICS: dict[str, dict[str, Any]] = {
    "endpoint_vertex_mae_mm": {"label": "Vertex MAE (mm)", "lower_is_better": True},
    "endpoint_vertex_euclidean_mm": {"label": "Vertex Euclidean distance (mm)", "lower_is_better": True},
    "endpoint_vertex_hd95_mm": {"label": "Corresponding-vertex HD95 (mm)", "lower_is_better": True},
    "endpoint_pca_mse": {"label": "PCA MSE", "lower_is_better": True},
    "volume_relative_error": {"label": "Volume relative error", "lower_is_better": True},
    "log_volume_rate_abs_error": {"label": "Log-volume-rate error", "lower_is_better": True},
    "local_normal_rate_mae": {"label": "Local normal-rate MAE", "lower_is_better": True},
    "local_normal_rate_pearson": {"label": "Local normal-rate Pearson", "lower_is_better": False},
    "local_normal_top_change_dice": {"label": "Top-change Dice", "lower_is_better": False},
}
REQUIRED_TABLES = (
    "cohort_counts.csv",
    "per_pair_metrics.csv",
    "metric_summary.csv",
    "paired_metric_comparison.csv",
    "paired_bootstrap_ci.csv",
    "gap_summary.csv",
    "subject_volume_trends.csv",
    "cohort_volume_trend_summary.csv",
    "cohort_volume_curve.csv",
    "selected_volume_trajectories.csv",
    "counterfactual_trajectories.csv",
    "selected_mesh_metrics.csv",
    "group_hotspot_metrics.csv",
)
REQUIRED_MESHES = (
    "template_faces.csv",
    "selected_cn_vertices.csv",
    "selected_ad_vertices.csv",
    "group_cn_change_maps.csv",
    "group_ad_change_maps.csv",
)


@dataclass(frozen=True)
class PairRecord:
    split: str
    source: int
    target: int
    diagnosis: str
    subject_id: str
    source_scan_id: str
    target_scan_id: str
    source_visit_order: int
    target_visit_order: int
    gap_years: float
    source_age_years: float
    target_age_years: float
    pair_type: str


@dataclass
class Geometry:
    mean_flat: np.ndarray
    components: np.ndarray
    faces: np.ndarray
    score_mean: np.ndarray
    score_std: np.ndarray

    def vertices(self, standardized_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(standardized_scores, dtype=np.float64)
        raw = scores * self.score_std + self.score_mean
        return (raw @ self.components + self.mean_flat).reshape(scores.shape[0], -1, 3)

    def volume(self, vertices: np.ndarray) -> np.ndarray:
        values = np.asarray(vertices, dtype=np.float64)
        first = values[:, self.faces[:, 0], :]
        second = values[:, self.faces[:, 1], :]
        third = values[:, self.faces[:, 2], :]
        signed = np.einsum("bfi,bfi->bf", first, np.cross(second, third)).sum(axis=1) / 6.0
        return np.maximum(np.abs(signed), 1.0e-8)


@dataclass
class InferenceContext:
    structure: str
    device: torch.device
    geometry: Geometry
    archives: dict[str, dict[str, np.ndarray]]
    cocycle: DirectDiagnosisResidualCocycleFlow
    brainode: PaperCoreODEFuncWithAttention
    brainode_substeps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=STRUCTURES)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--example-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-change-fraction", type=float, default=0.20)
    parser.add_argument("--group-map-max-pairs-per-diagnosis", type=int, default=0)
    parser.add_argument("--skip-group-maps", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate contracts and one matched pair without writing files.")
    return parser.parse_args()


def selected_paths(structure: str) -> dict[str, Path]:
    """Resolve the locked selected checkpoint for each structure.

    The report exposes the neutral name ``PCA Cocycle Flow``.  These paths are
    retained only as immutable provenance for the already completed runs.
    """
    selected_config = {
        "hippocampus": BASE_ROOT / "hippocampus_pca_cocycle_v4" / "cocycle_v5" / "configs" / "c4_cocycle_v5.json",
        "lateral_ventricle": BASE_ROOT / "lateral_ventricle_pca_cocycle_v4" / "cocycle_v5" / "configs" / "c1_cocycle_v5.json",
    }[structure]
    flow_config = read_json(selected_config)
    flow_run = selected_config.parent.parent / "training" / str(flow_config["training"]["run_name"])
    brain_config = BASE_ROOT / f"{structure}_pca_cocycle_v4" / "brainode_paper_core" / "configs" / "paper_core_primary.json"
    brain_settings = read_json(brain_config)
    brain_run = brain_config.parent.parent / "training" / str(brain_settings["training"]["run_name"])
    return {
        "flow_config": selected_config,
        "flow_checkpoint": flow_run / "checkpoints" / "best_shape.pt",
        "brain_config": brain_config,
        "brain_checkpoint": brain_run / "checkpoints" / "best.pt",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": sha256(resolved)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_checkpoint(path: Path, key: str, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    if payload.get("test_data_loaded", False) is not False:
        raise ValueError(f"Checkpoint reports test access during training: {path}")
    state = payload.get(key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Missing {key} in {path}")
    if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"Checkpoint has non-finite state: {path}")
    return payload


def load_context(structure: str, device: torch.device, splits: Iterable[str]) -> tuple[InferenceContext, dict[str, Any]]:
    paths = selected_paths(structure)
    flow_config = read_json(paths["flow_config"])
    brain_config = read_json(paths["brain_config"])
    if flow_config.get("scientific_contract", {}).get("strict_no_mci") is not True:
        raise ValueError("Cocycle Flow config is not strict no-MCI")
    if brain_config.get("scientific_contract", {}).get("strict_no_mci") is not True:
        raise ValueError("BrainODE config is not strict no-MCI")
    if flow_config.get("scientific_contract", {}).get("test_loaded_during_training") is not False:
        raise ValueError("Cocycle Flow training contract indicates test access")
    if brain_config.get("scientific_contract", {}).get("test_loaded_during_training") is not False:
        raise ValueError("BrainODE training contract indicates test access")
    flow_run_contract = paths["flow_checkpoint"].parents[1] / "run_contract.json"
    if not flow_run_contract.is_file() or read_json(flow_run_contract).get("test_loaded_during_training") is not False:
        raise ValueError("Cocycle Flow run contract does not prove test isolation")
    flow_input = read_json(resolve_path(flow_config["input_config"]))
    brain_input = read_json(resolve_path(brain_config["input_config"]))
    if flow_input["structure"] != brain_input["structure"]:
        raise ValueError("Model input structures differ")
    for split in ("train", "val", "test"):
        key = f"{split}_sequences"
        if Path(flow_input["dataset"][key]).resolve() != Path(brain_input["dataset"][key]).resolve():
            raise ValueError(f"Model sequence archive differs for {split}")
    if Path(flow_input["representation"]["pca_model_dir"]).resolve() != Path(brain_input["representation"]["pca_model_dir"]).resolve():
        raise ValueError("Model PCA directories differ")

    archives = {
        split: load_archive(resolve_path(flow_input["dataset"][f"{split}_sequences"]), split, 150)
        for split in splits
    }
    train_archive = load_archive(resolve_path(flow_input["dataset"]["train_sequences"]), "train", 150)
    pca = validate_pca_model(flow_input, 150)
    geometry = Geometry(
        mean_flat=pca["mean"].astype(np.float64),
        components=pca["components"].astype(np.float64),
        faces=pca["faces"].astype(np.int64),
        score_mean=train_archive["train_pca_mean_150"].astype(np.float64),
        score_std=np.maximum(train_archive["train_pca_std_150"].astype(np.float64), 1.0e-6),
    )
    flow_payload = load_checkpoint(paths["flow_checkpoint"], "flow_state_dict", device)
    cocycle = DirectDiagnosisResidualCocycleFlow(
        latent_dim=int(flow_config["model"]["latent_dim"]),
        width=int(flow_config["model"]["width"]),
        residual_blocks=int(flow_config["model"]["residual_blocks"]),
    ).to(device)
    cocycle.load_state_dict(flow_payload["flow_state_dict"])
    cocycle.eval()
    brain_payload = load_checkpoint(paths["brain_checkpoint"], "model_state_dict", device)
    brainode = PaperCoreODEFuncWithAttention(
        latent_dim=int(brain_config["model"]["latent_dim"]),
        condition_dim=int(brain_config["model"]["condition_dim"]),
    ).to(device)
    brainode.load_state_dict(brain_payload["model_state_dict"])
    brainode.eval()
    provenance = {
        "flow_config": signature(paths["flow_config"]),
        "flow_checkpoint": signature(paths["flow_checkpoint"]),
        "brainode_config": signature(paths["brain_config"]),
        "brainode_checkpoint": signature(paths["brain_checkpoint"]),
        "archives": {split: signature(resolve_path(flow_input["dataset"][f"{split}_sequences"])) for split in splits},
        "pca_model": {name: signature(Path(flow_input["representation"]["pca_model_dir"]) / filename) for name, filename in {"mean": "mean.npy", "components": "components_150.npy", "faces": "faces.npy"}.items()},
        "strict_no_mci": True,
        "source_meshes_modified": False,
        "test_loaded_during_training": False,
    }
    return InferenceContext(structure, device, geometry, archives, cocycle, brainode, int(brain_config["training"]["integration_substeps"])), provenance


def records_for_split(split: str, archive: dict[str, np.ndarray], pair_file: Path) -> list[PairRecord]:
    rows = load_pairs(pair_file, archive, split)
    subject = archive["visit_subject_ids"].astype(str)
    scans = archive["visit_scan_ids"].astype(str)
    visits = archive["visit_orders"].astype(np.int64)
    age = archive["visit_age_years"].astype(np.float64)
    output: list[PairRecord] = []
    for row in rows:
        output.append(PairRecord(
            split=split,
            source=int(row.source_index),
            target=int(row.target_index),
            diagnosis=str(row.diagnosis),
            subject_id=str(subject[row.source_index]),
            source_scan_id=str(scans[row.source_index]),
            target_scan_id=str(scans[row.target_index]),
            source_visit_order=int(visits[row.source_index]),
            target_visit_order=int(visits[row.target_index]),
            gap_years=float(age[row.target_index] - age[row.source_index]),
            source_age_years=float(age[row.source_index]),
            target_age_years=float(age[row.target_index]),
            pair_type=str(row.pair_type),
        ))
    return output


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_vectors = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    output = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(output, faces[:, corner], face_vectors)
    return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1.0e-12)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if len(a) < 3 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def top_change_dice(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    if not 0.0 < fraction < 1.0:
        raise ValueError("top-change fraction must be in (0, 1)")
    count = max(1, int(math.ceil(fraction * len(left))))
    a = np.argpartition(np.abs(left), -count)[-count:]
    b = np.argpartition(np.abs(right), -count)[-count:]
    return float(2.0 * len(set(a).intersection(set(b))) / (2 * count))


def gap_bin(pair: PairRecord) -> str:
    if pair.pair_type == "adjacent":
        return "adjacent"
    return "short" if pair.gap_years <= 2.0 else "long"


@torch.no_grad()
def infer_pair_predictions(context: InferenceContext, archive: dict[str, np.ndarray], records: list[PairRecord], label_override: int | None = None) -> dict[str, np.ndarray]:
    if not records:
        return {"source": np.empty((0, 150)), "target": np.empty((0, 150)), "cocycle_flow": np.empty((0, 150)), "brainode": np.empty((0, 150))}
    source_index = np.asarray([record.source for record in records], dtype=np.int64)
    target_index = np.asarray([record.target for record in records], dtype=np.int64)
    label = archive["visit_label_ad"][source_index].astype(np.float32)
    if label_override is not None:
        label.fill(float(label_override))
    source_z = torch.from_numpy(archive["visit_pca_standardized_150"][source_index].astype(np.float32)).to(context.device)
    source_time = torch.from_numpy(archive["visit_age_norm_train"][source_index].astype(np.float32)).to(context.device)
    target_time = torch.from_numpy(archive["visit_age_norm_train"][target_index].astype(np.float32)).to(context.device)
    label_tensor = torch.from_numpy(label).to(context.device)
    flow_prediction = context.cocycle.transport(source_z, source_time, target_time, label_tensor).cpu().numpy().astype(np.float64)
    raw_mean, raw_std = context.geometry.score_mean, context.geometry.score_std
    raw_values = archive["visit_pca_150"].astype(np.float32)
    brain_predictions: list[np.ndarray] = []
    for position in range(len(records)):
        values = torch.from_numpy(raw_values[source_index[position] : source_index[position] + 1]).to(context.device)
        times = torch.stack((source_time[position], target_time[position])).reshape(2)
        condition = torch.tensor([label[position]], dtype=torch.float32, device=context.device)
        predicted_raw = integrate_brainode(context.brainode, values, times, condition, context.brainode_substeps)[-1]
        brain_predictions.append(predicted_raw.cpu().numpy().astype(np.float64))
    brain_raw = np.stack(brain_predictions)
    return {
        "source": archive["visit_pca_standardized_150"][source_index].astype(np.float64),
        "target": archive["visit_pca_standardized_150"][target_index].astype(np.float64),
        "cocycle_flow": flow_prediction,
        "brainode": (brain_raw - raw_mean) / raw_std,
    }


def metric_rows(context: InferenceContext, archive: dict[str, np.ndarray], records: list[PairRecord], batch_size: int, top_fraction: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total_batches = max(1, math.ceil(len(records) / batch_size))
    for batch_number, start in enumerate(range(0, len(records), batch_size), start=1):
        current = records[start : start + batch_size]
        predicted = infer_pair_predictions(context, archive, current)
        decoded = {name: context.geometry.vertices(value) for name, value in predicted.items()}
        source_vertices, target_vertices = decoded["source"], decoded["target"]
        source_volume = context.geometry.volume(source_vertices)
        target_volume = context.geometry.volume(target_vertices)
        for model in ("brainode", "cocycle_flow"):
            prediction_vertices = decoded[model]
            prediction_volume = context.geometry.volume(prediction_vertices)
            distance = np.linalg.norm(prediction_vertices - target_vertices, axis=2)
            for index, record in enumerate(current):
                normal = vertex_normals(source_vertices[index], context.geometry.faces)
                actual_rate = np.sum((target_vertices[index] - source_vertices[index]) * normal, axis=1) / record.gap_years
                predicted_rate = np.sum((prediction_vertices[index] - source_vertices[index]) * normal, axis=1) / record.gap_years
                output.append({
                    "split": record.split,
                    "diagnosis": record.diagnosis,
                    "subject_id": record.subject_id,
                    "source_scan_id": record.source_scan_id,
                    "target_scan_id": record.target_scan_id,
                    "source_visit_order": record.source_visit_order,
                    "target_visit_order": record.target_visit_order,
                    "pair_type": record.pair_type,
                    "gap_bin": gap_bin(record),
                    "gap_years": record.gap_years,
                    "source_age_years": record.source_age_years,
                    "target_age_years": record.target_age_years,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "endpoint_pca_mse": float(np.mean((predicted[model][index] - predicted["target"][index]) ** 2)),
                    "endpoint_vertex_mae_mm": float(np.mean(np.abs(prediction_vertices[index] - target_vertices[index]))),
                    "endpoint_vertex_euclidean_mm": float(np.mean(distance[index])),
                    "endpoint_vertex_hd95_mm": float(np.quantile(distance[index], 0.95)),
                    "volume_relative_error": float(abs(prediction_volume[index] - target_volume[index]) / target_volume[index]),
                    "log_volume_rate_abs_error": float(abs((math.log(prediction_volume[index]) - math.log(target_volume[index])) / record.gap_years)),
                    "local_normal_rate_mae": float(np.mean(np.abs(predicted_rate - actual_rate))),
                    "local_normal_rate_pearson": pearson(predicted_rate, actual_rate),
                    "local_normal_top_change_dice": top_change_dice(predicted_rate, actual_rate, top_fraction),
                    "source_volume_mm3": float(source_volume[index]),
                    "target_volume_mm3": float(target_volume[index]),
                    "predicted_volume_mm3": float(prediction_volume[index]),
                    "observed_log_volume_rate": float((math.log(target_volume[index]) - math.log(source_volume[index])) / record.gap_years),
                    "predicted_log_volume_rate": float((math.log(prediction_volume[index]) - math.log(source_volume[index])) / record.gap_years),
                })
        print(f"{context.structure}: evaluated {batch_number}/{total_batches} batches ({min(start + len(current), len(records))}/{len(records)} pairs)", flush=True)
    return output


def add_all_split(frame: pd.DataFrame) -> pd.DataFrame:
    all_frame = frame.copy()
    all_frame["split"] = "all"
    return pd.concat((frame, all_frame), ignore_index=True)


def paired_tables(per_pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["split", "diagnosis", "subject_id", "source_scan_id", "target_scan_id", "source_visit_order", "target_visit_order", "gap_bin", "gap_years"]
    rows: list[pd.DataFrame] = []
    for metric, specification in METRICS.items():
        pivot = per_pair.pivot_table(index=keys, columns="model", values=metric, aggfunc="mean").dropna().reset_index()
        if set(MODEL_LABELS) - set(pivot.columns):
            raise ValueError(f"Missing matched model rows for {metric}")
        lower = bool(specification["lower_is_better"])
        pivot["metric"] = metric
        pivot["metric_label"] = specification["label"]
        pivot["direction"] = "lower" if lower else "higher"
        pivot["direction_symbol"] = "↓" if lower else "↑"
        pivot["brainode_value"] = pivot["brainode"]
        pivot["cocycle_flow_value"] = pivot["cocycle_flow"]
        pivot["delta_favoring_cocycle_flow"] = pivot["brainode_value"] - pivot["cocycle_flow_value"] if lower else pivot["cocycle_flow_value"] - pivot["brainode_value"]
        pivot["relative_gain_percent"] = np.where(lower, 100.0 * pivot["delta_favoring_cocycle_flow"] / pivot["brainode_value"].replace(0.0, np.nan), np.nan)
        pivot["cocycle_flow_wins"] = (pivot["delta_favoring_cocycle_flow"] > 0.0).astype(int)
        rows.append(pivot[keys + ["metric", "metric_label", "direction", "direction_symbol", "brainode_value", "cocycle_flow_value", "delta_favoring_cocycle_flow", "relative_gain_percent", "cocycle_flow_wins"]])
    paired = pd.concat(rows, ignore_index=True)
    summary = (
        add_all_split(paired)
        .groupby(["split", "diagnosis", "metric", "metric_label", "direction", "direction_symbol"], as_index=False)
        .agg(
            pairs=("delta_favoring_cocycle_flow", "size"),
            brainode_mean=("brainode_value", "mean"),
            cocycle_flow_mean=("cocycle_flow_value", "mean"),
            delta_favoring_cocycle_flow_mean=("delta_favoring_cocycle_flow", "mean"),
            relative_gain_percent=("relative_gain_percent", "mean"),
            cocycle_flow_win_fraction=("cocycle_flow_wins", "mean"),
        )
    )
    summary["better_model"] = np.where(summary["delta_favoring_cocycle_flow_mean"] > 0.0, MODEL_LABELS["cocycle_flow"], np.where(summary["delta_favoring_cocycle_flow_mean"] < 0.0, MODEL_LABELS["brainode"], "Tie"))
    return paired, summary


def bootstrap_table(paired: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for offset, ((split, diagnosis, metric), group) in enumerate(add_all_split(paired).groupby(["split", "diagnosis", "metric"], sort=True)):
        subject_values = group.groupby("subject_id")["delta_favoring_cocycle_flow"].mean().dropna().to_numpy(float)
        if not len(subject_values):
            continue
        rng = np.random.default_rng(seed + 1009 * offset)
        draws = rng.integers(0, len(subject_values), size=(replicates, len(subject_values)))
        samples = subject_values[draws].mean(axis=1)
        specification = METRICS[metric]
        rows.append({
            "split": split, "diagnosis": diagnosis, "metric": metric, "metric_label": specification["label"],
            "subjects": int(len(subject_values)), "replicates": int(replicates),
            "delta_favoring_cocycle_flow_mean": float(subject_values.mean()),
            "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975)),
            "probability_cocycle_flow_better": float(np.mean(samples > 0.0)),
        })
    return pd.DataFrame(rows)


def trend_tables(per_pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = per_pair[per_pair["source_visit_order"] == 0].copy()
    trends: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for (split, diagnosis, subject_id, model, model_label), group in baseline.groupby(["split", "diagnosis", "subject_id", "model", "model_label"], sort=True):
        group = group.sort_values(["target_visit_order", "target_age_years"]).drop_duplicates("target_visit_order")
        first = group.iloc[0]
        elapsed = np.concatenate(([0.0], group["target_age_years"].to_numpy(float) - float(first["source_age_years"])))
        observed = np.concatenate(([float(first["source_volume_mm3"])], group["target_volume_mm3"].to_numpy(float)))
        predicted = np.concatenate(([float(first["source_volume_mm3"])], group["predicted_volume_mm3"].to_numpy(float)))
        if np.unique(elapsed).size < 2:
            continue
        observed_slope = float(np.polyfit(elapsed, np.log(observed), 1)[0])
        predicted_slope = float(np.polyfit(elapsed, np.log(predicted), 1)[0])
        trends.append({
            "split": split, "diagnosis": diagnosis, "subject_id": subject_id, "model": model, "model_label": model_label,
            "visits": int(len(elapsed)), "maximum_followup_years": float(elapsed[-1]),
            "observed_log_volume_slope_per_year": observed_slope, "predicted_log_volume_slope_per_year": predicted_slope,
            "slope_absolute_error": abs(predicted_slope - observed_slope),
            "observed_mm3_per_year": float(np.polyfit(elapsed, observed, 1)[0]),
            "predicted_mm3_per_year": float(np.polyfit(elapsed, predicted, 1)[0]),
        })
        for time, actual, estimate in zip(elapsed, observed, predicted):
            curves.append({
                "split": split, "diagnosis": diagnosis, "subject_id": subject_id, "model": model, "model_label": model_label,
                "elapsed_years": float(time), "elapsed_year_bin": round(float(time) * 2.0) / 2.0,
                "observed_relative_volume": float(actual / observed[0]), "predicted_relative_volume": float(estimate / observed[0]),
            })
    subject = pd.DataFrame(trends)
    curve = pd.DataFrame(curves)
    summary_rows: list[dict[str, Any]] = []
    for (split, diagnosis, model, model_label), group in add_all_split(subject).groupby(["split", "diagnosis", "model", "model_label"], sort=True):
        observed = group["observed_log_volume_slope_per_year"].to_numpy(float)
        predicted = group["predicted_log_volume_slope_per_year"].to_numpy(float)
        summary_rows.append({
            "split": split, "diagnosis": diagnosis, "model": model, "model_label": model_label, "subjects": int(len(group)),
            "observed_slope_mean": float(observed.mean()), "predicted_slope_mean": float(predicted.mean()),
            "slope_absolute_error_mean": float(np.mean(np.abs(predicted - observed))),
            "slope_pearson": pearson(predicted, observed),
            "observed_mm3_per_year_mean": float(group["observed_mm3_per_year"].mean()),
            "predicted_mm3_per_year_mean": float(group["predicted_mm3_per_year"].mean()),
        })
    curve_summary = (
        curve.groupby(["split", "diagnosis", "model", "model_label", "elapsed_year_bin"], as_index=False)
        .agg(subjects=("subject_id", "nunique"), observed_relative_volume_mean=("observed_relative_volume", "mean"), predicted_relative_volume_mean=("predicted_relative_volume", "mean"))
    )
    return subject, pd.DataFrame(summary_rows), curve_summary


def select_examples(per_pair: pd.DataFrame, split: str) -> dict[str, dict[str, Any]]:
    candidates = per_pair[(per_pair["split"] == split) & (per_pair["model"] == "cocycle_flow") & (per_pair["source_visit_order"] == 0)]
    output: dict[str, dict[str, Any]] = {}
    for diagnosis in ("CN", "AD"):
        score = (
            candidates[candidates["diagnosis"] == diagnosis]
            .groupby("subject_id", as_index=False)
            .agg(future_visits=("target_visit_order", "nunique"), maximum_followup_years=("gap_years", "max"))
            .sort_values(["future_visits", "maximum_followup_years", "subject_id"], ascending=[False, False, True])
        )
        if score.empty:
            raise ValueError(f"No {split} baseline trajectory for {diagnosis}")
        row = score.iloc[0]
        output[diagnosis] = {"split": split, "diagnosis": diagnosis, "subject_id": str(row.subject_id), "future_visits": int(row.future_visits), "maximum_followup_years": float(row.maximum_followup_years), "selection_rule": "most follow-up visits, then longest follow-up, then subject ID"}
    return output


def selected_trajectories(per_pair: pd.DataFrame, selected: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diagnosis, info in selected.items():
        group = per_pair[(per_pair["split"] == info["split"]) & (per_pair["diagnosis"] == diagnosis) & (per_pair["subject_id"] == info["subject_id"]) & (per_pair["source_visit_order"] == 0)]
        source = group.iloc[0]
        rows.append({"diagnosis": diagnosis, "subject_id": info["subject_id"], "age_years": float(source.source_age_years), "visit_order": 0, "series": "Observed source", "model": "observed", "volume_mm3": float(source.source_volume_mm3)})
        future = group.drop_duplicates("target_visit_order").sort_values("target_visit_order")
        for item in future.itertuples(index=False):
            rows.append({"diagnosis": diagnosis, "subject_id": info["subject_id"], "age_years": float(item.target_age_years), "visit_order": int(item.target_visit_order), "series": "Observed follow-up", "model": "observed", "volume_mm3": float(item.target_volume_mm3)})
        for model, label in MODEL_LABELS.items():
            for item in group[group["model"] == model].sort_values("target_visit_order").itertuples(index=False):
                rows.append({"diagnosis": diagnosis, "subject_id": info["subject_id"], "age_years": float(item.target_age_years), "visit_order": int(item.target_visit_order), "series": label, "model": model, "volume_mm3": float(item.predicted_volume_mm3)})
    return pd.DataFrame(rows)


def matching_baseline_records(records: list[PairRecord], subject_id: str) -> list[PairRecord]:
    output = [record for record in records if record.subject_id == subject_id and record.source_visit_order == 0]
    if not output:
        raise ValueError(f"No baseline records for subject {subject_id}")
    return sorted(output, key=lambda item: item.target_visit_order)


def counterfactual_table(context: InferenceContext, records: dict[str, list[PairRecord]], selected: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diagnosis, info in selected.items():
        archive = context.archives[info["split"]]
        current = matching_baseline_records(records[info["split"]], info["subject_id"])
        source_vertices = context.geometry.vertices(archive["visit_pca_standardized_150"][np.asarray([current[0].source])])[0]
        source_volume = float(context.geometry.volume(source_vertices[None])[0])
        rows.append({"diagnosis": diagnosis, "subject_id": info["subject_id"], "age_years": current[0].source_age_years, "visit_order": 0, "condition": "observed", "model": "observed", "series": "Observed source", "volume_mm3": source_volume})
        target = context.geometry.vertices(archive["visit_pca_standardized_150"][np.asarray([item.target for item in current])])
        for item, volume in zip(current, context.geometry.volume(target)):
            rows.append({"diagnosis": diagnosis, "subject_id": info["subject_id"], "age_years": item.target_age_years, "visit_order": item.target_visit_order, "condition": "observed", "model": "observed", "series": "Observed follow-up", "volume_mm3": float(volume)})
        for condition, label in ((0, "CN"), (1, "AD")):
            predicted = infer_pair_predictions(context, archive, current, label_override=condition)
            for model in MODEL_LABELS:
                vertices = context.geometry.vertices(predicted[model])
                for item, volume in zip(current, context.geometry.volume(vertices)):
                    rows.append({"diagnosis": diagnosis, "subject_id": info["subject_id"], "age_years": item.target_age_years, "visit_order": item.target_visit_order, "condition": label, "model": model, "series": f"{MODEL_LABELS[model]} as {label}", "volume_mm3": float(volume)})
    return pd.DataFrame(rows)


def mesh_table(context: InferenceContext, archive: dict[str, np.ndarray], record: PairRecord) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prediction = infer_pair_predictions(context, archive, [record])
    vertices = {name: context.geometry.vertices(values)[0] for name, values in prediction.items()}
    normal = vertex_normals(vertices["source"], context.geometry.faces)
    actual = np.sum((vertices["target"] - vertices["source"]) * normal, axis=1) / record.gap_years
    table: dict[str, Any] = {
        "vertex_id": np.arange(len(vertices["source"])),
        "source_x": vertices["source"][:, 0], "source_y": vertices["source"][:, 1], "source_z": vertices["source"][:, 2],
        "target_x": vertices["target"][:, 0], "target_y": vertices["target"][:, 1], "target_z": vertices["target"][:, 2],
        "observed_normal_change_rate": actual,
    }
    summary: list[dict[str, Any]] = []
    for model in MODEL_LABELS:
        error = np.linalg.norm(vertices[model] - vertices["target"], axis=1)
        rate = np.sum((vertices[model] - vertices["source"]) * normal, axis=1) / record.gap_years
        table |= {f"{model}_x": vertices[model][:, 0], f"{model}_y": vertices[model][:, 1], f"{model}_z": vertices[model][:, 2], f"{model}_endpoint_error": error, f"{model}_normal_change_rate": rate}
        summary.append({"diagnosis": record.diagnosis, "subject_id": record.subject_id, "gap_years": record.gap_years, "model": model, "model_label": MODEL_LABELS[model], "vertex_error_mean": float(error.mean()), "normal_change_mae": float(np.abs(rate - actual).mean()), "normal_change_pearson": pearson(rate, actual), "hotspot_dice": top_change_dice(rate, actual, 0.20)})
    return pd.DataFrame(table), summary


def group_maps(context: InferenceContext, records: dict[str, list[PairRecord]], splits: Iterable[str], maximum: int, top_fraction: float) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for diagnosis in ("CN", "AD"):
        items = [(split, record) for split in splits for record in records[split] if record.diagnosis == diagnosis]
        if maximum and len(items) > maximum:
            take = np.linspace(0, len(items) - 1, maximum).round().astype(int)
            items = [items[index] for index in take]
        accumulator: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "observed": None, "brainode": None, "cocycle_flow": None})
        for position, (split, record) in enumerate(items, start=1):
            prediction = infer_pair_predictions(context, context.archives[split], [record])
            vertices = {name: context.geometry.vertices(value)[0] for name, value in prediction.items()}
            normal = vertex_normals(vertices["source"], context.geometry.faces)
            changes = {name: np.sum((vertices[name] - vertices["source"]) * normal, axis=1) / record.gap_years for name in ("target", "brainode", "cocycle_flow")}
            key = f"{split}:{record.subject_id}"
            entry = accumulator[key]
            entry["count"] += 1
            for name, value in (("observed", changes["target"]), ("brainode", changes["brainode"]), ("cocycle_flow", changes["cocycle_flow"])):
                entry[name] = value.astype(np.float64) if entry[name] is None else entry[name] + value
            if position % 50 == 0 or position == len(items):
                print(f"{context.structure}: {diagnosis} group map {position}/{len(items)} pairs", flush=True)
        maps = {name: np.mean(np.stack([entry[name] / entry["count"] for entry in accumulator.values()]), axis=0) for name in ("observed", "brainode", "cocycle_flow")}
        template = context.geometry.vertices(np.zeros((1, 150), dtype=np.float64))[0]
        table = {"vertex_id": np.arange(len(template)), "template_x": template[:, 0], "template_y": template[:, 1], "template_z": template[:, 2], "observed_normal_change_rate": maps["observed"]}
        for model in MODEL_LABELS:
            table[f"{model}_normal_change_rate"] = maps[model]
            rows.append({"diagnosis": diagnosis, "model": model, "model_label": MODEL_LABELS[model], "pairs": len(items), "subjects": len(accumulator), "aggregation": "mean within subject, then mean across subjects", "normal_change_mae": float(np.abs(maps[model] - maps["observed"]).mean()), "normal_change_pearson": pearson(maps[model], maps["observed"]), "hotspot_dice": top_change_dice(maps[model], maps["observed"], top_fraction), "top_fraction": top_fraction})
        tables[diagnosis] = pd.DataFrame(table)
    return tables, pd.DataFrame(rows)


def expected_files(output_dir: Path) -> list[Path]:
    files = [output_dir / "manifest.json", output_dir / "provenance.json", output_dir / "selected_examples.json"]
    files.extend(output_dir / "tables" / name for name in REQUIRED_TABLES)
    files.extend(output_dir / "meshes" / name for name in REQUIRED_MESHES)
    return files


def complete_cache_exists(output_dir: Path, provenance: dict[str, Any]) -> bool:
    manifest_path, provenance_path = output_dir / "manifest.json", output_dir / "provenance.json"
    if not manifest_path.is_file() or not provenance_path.is_file():
        return False
    return json.loads(manifest_path.read_text()).get("status") == "complete" and json.loads(provenance_path.read_text()) == provenance and all(path.is_file() for path in expected_files(output_dir))


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap_replicates < 1:
        raise ValueError("batch size and bootstrap replicates must be positive")
    if not 0.0 < args.top_change_fraction < 1.0:
        raise ValueError("top-change fraction must be in (0, 1)")
    if args.example_split not in args.splits:
        raise ValueError("example split must be included in --splits")
    device = choose_device(args.device)
    context, provenance = load_context(args.structure, device, args.splits)
    flow_config = read_json(selected_paths(args.structure)["flow_config"])
    input_config = read_json(resolve_path(flow_config["input_config"]))
    records = {split: records_for_split(split, context.archives[split], resolve_path(input_config["dataset"][f"{split}_pairs"])) for split in args.splits}
    print(f"PCA Cocycle Flow vs BrainODE | {args.structure} | device={device}", flush=True)
    print({split: len(value) for split, value in records.items()}, flush=True)
    if args.dry_run:
        split = next(iter(args.splits))
        row = metric_rows(context, context.archives[split], records[split][:1], 1, args.top_change_fraction)[0]
        if not all(math.isfinite(float(row[name])) for name in METRICS):
            raise RuntimeError("Dry-run metric audit failed")
        print(json.dumps({"status": "dry-run-pass", "structure": args.structure, "split": split, "pair": {key: row[key] for key in ("subject_id", "diagnosis", "gap_years")}, "source_meshes_modified": False}, indent=2), flush=True)
        return 0
    output_dir = args.output_dir or REPORT_ROOT / "prepared_cache" / args.structure
    output_dir = output_dir if output_dir.is_absolute() else (Path.cwd() / output_dir).resolve()
    provenance |= {"schema_version": SCHEMA_VERSION, "structure": args.structure, "splits": list(args.splits), "example_split": args.example_split, "seed": args.seed, "bootstrap_replicates": args.bootstrap_replicates, "top_change_fraction": args.top_change_fraction, "group_map_max_pairs_per_diagnosis": args.group_map_max_pairs_per_diagnosis, "skip_group_maps": args.skip_group_maps}
    if complete_cache_exists(output_dir, provenance):
        print(f"Complete matching cache already exists: {output_dir}", flush=True)
        return 0
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a partial or non-matching cache: {output_dir}")
    (output_dir / "tables").mkdir(parents=True, exist_ok=False)
    (output_dir / "meshes").mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "provenance.json", provenance)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "building", "created_at": utc_now(), "updated_at": utc_now(), "structure": args.structure, "source_meshes_modified": False, "required_files": [str(path.relative_to(output_dir)) for path in expected_files(output_dir) if path.name != "manifest.json"]}
    write_json(output_dir / "manifest.json", manifest)
    per_pair_rows = [row for split in args.splits for row in metric_rows(context, context.archives[split], records[split], args.batch_size, args.top_change_fraction)]
    per_pair = pd.DataFrame(per_pair_rows)
    paired, summary = paired_tables(per_pair)
    bootstrap = bootstrap_table(paired, args.bootstrap_replicates, args.seed)
    counts = add_all_split(per_pair).groupby(["split", "diagnosis", "model", "model_label"], as_index=False).agg(pairs=("subject_id", "size"), subjects=("subject_id", "nunique"))
    gaps = per_pair.groupby(["split", "diagnosis", "gap_bin", "model", "model_label"], as_index=False)[list(METRICS)].mean()
    trends, trend_summary, curves = trend_tables(per_pair)
    selected = select_examples(per_pair, args.example_split)
    write_frame(output_dir / "tables" / "cohort_counts.csv", counts)
    write_frame(output_dir / "tables" / "per_pair_metrics.csv", per_pair)
    write_frame(output_dir / "tables" / "metric_summary.csv", summary)
    write_frame(output_dir / "tables" / "paired_metric_comparison.csv", paired)
    write_frame(output_dir / "tables" / "paired_bootstrap_ci.csv", bootstrap)
    write_frame(output_dir / "tables" / "gap_summary.csv", gaps)
    write_frame(output_dir / "tables" / "subject_volume_trends.csv", trends)
    write_frame(output_dir / "tables" / "cohort_volume_trend_summary.csv", trend_summary)
    write_frame(output_dir / "tables" / "cohort_volume_curve.csv", curves)
    write_frame(output_dir / "tables" / "selected_volume_trajectories.csv", selected_trajectories(per_pair, selected))
    write_json(output_dir / "selected_examples.json", selected)
    manifest["scalar_tables"] = {"status": "complete", "pairs": int(len(per_pair)), "paired_metric_rows": int(len(paired)), "completed_at": utc_now()}
    manifest["updated_at"] = utc_now()
    write_json(output_dir / "manifest.json", manifest)
    write_frame(output_dir / "tables" / "counterfactual_trajectories.csv", counterfactual_table(context, records, selected))
    selected_metrics: list[dict[str, Any]] = []
    for diagnosis, info in selected.items():
        examples = matching_baseline_records(records[info["split"]], info["subject_id"])
        table, metrics = mesh_table(context, context.archives[info["split"]], max(examples, key=lambda item: (item.gap_years, item.target_visit_order)))
        write_frame(output_dir / "meshes" / f"selected_{diagnosis.lower()}_vertices.csv", table)
        selected_metrics.extend(metrics)
    write_frame(output_dir / "tables" / "selected_mesh_metrics.csv", pd.DataFrame(selected_metrics))
    face = context.geometry.faces
    write_frame(output_dir / "meshes" / "template_faces.csv", pd.DataFrame({"face_id": np.arange(len(face)), "vertex_0": face[:, 0], "vertex_1": face[:, 1], "vertex_2": face[:, 2]}))
    if args.skip_group_maps:
        blank = pd.DataFrame(columns=["vertex_id", "template_x", "template_y", "template_z", "observed_normal_change_rate", "brainode_normal_change_rate", "cocycle_flow_normal_change_rate"])
        write_frame(output_dir / "meshes" / "group_cn_change_maps.csv", blank)
        write_frame(output_dir / "meshes" / "group_ad_change_maps.csv", blank)
        write_frame(output_dir / "tables" / "group_hotspot_metrics.csv", pd.DataFrame(columns=["diagnosis", "model", "model_label", "pairs", "subjects", "aggregation", "normal_change_mae", "normal_change_pearson", "hotspot_dice", "top_fraction"]))
        group_status = "skipped"
    else:
        group_table, group_metrics = group_maps(context, records, args.splits, args.group_map_max_pairs_per_diagnosis, args.top_change_fraction)
        write_frame(output_dir / "meshes" / "group_cn_change_maps.csv", group_table["CN"])
        write_frame(output_dir / "meshes" / "group_ad_change_maps.csv", group_table["AD"])
        write_frame(output_dir / "tables" / "group_hotspot_metrics.csv", group_metrics)
        group_status = "complete"
    manifest |= {"status": "complete", "completed_at": utc_now(), "updated_at": utc_now(), "selected_examples": selected, "group_maps": group_status, "rows": {"per_pair": int(len(per_pair)), "subject_trends": int(len(trends))}}
    write_json(output_dir / "manifest.json", manifest)
    missing = [str(path) for path in expected_files(output_dir) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete cache: {missing}")
    print(f"Prepared complete cache: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
