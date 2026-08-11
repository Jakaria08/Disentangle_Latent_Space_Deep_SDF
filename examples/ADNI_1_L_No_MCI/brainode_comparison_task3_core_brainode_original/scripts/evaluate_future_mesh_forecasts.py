#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/future_mesh_forecast_comparison"
)
DEFAULT_SPLITS = ("train", "val", "test")

OLD_HELPER_DIR = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI"
for import_path in (REPO_ROOT, OLD_HELPER_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import direct_flow_rich_notebook_helpers as flow_helpers  # noqa: E402


FLOW_MODELS = {
    "old_pca32_siren": {
        "dataset": "old_adni",
        "family": "SIREN PCA32 flow",
        "experiment_dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_pca32_siren_optimized",
    },
    "old_smallnet_deepsdf": {
        "dataset": "old_adni",
        "family": "DeepSDF small-net flow",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI/"
            "longitudinal_direct_full256_smallnet_deepsdf_optimized"
        ),
    },
    "old_siren_full256_direct": {
        "dataset": "old_adni",
        "family": "SIREN full256 direct flow",
        "experiment_dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized",
    },
    "qc_siren_drop_bad_min2": {
        "dataset": "qc_large",
        "family": "SIREN QC direct/composed",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2"
        ),
    },
    "qc_siren_local_decomp_volume": {
        "dataset": "qc_large",
        "family": "SIREN QC local decomposed volume",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_local_decomposed_flow_qc_volume_v1"
        ),
    },
    "qc_siren_latent_ode": {
        "dataset": "qc_large",
        "family": "SIREN QC latent ODE",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1"
        ),
    },
}

BRAINODE_MODELS = {
    "old_brainode_pca150": {
        "dataset": "old_adni",
        "family": "BrainODE PCA150",
        "task_dir": "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original",
    },
    "qc_brainode_pca150": {
        "dataset": "qc_large",
        "family": "BrainODE PCA150",
        "task_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "brainode_pca150_qc_stable"
        ),
    },
}

LOWER_IS_BETTER_METRICS = (
    "gt_to_pred_mean",
    "pred_to_gt_mean",
    "chamfer_l1",
    "chamfer_l2_squared",
    "assd",
    "hd95",
    "volume_abs_error",
    "volume_relative_error",
    "surface_area_abs_error",
    "surface_area_relative_error",
)


@dataclass(frozen=True)
class PairSpec:
    dataset: str
    split: str
    subject_id: str
    diagnosis: str
    label_ad: int
    source_scan_id: str
    target_scan_id: str
    source_visit_order: int
    target_visit_order: int
    source_age_norm: float
    target_age_norm: float
    source_age_years: float
    target_age_years: float
    source_mesh_path: str
    target_mesh_path: str

    @property
    def pair_type(self) -> str:
        return "adjacent" if self.target_visit_order - self.source_visit_order == 1 else "nonadjacent"

    @property
    def gap_years(self) -> float:
        return float(self.target_age_years - self.source_age_years)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode real future forecast meshes for BrainODE and SIREN/DeepSDF flow "
            "models, then compare every predicted endpoint to the true target mesh "
            "using sampled-surface Chamfer/ASSD/HD95/volume/surface-area metrics."
        )
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model names, or all, brainode, flow. Defaults to all configured BrainODE "
            "and flow models."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=DEFAULT_SPLITS)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--surface-samples", type=int, default=30000)
    parser.add_argument("--mesh-resolution", type=int, default=80)
    parser.add_argument("--mesh-max-batch", type=int, default=2**18)
    parser.add_argument("--composed-step-years", type=float, default=0.5)
    parser.add_argument(
        "--max-pairs-per-split",
        type=int,
        default=0,
        help="Limit pairs per model/split for a smoke run. Use 0 for all pairs.",
    )
    parser.add_argument(
        "--selected-subjects-csv",
        default="",
        help=(
            "Optional selected_subject_manifest.csv. When provided, only these "
            "dataset/split/subject IDs are used for mesh/trend comparison."
        ),
    )
    parser.add_argument(
        "--balanced-subject-selection",
        action="store_true",
        help=(
            "Without --selected-subjects-csv, select up to --subjects-per-diagnosis "
            "AD and CN subjects per dataset/split before evaluating pairs."
        ),
    )
    parser.add_argument("--subjects-per-diagnosis", type=int, default=25)
    parser.add_argument(
        "--selected-pair-source-mode",
        choices=("baseline_only", "all_sources"),
        default="all_sources",
        help=(
            "For selected subjects, evaluate baseline-to-future pairs only or all "
            "source-to-future pairs."
        ),
    )
    parser.add_argument(
        "--pair-type",
        choices=("all", "adjacent", "nonadjacent"),
        default="all",
    )
    parser.add_argument(
        "--save-meshes",
        action="store_true",
        help="Also export predicted meshes. Metrics are computed even when meshes are not saved.",
    )
    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Evaluate BrainODE only. Useful on CPU because SDF mesh decoding requires CUDA.",
    )
    parser.add_argument(
        "--skip-brainode",
        action="store_true",
        help="Evaluate SIREN/DeepSDF flow models only.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def selected_models(args: argparse.Namespace) -> list[str]:
    requested = list(args.models)
    if requested == ["all"]:
        names = [*BRAINODE_MODELS.keys(), *FLOW_MODELS.keys()]
    elif requested == ["brainode"]:
        names = list(BRAINODE_MODELS.keys())
    elif requested == ["flow"]:
        names = list(FLOW_MODELS.keys())
    else:
        names = requested

    if args.skip_brainode:
        names = [name for name in names if name not in BRAINODE_MODELS]
    if args.skip_flow:
        names = [name for name in names if name not in FLOW_MODELS]

    unknown = sorted(set(names) - set(BRAINODE_MODELS) - set(FLOW_MODELS))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return names


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_selected_subjects(path: str | Path) -> dict[tuple[str, str], set[str]]:
    if not path:
        return {}
    rows = read_csv(repo_path(path))
    selected: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).strip()
        split = str(row.get("split", "")).strip()
        subject_id = str(row.get("subject_id", "")).strip()
        if not dataset or not split or not subject_id:
            continue
        selected.setdefault((dataset, split), set()).add(subject_id)
    return selected


def select_balanced_subjects_from_stats(stats: Sequence[dict[str, Any]], subjects_per_diagnosis: int) -> set[str]:
    if not stats:
        return set()
    frame = pd.DataFrame(stats)
    selected: set[str] = set()
    for diagnosis in ("AD", "CN"):
        subset = frame.loc[frame["diagnosis"].astype(str) == diagnosis].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["candidate_targets", "candidate_pairs", "max_gap_years", "subject_id"],
            ascending=[False, False, False, True],
        )
        selected.update(subset.head(int(subjects_per_diagnosis))["subject_id"].astype(str).tolist())
    return selected


def balanced_subjects_from_archive(
    archive: dict[str, np.ndarray],
    subjects_per_diagnosis: int,
) -> set[str]:
    offsets = archive["subject_visit_offsets"]
    stats: list[dict[str, Any]] = []
    for subject_index in range(len(offsets) - 1):
        start = int(offsets[subject_index])
        end = int(offsets[subject_index + 1])
        if end - start < 2:
            continue
        ages = archive["visit_continuous_age_years"][start:end].astype(float)
        subject_id = str(archive["visit_subject_ids"][start])
        stats.append(
            {
                "subject_id": subject_id,
                "diagnosis": str(archive["visit_diagnoses"][start]),
                "candidate_pairs": int((end - start) * (end - start - 1) / 2),
                "candidate_targets": int(end - start - 1),
                "max_gap_years": float(np.max(ages) - np.min(ages)) if len(ages) else float("nan"),
            }
        )
    return select_balanced_subjects_from_stats(stats, subjects_per_diagnosis)


def balanced_subjects_from_pair_frame(
    frame: pd.DataFrame,
    subjects_per_diagnosis: int,
) -> set[str]:
    if frame.empty or not {"subject_id", "diagnosis"}.issubset(frame.columns):
        return set()
    working = frame.copy()
    for column in ["subject_id", "diagnosis"]:
        working[column] = working[column].astype(str)
    for column in ["source_age_years", "target_age_years", "gap_years"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    stats: list[dict[str, Any]] = []
    for (subject_id, diagnosis), group in working.groupby(["subject_id", "diagnosis"], sort=True):
        ages = pd.concat(
            [
                group["source_age_years"] if "source_age_years" in group else pd.Series(dtype=float),
                group["target_age_years"] if "target_age_years" in group else pd.Series(dtype=float),
            ],
            ignore_index=True,
        ).dropna()
        stats.append(
            {
                "subject_id": str(subject_id),
                "diagnosis": str(diagnosis),
                "candidate_pairs": int(len(group)),
                "candidate_targets": int(group["target_scan_id"].astype(str).nunique())
                if "target_scan_id" in group
                else 0,
                "max_gap_years": float(group["gap_years"].max())
                if "gap_years" in group and group["gap_years"].notna().any()
                else (float(ages.max() - ages.min()) if len(ages) else float("nan")),
            }
        )
    return select_balanced_subjects_from_stats(stats, subjects_per_diagnosis)


def configured_subjects_for_split(
    args: argparse.Namespace,
    *,
    dataset: str,
    split: str,
) -> tuple[set[str], bool]:
    selected = getattr(args, "_selected_subjects_by_dataset_split", {})
    if selected:
        return set(selected.get((dataset, split), set())), True
    return set(), bool(args.balanced_subject_selection)


def filter_pair_frame_for_subject_selection(
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    dataset: str,
    split: str,
) -> tuple[pd.DataFrame, set[str], bool]:
    selected_subjects, enabled = configured_subjects_for_split(args, dataset=dataset, split=split)
    if enabled and not selected_subjects and bool(args.balanced_subject_selection):
        selected_subjects = balanced_subjects_from_pair_frame(frame, int(args.subjects_per_diagnosis))
    working = frame.copy()
    if enabled:
        working = working.loc[working["subject_id"].astype(str).isin(selected_subjects)].copy()
    if str(args.selected_pair_source_mode) == "baseline_only" and "source_visit_order" in working.columns:
        source_order = pd.to_numeric(working["source_visit_order"], errors="coerce")
        working = working.assign(_source_visit_order_numeric=source_order)
        min_source = working.groupby("subject_id")["_source_visit_order_numeric"].transform("min")
        working = working.loc[working["_source_visit_order_numeric"] == min_source].drop(
            columns=["_source_visit_order_numeric"]
        )
    return working, selected_subjects, enabled


def deterministic_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def safe_stem(value: str, max_chars: int = 36) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned[:max_chars].strip("_") or "scan"


@lru_cache(maxsize=1024)
def load_mesh_from_path(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Empty mesh scene at {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh at {path}, got {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Empty mesh at {path}")
    return loaded


def load_mesh(mesh_like: str | Path | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like
    return load_mesh_from_path(str(mesh_like))


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def finite_relative(abs_error: float, target_value: float) -> float:
    if not math.isfinite(abs_error) or not math.isfinite(target_value) or abs(target_value) <= 0.0:
        return float("nan")
    return float(abs_error / abs(target_value))


def empty_geometry_metrics(error: str) -> dict[str, Any]:
    return {
        "decode_success": False,
        "decode_error": error,
        "gt_to_pred_mean": float("nan"),
        "pred_to_gt_mean": float("nan"),
        "chamfer_l1": float("nan"),
        "chamfer_l2_squared": float("nan"),
        "assd": float("nan"),
        "hd95": float("nan"),
        "predicted_volume": float("nan"),
        "target_volume": float("nan"),
        "volume_abs_error": float("nan"),
        "volume_signed_error": float("nan"),
        "volume_relative_error": float("nan"),
        "predicted_surface_area": float("nan"),
        "target_surface_area": float("nan"),
        "surface_area_abs_error": float("nan"),
        "surface_area_signed_error": float("nan"),
        "surface_area_relative_error": float("nan"),
    }


def geometry_metrics(
    *,
    predicted_mesh: trimesh.Trimesh | None,
    target_mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    sample_count: int,
    seed_key: str,
    decode_error: str | None = None,
) -> dict[str, Any]:
    if predicted_mesh is None:
        return empty_geometry_metrics(decode_error or "prediction mesh is missing")

    pred_points = sample_surface(
        predicted_mesh,
        sample_count,
        deterministic_seed(seed_key, "predicted_surface_points"),
    )
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    gt_to_pred, _ = pred_tree.query(target_points, k=1)
    pred_to_gt, _ = target_tree.query(pred_points, k=1)
    gt_mean = float(np.mean(gt_to_pred))
    pred_mean = float(np.mean(pred_to_gt))
    all_distances = np.concatenate([gt_to_pred, pred_to_gt])

    predicted_volume = float(abs(predicted_mesh.volume))
    target_volume = float(abs(target_mesh.volume))
    volume_signed = predicted_volume - target_volume
    volume_abs = abs(volume_signed)
    predicted_area = float(predicted_mesh.area)
    target_area = float(target_mesh.area)
    area_signed = predicted_area - target_area
    area_abs = abs(area_signed)

    return {
        "decode_success": True,
        "decode_error": "",
        "gt_to_pred_mean": gt_mean,
        "pred_to_gt_mean": pred_mean,
        "chamfer_l1": gt_mean + pred_mean,
        "chamfer_l2_squared": float(np.mean(gt_to_pred**2) + np.mean(pred_to_gt**2)),
        "assd": 0.5 * (gt_mean + pred_mean),
        "hd95": float(np.quantile(all_distances, 0.95)),
        "predicted_volume": predicted_volume,
        "target_volume": target_volume,
        "volume_abs_error": volume_abs,
        "volume_signed_error": volume_signed,
        "volume_relative_error": finite_relative(volume_abs, target_volume),
        "predicted_surface_area": predicted_area,
        "target_surface_area": target_area,
        "surface_area_abs_error": area_abs,
        "surface_area_signed_error": area_signed,
        "surface_area_relative_error": finite_relative(area_abs, target_area),
    }


def row_uid(parts: Iterable[object]) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def export_mesh(
    *,
    output_dir: Path,
    save_meshes: bool,
    mesh: trimesh.Trimesh | None,
    dataset: str,
    model_name: str,
    split: str,
    transport_method: str,
    source_scan_id: str,
    target_scan_id: str,
    uid: str,
) -> str:
    if not save_meshes or mesh is None:
        return ""
    mesh_dir = output_dir / "predicted_meshes" / dataset / model_name / split / transport_method
    mesh_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{safe_stem(source_scan_id)}__to__{safe_stem(target_scan_id)}__{uid}"
    )
    path = mesh_dir / f"{stem}.ply"
    mesh.export(path)
    return str(path)


def make_prediction_row(
    *,
    pair: PairSpec,
    model_name: str,
    family: str,
    transport_method: str,
    predicted_mesh: trimesh.Trimesh | None,
    target_mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    source_gt_metrics: dict[str, Any],
    model_no_change_metrics: dict[str, Any],
    sample_count: int,
    output_dir: Path,
    save_meshes: bool,
    decode_error: str | None = None,
    checkpoint: str = "",
    precomputed_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    uid = row_uid(
        (
            pair.dataset,
            model_name,
            pair.split,
            transport_method,
            pair.source_scan_id,
            pair.target_scan_id,
        )
    )
    metrics = (
        dict(precomputed_metrics)
        if precomputed_metrics is not None
        else geometry_metrics(
            predicted_mesh=predicted_mesh,
            target_mesh=target_mesh,
            target_points=target_points,
            sample_count=sample_count,
            seed_key=f"{uid}|{transport_method}",
            decode_error=decode_error,
        )
    )
    row: dict[str, Any] = {
        "row_uid": uid,
        "dataset": pair.dataset,
        "model": model_name,
        "family": family,
        "split": pair.split,
        "subject_id": pair.subject_id,
        "diagnosis": pair.diagnosis,
        "label_ad": pair.label_ad,
        "source_scan_id": pair.source_scan_id,
        "target_scan_id": pair.target_scan_id,
        "source_visit_order": pair.source_visit_order,
        "target_visit_order": pair.target_visit_order,
        "pair_type": pair.pair_type,
        "source_age_years": pair.source_age_years,
        "target_age_years": pair.target_age_years,
        "gap_years": pair.gap_years,
        "source_age_norm": pair.source_age_norm,
        "target_age_norm": pair.target_age_norm,
        "source_mesh_path": pair.source_mesh_path,
        "target_mesh_path": pair.target_mesh_path,
        "transport_method": transport_method,
        "sampled_surface_points": sample_count,
        "checkpoint": checkpoint,
    }
    row.update(metrics)
    row["predicted_mesh_path"] = export_mesh(
        output_dir=output_dir,
        save_meshes=save_meshes,
        mesh=predicted_mesh,
        dataset=pair.dataset,
        model_name=model_name,
        split=pair.split,
        transport_method=transport_method,
        source_scan_id=pair.source_scan_id,
        target_scan_id=pair.target_scan_id,
        uid=uid,
    )
    for metric in LOWER_IS_BETTER_METRICS:
        source_value = source_gt_metrics.get(metric, float("nan"))
        model_value = model_no_change_metrics.get(metric, float("nan"))
        value = metrics.get(metric, float("nan"))
        row[f"source_gt_no_change_{metric}"] = source_value
        row[f"model_no_change_{metric}"] = model_value
        row[f"{metric}_improvement_vs_source_gt_no_change"] = (
            float(source_value) - float(value)
            if math.isfinite(float(source_value)) and math.isfinite(float(value))
            else float("nan")
        )
        row[f"{metric}_improvement_vs_model_no_change"] = (
            float(model_value) - float(value)
            if math.isfinite(float(model_value)) and math.isfinite(float(value))
            else float("nan")
        )
    return row


def import_brainode_modules(task_dir: Path) -> dict[str, Any]:
    scripts_dir = task_dir / "scripts"
    if not scripts_dir.is_dir():
        raise FileNotFoundError(f"Missing BrainODE scripts dir: {scripts_dir}")
    original_path = list(sys.path)
    for module_name in ("brainode_model", "core_brainode_common", "train_core_brainode"):
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(scripts_dir))
    try:
        return {
            "brainode_model": importlib.import_module("brainode_model"),
            "common": importlib.import_module("core_brainode_common"),
            "train": importlib.import_module("train_core_brainode"),
        }
    finally:
        sys.path[:] = original_path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def resolve_device(value: str | None) -> torch.device:
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_brainode_checkpoint(
    *,
    task_dir: Path,
    config: dict[str, Any],
    common: Any,
    checkpoint: str,
    run_name: str,
) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.suffix == ".pth":
        resolved = common.resolve_repo_path(path)
        if resolved.is_file():
            return resolved
    name = checkpoint if checkpoint.endswith(".pth") else f"{checkpoint}.pth"
    output_root = common.resolve_repo_path(config["training"]["output_root"])
    candidate = output_root / run_name / "checkpoints" / name
    if candidate.is_file():
        return candidate
    fallback = task_dir / "training" / run_name / "checkpoints" / name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing BrainODE checkpoint {checkpoint!r} for {task_dir}")


def pca_mesh(coefficients: np.ndarray, mean_flat: np.ndarray, components: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    flat = coefficients.astype(np.float32) @ components.astype(np.float32) + mean_flat.astype(np.float32)
    vertices = flat.reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def iter_brainode_pairs(
    *,
    dataset_name: str,
    split: str,
    archive: dict[str, np.ndarray],
    scan_manifest: dict[str, dict[str, str]],
    pair_type: str,
    max_pairs: int,
    selected_subject_ids: set[str] | None = None,
    selected_pair_source_mode: str = "all_sources",
) -> Iterable[tuple[PairSpec, np.ndarray, np.ndarray, np.ndarray]]:
    offsets = archive["subject_visit_offsets"]
    yielded = 0
    for subject_index in range(len(offsets) - 1):
        start = int(offsets[subject_index])
        end = int(offsets[subject_index + 1])
        subject_id = str(archive["visit_subject_ids"][start])
        if selected_subject_ids is not None and subject_id not in selected_subject_ids:
            continue
        for source_index in range(start, end - 1):
            if selected_pair_source_mode == "baseline_only" and source_index != start:
                continue
            for target_index in range(source_index + 1, end):
                source_order = int(archive["visit_orders"][source_index])
                target_order = int(archive["visit_orders"][target_index])
                current_pair_type = "adjacent" if target_order - source_order == 1 else "nonadjacent"
                if pair_type != "all" and current_pair_type != pair_type:
                    continue
                source_scan_id = str(archive["visit_scan_ids"][source_index])
                target_scan_id = str(archive["visit_scan_ids"][target_index])
                source_manifest = scan_manifest[source_scan_id]
                target_manifest = scan_manifest[target_scan_id]
                time_slice = slice(source_index, target_index + 1)
                pair = PairSpec(
                    dataset=dataset_name,
                    split=split,
                    subject_id=subject_id,
                    diagnosis=str(archive["visit_diagnoses"][source_index]),
                    label_ad=int(archive["visit_label_ad"][source_index]),
                    source_scan_id=source_scan_id,
                    target_scan_id=target_scan_id,
                    source_visit_order=source_order,
                    target_visit_order=target_order,
                    source_age_norm=float(archive["visit_continuous_age_norm"][source_index]),
                    target_age_norm=float(archive["visit_continuous_age_norm"][target_index]),
                    source_age_years=float(archive["visit_continuous_age_years"][source_index]),
                    target_age_years=float(archive["visit_continuous_age_years"][target_index]),
                    source_mesh_path=source_manifest["mesh_path"],
                    target_mesh_path=target_manifest["mesh_path"],
                )
                yield (
                    pair,
                    archive["visit_continuous_age_norm"][time_slice].astype(np.float32).copy(),
                    archive["visit_pca_150"][time_slice].astype(np.float32).copy(),
                    archive["visit_cognition"][time_slice].astype(np.float32).copy(),
                )
                yielded += 1
                if max_pairs > 0 and yielded >= max_pairs:
                    return


@torch.no_grad()
def evaluate_brainode_model(
    *,
    model_name: str,
    spec: dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    dataset_name = spec["dataset"]
    task_dir = repo_path(spec["task_dir"])
    modules = import_brainode_modules(task_dir)
    common = modules["common"]
    train = modules["train"]
    brainode_model = modules["brainode_model"]

    config = common.load_config(task_dir / "configs" / "core_brainode.json")
    training_config = dict(config["training"])
    model_config = dict(config["model"])
    full_config = train.full_brainode_config(config)
    run_name = str(training_config["run_name"])
    device = resolve_device(args.device or training_config.get("device"))
    checkpoint_path = resolve_brainode_checkpoint(
        task_dir=task_dir,
        config=config,
        common=common,
        checkpoint=args.checkpoint,
        run_name=run_name,
    )

    train_archive = load_npz(task_dir / "dataset" / "train_subject_sequences.npz")
    latent_dim = int(train_archive["visit_pca_150"].shape[1])
    model = train.build_model(
        latent_dim=latent_dim,
        model_config=model_config,
        full_config=full_config,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    pca_model_dir = common.resolve_repo_path(config["task2"]["pca_model_dir"])
    mean_flat = np.load(pca_model_dir / "mean.npy").astype(np.float32)
    components = np.load(pca_model_dir / "components_256.npy").astype(np.float32)[:latent_dim]
    faces = np.load(pca_model_dir / "faces.npy").astype(np.int64)
    scan_manifest = {
        row["scan_id"]: row for row in read_csv(task_dir / "metadata" / "core_brainode_scan_manifest.csv")
    }

    rows: list[dict[str, Any]] = []
    for split in args.splits:
        archive = load_npz(task_dir / "dataset" / f"{split}_subject_sequences.npz")
        selected_subject_ids, selection_enabled = configured_subjects_for_split(
            args,
            dataset=dataset_name,
            split=split,
        )
        if selection_enabled and not selected_subject_ids and bool(args.balanced_subject_selection):
            selected_subject_ids = balanced_subjects_from_archive(
                archive,
                int(args.subjects_per_diagnosis),
            )
        print(
            "[future_mesh] "
            f"model={model_name} split={split} stage=brainode "
            f"subject_selection={'on' if selection_enabled else 'off'} "
            f"selected_subjects={len(selected_subject_ids) if selection_enabled else 'all'} "
            f"pair_source_mode={args.selected_pair_source_mode} pair_type={args.pair_type}",
            flush=True,
        )
        split_pairs = 0
        for pair, times_np, targets_np, conditions_np in iter_brainode_pairs(
            dataset_name=dataset_name,
            split=split,
            archive=archive,
            scan_manifest=scan_manifest,
            pair_type=args.pair_type,
            max_pairs=int(args.max_pairs_per_split),
            selected_subject_ids=selected_subject_ids if selection_enabled else None,
            selected_pair_source_mode=str(args.selected_pair_source_mode),
        ):
            split_pairs += 1
            source_gt_mesh = load_mesh(pair.source_mesh_path)
            target_mesh = load_mesh(pair.target_mesh_path)
            pair_seed = f"{pair.dataset}|{pair.split}|{pair.source_scan_id}|{pair.target_scan_id}"
            target_points = sample_surface(
                target_mesh,
                int(args.surface_samples),
                deterministic_seed(pair_seed, "target_surface_points"),
            )
            source_gt_metrics = geometry_metrics(
                predicted_mesh=source_gt_mesh,
                target_mesh=target_mesh,
                target_points=target_points,
                sample_count=int(args.surface_samples),
                seed_key=f"{pair_seed}|source_gt_no_change",
            )
            source_pca_mesh = pca_mesh(targets_np[0], mean_flat, components, faces)
            pca_no_change_metrics = geometry_metrics(
                predicted_mesh=source_pca_mesh,
                target_mesh=target_mesh,
                target_points=target_points,
                sample_count=int(args.surface_samples),
                seed_key=f"{pair_seed}|{model_name}|model_no_change",
            )

            times = torch.from_numpy(times_np).float().view(1, -1).to(device)
            targets = torch.from_numpy(targets_np).float().view(1, len(times_np), -1).to(device)
            condition = torch.tensor([float(conditions_np[0])], dtype=torch.float32, device=device)
            initial_state = targets[:, 0, :]
            if bool(full_config.get("use_autoregressive_rollout", False)):
                prediction, _ = brainode_model.integrate_autoregressive_rk4(
                    func=model,
                    initial_state=initial_state,
                    times=times,
                    initial_condition=condition,
                    substeps=int(training_config["integration_substeps"]),
                )
            else:
                prediction = brainode_model.integrate_sequence_rk4(
                    func=model,
                    initial_state=initial_state,
                    times=times,
                    condition=condition,
                    substeps=int(training_config["integration_substeps"]),
                )
            endpoint = prediction[0, -1, :].detach().cpu().numpy().astype(np.float32)
            endpoint_mesh = pca_mesh(endpoint, mean_flat, components, faces)

            rows.append(
                make_prediction_row(
                    pair=pair,
                    model_name=model_name,
                    family=spec["family"],
                    transport_method="brainode_endpoint",
                    predicted_mesh=endpoint_mesh,
                    target_mesh=target_mesh,
                    target_points=target_points,
                    source_gt_metrics=source_gt_metrics,
                    model_no_change_metrics=pca_no_change_metrics,
                    sample_count=int(args.surface_samples),
                    output_dir=output_dir,
                    save_meshes=bool(args.save_meshes),
                    checkpoint=str(checkpoint_path),
                )
            )
            rows.append(
                make_prediction_row(
                    pair=pair,
                    model_name=model_name,
                    family=spec["family"],
                    transport_method="model_no_change",
                    predicted_mesh=source_pca_mesh,
                    target_mesh=target_mesh,
                    target_points=target_points,
                    source_gt_metrics=source_gt_metrics,
                    model_no_change_metrics=pca_no_change_metrics,
                    sample_count=int(args.surface_samples),
                    output_dir=output_dir,
                    save_meshes=bool(args.save_meshes),
                    checkpoint=str(checkpoint_path),
                    precomputed_metrics=pca_no_change_metrics,
                )
            )
        print(
            "[future_mesh] "
            f"model={model_name} split={split} stage=brainode selected_pairs={split_pairs} "
            f"output_rows={split_pairs * 2}",
            flush=True,
        )
    return rows


def flow_pair_from_row(dataset_name: str, row: pd.Series, source_row: pd.Series, target_row: pd.Series) -> PairSpec:
    return PairSpec(
        dataset=dataset_name,
        split=str(row["split"]),
        subject_id=str(row["subject_id"]),
        diagnosis=str(row["diagnosis"]),
        label_ad=int(row["label_ad"]),
        source_scan_id=str(row["source_scan_id"]),
        target_scan_id=str(row["target_scan_id"]),
        source_visit_order=int(row["source_visit_order"]),
        target_visit_order=int(row["target_visit_order"]),
        source_age_norm=float(row["source_age_norm"]),
        target_age_norm=float(row["target_age_norm"]),
        source_age_years=float(row["source_age_years"]),
        target_age_years=float(row["target_age_years"]),
        source_mesh_path=str(source_row["mesh_path"]),
        target_mesh_path=str(target_row["mesh_path"]),
    )


@torch.no_grad()
def evaluate_flow_model(
    *,
    model_name: str,
    spec: dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    bundle = flow_helpers.load_bundle(
        repo_path(spec["experiment_dir"]),
        checkpoint=args.checkpoint,
        device=args.device,
    )
    source_decode_cache: dict[tuple[str, str], tuple[trimesh.Trimesh | None, str | None]] = {}
    rows: list[dict[str, Any]] = []

    for split in args.splits:
        pair_frame = flow_helpers.load_pair_metrics(bundle, split)
        if args.pair_type != "all":
            pair_frame = pair_frame.loc[pair_frame["pair_type"].astype(str) == args.pair_type].copy()
        pair_frame, selected_subject_ids, selection_enabled = filter_pair_frame_for_subject_selection(
            pair_frame,
            args=args,
            dataset=spec["dataset"],
            split=split,
        )
        if int(args.max_pairs_per_split) > 0:
            pair_frame = pair_frame.head(int(args.max_pairs_per_split)).copy()
        print(
            "[future_mesh] "
            f"model={model_name} split={split} stage=flow "
            f"subject_selection={'on' if selection_enabled else 'off'} "
            f"selected_subjects={len(selected_subject_ids) if selection_enabled else 'all'} "
            f"selected_pairs={len(pair_frame)} pair_source_mode={args.selected_pair_source_mode} "
            f"pair_type={args.pair_type}",
            flush=True,
        )

        split_pairs = 0
        for _, pair_row in pair_frame.iterrows():
            split_pairs += 1
            source_scan_id = str(pair_row["source_scan_id"])
            target_scan_id = str(pair_row["target_scan_id"])
            source_row = flow_helpers.scan_row(bundle, source_scan_id)
            target_row = flow_helpers.scan_row(bundle, target_scan_id)
            pair = flow_pair_from_row(spec["dataset"], pair_row, source_row, target_row)

            source_gt_mesh = load_mesh(pair.source_mesh_path)
            target_mesh = load_mesh(pair.target_mesh_path)
            pair_seed = f"{pair.dataset}|{pair.split}|{pair.source_scan_id}|{pair.target_scan_id}"
            target_points = sample_surface(
                target_mesh,
                int(args.surface_samples),
                deterministic_seed(pair_seed, "target_surface_points"),
            )
            source_gt_metrics = geometry_metrics(
                predicted_mesh=source_gt_mesh,
                target_mesh=target_mesh,
                target_points=target_points,
                sample_count=int(args.surface_samples),
                seed_key=f"{pair_seed}|source_gt_no_change",
            )

            source_latent = flow_helpers._latent_tensor(bundle, split, source_scan_id)
            cache_key = (split, source_scan_id)
            if cache_key not in source_decode_cache:
                source_decode_cache[cache_key] = flow_helpers.try_decode_mesh(
                    bundle,
                    source_latent,
                    resolution=int(args.mesh_resolution),
                    max_batch=int(args.mesh_max_batch),
                )
            model_no_change_mesh, model_no_change_error = source_decode_cache[cache_key]
            model_no_change_metrics = geometry_metrics(
                predicted_mesh=model_no_change_mesh,
                target_mesh=target_mesh,
                target_points=target_points,
                sample_count=int(args.surface_samples),
                seed_key=f"{pair_seed}|{model_name}|model_no_change",
                decode_error=model_no_change_error,
            )

            source_time = flow_helpers._time_tensor(bundle, pair.source_age_norm)
            target_time = flow_helpers._time_tensor(bundle, pair.target_age_norm)
            condition = flow_helpers._condition_tensor(bundle, pair.label_ad)
            direct_latent = flow_helpers.transport_direct(
                bundle,
                source_latent,
                source_time,
                target_time,
                condition,
            )
            direct_mesh, direct_error = flow_helpers.try_decode_mesh(
                bundle,
                direct_latent,
                resolution=int(args.mesh_resolution),
                max_batch=int(args.mesh_max_batch),
            )
            composed_latent = flow_helpers.transport_composed_fixed_step(
                bundle,
                source_latent,
                start_age_years=pair.source_age_years,
                end_age_years=pair.target_age_years,
                label_ad=pair.label_ad,
                step_years=float(args.composed_step_years),
            )
            composed_mesh, composed_error = flow_helpers.try_decode_mesh(
                bundle,
                composed_latent,
                resolution=int(args.mesh_resolution),
                max_batch=int(args.mesh_max_batch),
            )

            for method_name, mesh, error in (
                ("direct", direct_mesh, direct_error),
                ("composed", composed_mesh, composed_error),
                ("model_no_change", model_no_change_mesh, model_no_change_error),
            ):
                rows.append(
                    make_prediction_row(
                        pair=pair,
                        model_name=model_name,
                        family=spec["family"],
                        transport_method=method_name,
                        predicted_mesh=mesh,
                        target_mesh=target_mesh,
                        target_points=target_points,
                        source_gt_metrics=source_gt_metrics,
                        model_no_change_metrics=model_no_change_metrics,
                        sample_count=int(args.surface_samples),
                        output_dir=output_dir,
                        save_meshes=bool(args.save_meshes),
                        decode_error=error,
                        checkpoint=str(bundle.experiment_dir / "ModelParameters" / f"{args.checkpoint}.pth"),
                        precomputed_metrics=(
                            model_no_change_metrics if method_name == "model_no_change" else None
                        ),
                    )
                )
        print(
            "[future_mesh] "
            f"model={model_name} split={split} stage=flow selected_pairs={split_pairs} "
            f"output_rows={split_pairs * 3}",
            flush=True,
        )
    return rows


def finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else float("nan")


def finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if len(numeric) else float("nan")


def finite_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.std(ddof=0)) if len(numeric) else float("nan")


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    for column in [*LOWER_IS_BETTER_METRICS, *(f"{m}_improvement_vs_model_no_change" for m in LOWER_IS_BETTER_METRICS)]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    group_cols = [
        "dataset",
        "model",
        "family",
        "split",
        "diagnosis",
        "pair_type",
        "transport_method",
    ]
    summary: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        out = dict(zip(group_cols, key))
        out["rows"] = int(len(group))
        out["decode_success_fraction"] = float(group["decode_success"].astype(bool).mean())
        for metric in LOWER_IS_BETTER_METRICS:
            if metric not in group:
                continue
            out[f"{metric}_mean"] = finite_mean(group[metric])
            out[f"{metric}_median"] = finite_median(group[metric])
            out[f"{metric}_std"] = finite_std(group[metric])
            improvement = f"{metric}_improvement_vs_model_no_change"
            if improvement in group:
                out[f"{improvement}_mean"] = finite_mean(group[improvement])
        summary.append(out)
    return summary


def main() -> int:
    args = parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("--surface-samples must be positive")
    if int(args.mesh_resolution) <= 0:
        raise ValueError("--mesh-resolution must be positive")
    if float(args.composed_step_years) <= 0.0:
        raise ValueError("--composed-step-years must be positive")

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = selected_models(args)
    selected_subjects = load_selected_subjects(args.selected_subjects_csv)
    args._selected_subjects_by_dataset_split = selected_subjects
    if selected_subjects:
        selected_count = sum(len(values) for values in selected_subjects.values())
        print(
            "[future_mesh] "
            f"loaded_selected_subjects={selected_count} "
            f"groups={len(selected_subjects)} source={repo_path(args.selected_subjects_csv)}",
            flush=True,
        )
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for model_name in model_names:
        try:
            if model_name in BRAINODE_MODELS:
                rows = evaluate_brainode_model(
                    model_name=model_name,
                    spec=BRAINODE_MODELS[model_name],
                    args=args,
                    output_dir=output_dir,
                )
            else:
                rows = evaluate_flow_model(
                    model_name=model_name,
                    spec=FLOW_MODELS[model_name],
                    args=args,
                    output_dir=output_dir,
                )
            all_rows.extend(rows)
        except Exception as exc:
            failures.append(
                {
                    "model": model_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    summary_rows = summarize_rows(all_rows)
    exported_mesh_paths = sorted(
        {
            str(row.get("predicted_mesh_path", ""))
            for row in all_rows
            if str(row.get("predicted_mesh_path", "")).strip()
        }
    )
    write_csv(output_dir / "future_mesh_per_pair.csv", all_rows)
    write_csv(output_dir / "future_mesh_summary.csv", summary_rows)
    write_csv(output_dir / "future_mesh_failures.csv", failures)

    payload = {
        "output_dir": str(output_dir),
        "models_requested": model_names,
        "splits": list(args.splits),
        "pair_type": args.pair_type,
        "max_pairs_per_split": int(args.max_pairs_per_split),
        "selected_subjects_csv": str(repo_path(args.selected_subjects_csv)) if args.selected_subjects_csv else "",
        "selected_subject_groups": len(selected_subjects),
        "selected_subject_rows": int(sum(len(values) for values in selected_subjects.values())),
        "balanced_subject_selection": bool(args.balanced_subject_selection),
        "subjects_per_diagnosis": int(args.subjects_per_diagnosis),
        "selected_pair_source_mode": str(args.selected_pair_source_mode),
        "surface_samples": int(args.surface_samples),
        "mesh_resolution": int(args.mesh_resolution),
        "mesh_max_batch": int(args.mesh_max_batch),
        "composed_step_years": float(args.composed_step_years),
        "save_meshes": bool(args.save_meshes),
        "exported_predicted_meshes": int(len(exported_mesh_paths)),
        "internal_predicted_mesh_rows": int(len(all_rows)),
        "per_pair_rows": len(all_rows),
        "summary_rows": len(summary_rows),
        "failure_rows": len(failures),
        "note": (
            "This is the fair real-mesh future forecast comparison. BrainODE endpoint "
            "predictions are inverse-PCA meshes; SIREN/DeepSDF predictions are decoded "
            "SDF meshes. All rows compare to the same true target mesh with sampled "
            "surface points."
        ),
    }
    write_json(output_dir / "future_mesh_run.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
