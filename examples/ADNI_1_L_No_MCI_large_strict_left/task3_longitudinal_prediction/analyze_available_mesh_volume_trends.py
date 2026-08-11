#!/usr/bin/env python3
"""Analyze longitudinal volume trends from existing ADNI mesh artifacts.

The script performs two linked tests:

1. Test A measures longitudinal volume trends in the ground-truth meshes.
2. Test B checks whether independently reconstructed SIREN/DeepSDF scans
   preserve those trends.

No latent is fitted or modified.  The old ADNI reconstructions are read from
disk.  For the large cohort, the script reads either the existing 60-scan
no-skip reconstruction subset or a full mesh directory produced by the
separate multi-GPU decoding stage.  This script never loads a decoder.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "README.md").is_file()
            and (parent / "examples").is_dir()
            and (parent / "networks").is_dir()
        ):
            return parent
    raise RuntimeError("Could not locate the Deep3DComp repository root")


REPO_ROOT = find_repo_root()
LARGE_TASK2 = (
    REPO_ROOT
    / "examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations"
)
LARGE_FLOW = (
    REPO_ROOT
    / "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction"
    / "siren_no_skip_flow_real_sdf_observed_virtual_cocycle"
)
LARGE_QC = (
    REPO_ROOT
    / "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction"
    / "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2"
)
OLD_TASK1 = (
    REPO_ROOT
    / "examples/ADNI_1_L_No_MCI/brainode_comparison_task1_manifest_original"
)
OLD_TASK2 = (
    REPO_ROOT
    / "examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations_original"
)

LARGE_MANIFEST = (
    LARGE_TASK2 / "metadata/adni_large_strict_no_mci_left_manifest.csv"
)
LARGE_STABLE_MANIFEST = (
    LARGE_FLOW / "metadata/adni_large_strict_no_mci_left_direct_flow_records.csv"
)
LARGE_QC_MANIFEST = (
    LARGE_QC
    / "metadata/adni_large_strict_no_mci_left_direct_flow_records_qc_drop_bad_scans_min2.csv"
)
OLD_MANIFEST = OLD_TASK1 / "metadata/adni_no_mci_left_original_clean.csv"

LARGE_SIREN_DIR = (
    LARGE_TASK2 / "inr/siren_naisr_5x512_warmstart_no_skip"
)
OLD_SIREN_DIR = OLD_TASK2 / "inr/siren_naisr_enhanced"
OLD_DEEPSDF_DIR = OLD_TASK2 / "inr/deepsdf_eikonal_spec_fast"
AUDIT_ROOT = (
    Path(__file__).resolve().parent
    / "analysis/available_representation_volume_audit"
)
DEFAULT_FULL_LARGE_SIREN_MESH_DIR = LARGE_SIREN_DIR / "reconstructed_meshes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run subject-weighted ground-truth and frozen-representation "
            "volume-trend audits on the available large and old ADNI data."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(AUDIT_ROOT / "existing_only"),
        help="Directory for analysis CSV, JSON, and HTML outputs.",
    )
    parser.add_argument(
        "--mesh-cache",
        default=str(AUDIT_ROOT / "mesh_volume_cache.csv"),
        help=(
            "Shared persistent cache for volumes/topology of unchanged OBJ/PLY files."
        ),
    )
    parser.add_argument(
        "--large-siren-mesh-dir",
        default=str(DEFAULT_FULL_LARGE_SIREN_MESH_DIR),
        help=(
            "Full large no-skip SIREN mesh directory. If incomplete or absent, "
            "the existing 60-scan subset is still analyzed and coverage is reported."
        ),
    )
    parser.add_argument(
        "--refresh-mesh-cache",
        action="store_true",
        help="Re-read every mesh even if its path, size, and modification time match.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Subject-level bootstrap samples used for group confidence intervals.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for bootstrap intervals."
    )
    return parser.parse_args()


def require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing))


def read_manifest(path: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"scan_id": str, "subject_id": str})
    required = {
        "scan_id",
        "subject_id",
        "split",
        "diagnosis",
        "age_years",
        "mesh_path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    frame = frame.copy()
    frame["cohort"] = cohort
    frame["scan_id"] = frame["scan_id"].astype(str).map(lambda value: Path(value).stem)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["diagnosis"] = frame["diagnosis"].astype(str)
    frame["age_years"] = pd.to_numeric(frame["age_years"], errors="coerce")
    frame["visit_order"] = pd.to_numeric(
        frame.get("visit_order", 0), errors="coerce"
    ).fillna(0).astype(int)
    frame["months_from_baseline"] = pd.to_numeric(
        frame.get("months_from_baseline", np.nan), errors="coerce"
    )
    if "left_mesh_volume_mm3" in frame:
        frame["metadata_mesh_volume_mm3"] = pd.to_numeric(
            frame["left_mesh_volume_mm3"], errors="coerce"
        )
    else:
        frame["metadata_mesh_volume_mm3"] = np.nan
    if "left_mask_volume_mm3" in frame:
        frame["mask_volume_mm3"] = pd.to_numeric(
            frame["left_mask_volume_mm3"], errors="coerce"
        )
    else:
        frame["mask_volume_mm3"] = np.nan
    if frame["scan_id"].duplicated().any():
        duplicate = frame.loc[frame["scan_id"].duplicated(), "scan_id"].iloc[0]
        raise ValueError(f"Duplicate scan ID {duplicate!r} in {path}")
    return frame


@lru_cache(maxsize=64)
def load_obj_geometry(path_string: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_string)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for offset in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[offset], indices[offset + 1]])
    if not vertices or not faces:
        raise ValueError(f"Mesh has no vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def triangular_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    signed = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    ).sum() / 6.0
    return abs(float(signed))


def obj_volume(path: str | Path) -> float:
    vertices, faces = load_obj_geometry(str(Path(path).resolve()))
    return triangular_mesh_volume(vertices, faces)


def mesh_is_watertight(faces: np.ndarray) -> bool:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts) > 0 and np.all(counts == 2))


def inspect_mesh(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".obj":
        vertices, faces = load_obj_geometry(str(path.resolve()))
        return {
            "volume": triangular_mesh_volume(vertices, faces),
            "watertight": mesh_is_watertight(faces),
            "vertex_count": len(vertices),
            "face_count": len(faces),
        }
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to read reconstructed PLY meshes") from exc
    loaded = trimesh.load(path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return {
        "volume": abs(float(loaded.volume)),
        "watertight": bool(loaded.is_watertight),
        "vertex_count": int(len(loaded.vertices)),
        "face_count": int(len(loaded.faces)),
    }


MESH_CACHE_COLUMNS = [
    "mesh_path",
    "size_bytes",
    "mtime_ns",
    "status",
    "error",
    "volume",
    "watertight",
    "vertex_count",
    "face_count",
]


def load_mesh_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype={"mesh_path": str})
    return {
        str(row["mesh_path"]): row
        for row in frame.to_dict("records")
    }


def flush_mesh_cache(cache: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(list(cache.values()), columns=MESH_CACHE_COLUMNS)
    if len(frame):
        frame = frame.sort_values("mesh_path").reset_index(drop=True)
    write_csv_atomic(frame, path)


def cached_mesh_stats(
    path: str | Path,
    cache: dict[str, dict[str, Any]],
    *,
    refresh: bool,
) -> dict[str, Any]:
    mesh_path = Path(path).expanduser().resolve()
    key = str(mesh_path)
    if not mesh_path.is_file():
        return {
            "mesh_path": key,
            "size_bytes": np.nan,
            "mtime_ns": np.nan,
            "status": "missing",
            "error": "missing mesh",
            "volume": np.nan,
            "watertight": False,
            "vertex_count": np.nan,
            "face_count": np.nan,
        }
    stat = mesh_path.stat()
    existing = cache.get(key)
    if (
        not refresh
        and existing is not None
        and int(existing.get("size_bytes", -1)) == int(stat.st_size)
        and int(existing.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
    ):
        return existing
    record: dict[str, Any] = {
        "mesh_path": key,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "status": "failed",
        "error": "",
        "volume": np.nan,
        "watertight": False,
        "vertex_count": np.nan,
        "face_count": np.nan,
    }
    try:
        record.update(inspect_mesh(mesh_path))
        record["status"] = "ok"
    except Exception as exc:
        record["error"] = str(exc)
    cache[key] = record
    return record


def compute_ground_truth_volumes(
    large_all: pd.DataFrame,
    old_small: pd.DataFrame,
    *,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    refresh: bool,
) -> pd.DataFrame:
    unique_rows = pd.concat(
        [large_all.assign(source_dataset="large"), old_small.assign(source_dataset="old")],
        ignore_index=True,
    ).drop_duplicates(["source_dataset", "scan_id"])
    result: list[dict[str, Any]] = []
    total = len(unique_rows)
    for index, row in enumerate(unique_rows.itertuples(index=False), start=1):
        mesh_path = Path(row.mesh_path)
        stats = cached_mesh_stats(mesh_path, cache, refresh=refresh)
        result.append(
            {
                "source_dataset": row.source_dataset,
                "scan_id": row.scan_id,
                "ground_truth_volume": stats["volume"],
                "ground_truth_status": stats["status"],
                "ground_truth_error": stats["error"],
                "ground_truth_watertight": stats["watertight"],
                "ground_truth_vertex_count": stats["vertex_count"],
                "ground_truth_face_count": stats["face_count"],
                "mesh_path": str(mesh_path),
            }
        )
        if index % 250 == 0 or index == total:
            flush_mesh_cache(cache, cache_path)
            print(f"Ground-truth mesh volumes: {index}/{total}")
    return pd.DataFrame(result)


def old_reconstruction_volumes(
    metadata: pd.DataFrame,
    method: str,
    directory: Path,
    *,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    refresh: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(metadata)
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        path = directory / "reconstructed_meshes" / f"{row.scan_id}.ply"
        stats = cached_mesh_stats(path, cache, refresh=refresh)
        rows.append(
            {
                "scan_id": row.scan_id,
                "method": method,
                "reconstructed_volume": stats["volume"],
                "reconstruction_watertight": stats["watertight"],
                "reconstruction_status": stats["status"],
                "reconstruction_error": stats["error"],
                "reconstruction_mesh_path": str(path),
                "artifact_source": "complete_old_reconstruction",
            }
        )
        if index % 250 == 0 or index == total:
            flush_mesh_cache(cache, cache_path)
            print(f"Old {method} reconstruction volumes: {index}/{total}")
    return pd.DataFrame(rows)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def large_siren_reconstruction_volumes(
    metadata: pd.DataFrame,
    *,
    full_mesh_dir: Path,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    refresh: bool,
) -> pd.DataFrame:
    subset_roots = [
        LARGE_SIREN_DIR / "chamfer_20_per_split_best/meshes",
        LARGE_SIREN_DIR / "quick_chamfer_best_subset/meshes",
    ]
    records: list[dict[str, Any]] = []
    total = len(metadata)
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        candidates = [
            (full_mesh_dir / row.split / f"{row.scan_id}.ply", "full_multi_gpu"),
            (full_mesh_dir / f"{row.scan_id}.ply", "full_multi_gpu"),
        ]
        for root in subset_roots:
            candidates.extend(
                [
                    (root / row.split / f"{row.scan_id}.ply", "existing_60_subset"),
                    (root / f"{row.scan_id}.ply", "existing_60_subset"),
                ]
            )
        selected_path, artifact_source = next(
            ((path, source) for path, source in candidates if path.is_file()),
            (candidates[0][0], "unavailable"),
        )
        stats = cached_mesh_stats(selected_path, cache, refresh=refresh)
        records.append(
            {
                "scan_id": row.scan_id,
                "method": "siren",
                "reconstructed_volume": stats["volume"],
                "reconstruction_watertight": stats["watertight"],
                "reconstruction_status": stats["status"],
                "reconstruction_error": stats["error"],
                "reconstruction_mesh_path": str(selected_path),
                "artifact_source": artifact_source,
            }
        )
        if index % 250 == 0 or index == total:
            flush_mesh_cache(cache, cache_path)
            available = sum(item["reconstruction_status"] == "ok" for item in records)
            print(f"Large SIREN existing meshes: {index}/{total}, available={available}")
    return pd.DataFrame(records)


def load_latent_archives(directory: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        path = directory / "latents" / f"{split}_latents.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            for scan_id, latent in zip(payload["scan_ids"], payload["latents"]):
                key = Path(str(scan_id)).stem
                result[key] = np.asarray(latent, dtype=np.float64)
    return result


def subject_diagnosis(values: Iterable[str]) -> str:
    unique = sorted(set(str(value) for value in values))
    return unique[0] if len(unique) == 1 else "MIXED"


def build_scan_volume_table(
    cohorts: Mapping[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    large_siren: pd.DataFrame,
    old_siren: pd.DataFrame,
    old_deepsdf: pd.DataFrame,
) -> pd.DataFrame:
    gt_lookup = ground_truth.set_index(["source_dataset", "scan_id"])
    large_siren_lookup = large_siren.set_index("scan_id") if len(large_siren) else None
    old_siren_lookup = old_siren.set_index("scan_id")
    old_deepsdf_lookup = old_deepsdf.set_index("scan_id")
    rows: list[dict[str, Any]] = []
    for cohort, frame in cohorts.items():
        source_dataset = "old" if cohort == "old_small" else "large"
        diagnosis_map = {
            subject: subject_diagnosis(group["diagnosis"])
            for subject, group in frame.groupby("subject_id", sort=False)
        }
        for item in frame.itertuples(index=False):
            common = {
                "cohort": cohort,
                "source_dataset": source_dataset,
                "scan_id": item.scan_id,
                "subject_id": item.subject_id,
                "split": item.split,
                "diagnosis": item.diagnosis,
                "subject_diagnosis": diagnosis_map[item.subject_id],
                "age_years": float(item.age_years),
                "visit_order": int(item.visit_order),
                "months_from_baseline": item.months_from_baseline,
                "metadata_mesh_volume_mm3": item.metadata_mesh_volume_mm3,
                "mask_volume_mm3": item.mask_volume_mm3,
                "ground_truth_mesh_path": item.mesh_path,
            }
            gt = gt_lookup.loc[(source_dataset, item.scan_id)]
            rows.append(
                {
                    **common,
                    "method": "ground_truth",
                    "volume": gt["ground_truth_volume"],
                    "status": gt["ground_truth_status"],
                    "error": gt["ground_truth_error"],
                    "reconstruction_mesh_path": gt["mesh_path"],
                    "artifact_source": "ground_truth_mesh",
                    "watertight": gt["ground_truth_watertight"],
                }
            )
            representation_sources: list[tuple[str, Any]] = []
            if source_dataset == "large" and large_siren_lookup is not None:
                if item.scan_id in large_siren_lookup.index:
                    representation_sources.append(
                        ("siren", large_siren_lookup.loc[item.scan_id])
                    )
            elif source_dataset == "old":
                if item.scan_id in old_siren_lookup.index:
                    representation_sources.append(
                        ("siren", old_siren_lookup.loc[item.scan_id])
                    )
                if item.scan_id in old_deepsdf_lookup.index:
                    representation_sources.append(
                        ("deepsdf", old_deepsdf_lookup.loc[item.scan_id])
                    )
            for method, reconstructed in representation_sources:
                rows.append(
                    {
                        **common,
                        "method": method,
                        "volume": reconstructed["reconstructed_volume"],
                        "status": reconstructed["reconstruction_status"],
                        "error": reconstructed["reconstruction_error"],
                        "reconstruction_mesh_path": reconstructed[
                            "reconstruction_mesh_path"
                        ],
                        "artifact_source": reconstructed["artifact_source"],
                        "watertight": reconstructed["reconstruction_watertight"],
                    }
                )
    return pd.DataFrame(rows)


def build_pair_rates(scan_volumes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = scan_volumes.loc[
        (scan_volumes["status"] == "ok") & np.isfinite(scan_volumes["volume"])
    ]
    group_columns = ["cohort", "method", "subject_id"]
    for (cohort, method, subject), group in valid.groupby(group_columns, sort=False):
        group = group.sort_values(["age_years", "visit_order", "scan_id"])
        records = list(group.itertuples(index=False))
        for source_index, target_index in itertools.combinations(range(len(records)), 2):
            source, target = records[source_index], records[target_index]
            gap = float(target.age_years - source.age_years)
            if not np.isfinite(gap) or gap <= 0.0:
                continue
            delta = float(target.volume - source.volume)
            rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "subject_id": subject,
                    "split": source.split,
                    "subject_diagnosis": source.subject_diagnosis,
                    "source_scan_id": source.scan_id,
                    "target_scan_id": target.scan_id,
                    "source_age": source.age_years,
                    "target_age": target.age_years,
                    "midpoint_age": 0.5 * (source.age_years + target.age_years),
                    "gap_years": gap,
                    "adjacent": target_index == source_index + 1,
                    "source_volume": source.volume,
                    "target_volume": target.volume,
                    "volume_delta": delta,
                    "annual_volume_rate": delta / gap,
                    "annual_percent_rate": 100.0 * delta / max(source.volume, 1.0e-12) / gap,
                    "volume_decreased": delta < 0.0,
                }
            )
    return pd.DataFrame(rows)


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = pd.Series(left, dtype=float)
    y = pd.Series(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    return float(x[mask].rank().corr(y[mask].rank(), method="pearson"))


def build_subject_summaries(
    scan_volumes: pd.DataFrame,
    pair_rates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = scan_volumes.loc[
        (scan_volumes["status"] == "ok") & np.isfinite(scan_volumes["volume"])
    ]
    for keys, scans in valid.groupby(["cohort", "method", "subject_id"], sort=False):
        cohort, method, subject = keys
        scans = scans.sort_values(["age_years", "visit_order", "scan_id"])
        subject_pairs = pair_rates.loc[
            (pair_rates["cohort"] == cohort)
            & (pair_rates["method"] == method)
            & (pair_rates["subject_id"] == subject)
        ]
        adjacent = subject_pairs.loc[subject_pairs["adjacent"]]
        ages = scans["age_years"].to_numpy(dtype=float)
        volumes = scans["volume"].to_numpy(dtype=float)
        slope = np.nan
        if len(scans) >= 2 and np.ptp(ages) > 0.0:
            slope = float(np.polyfit(ages, volumes, 1)[0])
        percent_slope = 100.0 * slope / max(float(volumes[0]), 1.0e-12)
        rows.append(
            {
                "cohort": cohort,
                "method": method,
                "subject_id": subject,
                "split": scans["split"].iloc[0],
                "subject_diagnosis": scans["subject_diagnosis"].iloc[0],
                "scan_count": len(scans),
                "pair_count": len(subject_pairs),
                "age_min": float(ages.min()),
                "age_max": float(ages.max()),
                "followup_years": float(ages.max() - ages.min()),
                "ols_volume_slope": slope,
                "ols_percent_slope": percent_slope,
                "mean_pair_rate": subject_pairs["annual_volume_rate"].mean(),
                "median_pair_rate": subject_pairs["annual_volume_rate"].median(),
                "mean_pair_percent_rate": subject_pairs["annual_percent_rate"].mean(),
                "mean_adjacent_rate": adjacent["annual_volume_rate"].mean(),
                "decreasing_pair_fraction": subject_pairs["volume_decreased"].mean(),
                "overall_decrease": bool(volumes[-1] < volumes[0]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: Sequence[float], samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0 or samples <= 0:
        return np.nan, np.nan
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_group_summary(
    subjects: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    keys = ["cohort", "method", "split", "subject_diagnosis"]
    for values, group in subjects.groupby(keys, dropna=False, sort=True):
        slopes = group["ols_percent_slope"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(slopes, bootstrap_samples, rng)
        rows.append(
            {
                **dict(zip(keys, values)),
                "subjects": len(group),
                "scans": int(group["scan_count"].sum()),
                "mean_annual_percent_slope": float(np.nanmean(slopes)),
                "median_annual_percent_slope": float(np.nanmedian(slopes)),
                "mean_slope_ci95_low": ci_low,
                "mean_slope_ci95_high": ci_high,
                "subjects_decreasing_fraction": float(
                    np.mean(group["ols_volume_slope"] < 0.0)
                ),
                "mean_pair_percent_rate": float(
                    np.nanmean(group["mean_pair_percent_rate"])
                ),
                "mean_decreasing_pair_fraction": float(
                    np.nanmean(group["decreasing_pair_fraction"])
                ),
            }
        )
    return pd.DataFrame(rows)


def build_representation_comparisons(
    pair_rates: pd.DataFrame,
    subjects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_pairs = pair_rates.loc[pair_rates["method"] == "ground_truth"].copy()
    representation_pairs = pair_rates.loc[pair_rates["method"] != "ground_truth"].copy()
    merge_keys = ["cohort", "subject_id", "source_scan_id", "target_scan_id"]
    paired = representation_pairs.merge(
        gt_pairs[
            merge_keys
            + ["annual_volume_rate", "annual_percent_rate", "volume_decreased"]
        ],
        on=merge_keys,
        how="inner",
        suffixes=("_rec", "_gt"),
    )
    subject_rows: list[dict[str, Any]] = []
    for values, group in paired.groupby(
        ["cohort", "method", "subject_id"], sort=False
    ):
        cohort, method, subject = values
        rate_error = group["annual_volume_rate_rec"] - group["annual_volume_rate_gt"]
        percent_error = (
            group["annual_percent_rate_rec"] - group["annual_percent_rate_gt"]
        )
        subject_rows.append(
            {
                "cohort": cohort,
                "method": method,
                "subject_id": subject,
                "split": group["split"].iloc[0],
                "subject_diagnosis": group["subject_diagnosis"].iloc[0],
                "matched_pairs": len(group),
                "annual_rate_mae": float(np.mean(np.abs(rate_error))),
                "annual_rate_bias": float(np.mean(rate_error)),
                "annual_percent_rate_mae": float(np.mean(np.abs(percent_error))),
                "annual_percent_rate_bias": float(np.mean(percent_error)),
                "pair_sign_agreement": float(
                    np.mean(
                        group["volume_decreased_rec"].to_numpy()
                        == group["volume_decreased_gt"].to_numpy()
                    )
                ),
                "pair_rate_spearman": safe_spearman(
                    group["annual_volume_rate_rec"],
                    group["annual_volume_rate_gt"],
                ),
            }
        )
    subject_comparison = pd.DataFrame(subject_rows)

    gt_subjects = subjects.loc[subjects["method"] == "ground_truth"].copy()
    rec_subjects = subjects.loc[subjects["method"] != "ground_truth"].copy()
    slope_paired = rec_subjects.merge(
        gt_subjects[
            ["cohort", "subject_id", "ols_volume_slope", "ols_percent_slope"]
        ],
        on=["cohort", "subject_id"],
        how="inner",
        suffixes=("_rec", "_gt"),
    )
    group_rows: list[dict[str, Any]] = []
    keys = ["cohort", "method", "split", "subject_diagnosis"]
    for values, group in subject_comparison.groupby(keys, sort=True):
        cohort, method, split, diagnosis = values
        slopes = slope_paired.loc[
            (slope_paired["cohort"] == cohort)
            & (slope_paired["method"] == method)
            & (slope_paired["split"] == split)
            & (slope_paired["subject_diagnosis"] == diagnosis)
        ]
        group_rows.append(
            {
                **dict(zip(keys, values)),
                "subjects": len(group),
                "mean_annual_rate_mae": group["annual_rate_mae"].mean(),
                "mean_annual_rate_bias": group["annual_rate_bias"].mean(),
                "mean_annual_percent_rate_mae": group[
                    "annual_percent_rate_mae"
                ].mean(),
                "mean_pair_sign_agreement": group["pair_sign_agreement"].mean(),
                "mean_within_subject_rate_spearman": group[
                    "pair_rate_spearman"
                ].mean(),
                "subject_slope_spearman": safe_spearman(
                    slopes["ols_percent_slope_rec"],
                    slopes["ols_percent_slope_gt"],
                ),
                "subject_slope_sign_agreement": float(
                    np.mean(
                        (slopes["ols_volume_slope_rec"] < 0.0).to_numpy()
                        == (slopes["ols_volume_slope_gt"] < 0.0).to_numpy()
                    )
                )
                if len(slopes)
                else np.nan,
            }
        )
    return subject_comparison, pd.DataFrame(group_rows)


def latent_diagnostics_for_cohort(
    metadata: pd.DataFrame,
    latents: Mapping[str, np.ndarray],
    ground_truth_volumes: Mapping[str, float],
    method: str,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for subject, group in metadata.groupby("subject_id", sort=False):
        group = group.loc[group["scan_id"].isin(latents)].sort_values(
            ["age_years", "visit_order", "scan_id"]
        )
        if len(group) < 2:
            continue
        records = list(group.itertuples(index=False))
        latent_steps: list[np.ndarray] = []
        mesh_speeds: list[float] = []
        latent_speeds: list[float] = []
        volume_rates: list[float] = []
        for source, target in zip(records[:-1], records[1:]):
            gap = float(target.age_years - source.age_years)
            if gap <= 0.0:
                continue
            delta = latents[target.scan_id] - latents[source.scan_id]
            source_vertices, _ = load_obj_geometry(str(Path(source.mesh_path).resolve()))
            target_vertices, _ = load_obj_geometry(str(Path(target.mesh_path).resolve()))
            if source_vertices.shape != target_vertices.shape:
                mesh_speed = np.nan
            else:
                mesh_speed = float(
                    np.linalg.norm(target_vertices - source_vertices)
                    / math.sqrt(len(source_vertices))
                    / gap
                )
            source_volume = float(ground_truth_volumes[source.scan_id])
            target_volume = float(ground_truth_volumes[target.scan_id])
            volume_rate = (target_volume - source_volume) / gap
            latent_speed = float(np.linalg.norm(delta) / gap)
            latent_steps.append(delta)
            mesh_speeds.append(mesh_speed)
            latent_speeds.append(latent_speed)
            volume_rates.append(volume_rate)
            pair_rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "subject_id": subject,
                    "split": source.split,
                    "subject_diagnosis": subject_diagnosis(group["diagnosis"]),
                    "source_scan_id": source.scan_id,
                    "target_scan_id": target.scan_id,
                    "gap_years": gap,
                    "latent_speed": latent_speed,
                    "ground_truth_mesh_speed": mesh_speed,
                    "ground_truth_volume_rate": volume_rate,
                }
            )
        turn_cosines = []
        for left, right in zip(latent_steps[:-1], latent_steps[1:]):
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            if denominator > 0.0:
                turn_cosines.append(float(np.dot(left, right) / denominator))
        subject_rows.append(
            {
                "cohort": cohort,
                "method": method,
                "subject_id": subject,
                "split": group["split"].iloc[0],
                "subject_diagnosis": subject_diagnosis(group["diagnosis"]),
                "scan_count": len(group),
                "adjacent_steps": len(latent_steps),
                "successive_delta_cosine": np.mean(turn_cosines)
                if turn_cosines
                else np.nan,
                "latent_speed_vs_mesh_speed_spearman": safe_spearman(
                    latent_speeds, mesh_speeds
                ),
                "latent_speed_vs_abs_volume_rate_spearman": safe_spearman(
                    latent_speeds, np.abs(volume_rates)
                ),
                "mean_latent_speed": np.mean(latent_speeds)
                if latent_speeds
                else np.nan,
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(subject_rows)


def build_availability(
    cohorts: Mapping[str, pd.DataFrame],
    scan_volumes: pd.DataFrame,
    latent_maps: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, metadata in cohorts.items():
        methods = ["ground_truth", "siren", "deepsdf"]
        for method in methods:
            volumes = scan_volumes.loc[
                (scan_volumes["cohort"] == cohort)
                & (scan_volumes["method"] == method)
                & (scan_volumes["status"] == "ok")
            ]
            latent_map = latent_maps.get((cohort, method), {})
            rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "metadata_scans": len(metadata),
                    "subjects": metadata["subject_id"].nunique(),
                    "available_volumes": volumes["scan_id"].nunique(),
                    "available_latents": len(set(metadata["scan_id"]).intersection(latent_map)),
                    "coverage_fraction": volumes["scan_id"].nunique()
                    / max(len(metadata), 1),
                }
            )
    return pd.DataFrame(rows)


def dataframe_html(frame: pd.DataFrame, max_rows: int = 100) -> str:
    shown = frame.head(max_rows).copy()
    numeric = shown.select_dtypes(include=[np.number]).columns
    shown[numeric] = shown[numeric].round(5)
    return shown.to_html(index=False, escape=True, border=0, classes="dataframe")


def write_html_report(
    output_path: Path,
    availability: pd.DataFrame,
    group_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    latent_subjects: pd.DataFrame,
) -> None:
    plot_blocks: list[str] = []
    try:
        import plotly.express as px
        import plotly.io as pio

        slope_frame = group_summary.loc[
            group_summary["subject_diagnosis"].isin(["CN", "AD"])
        ]
        if len(slope_frame):
            figure = px.bar(
                slope_frame,
                x="cohort",
                y="mean_annual_percent_slope",
                color="method",
                facet_row="subject_diagnosis",
                barmode="group",
                title="Subject-weighted annual volume slope",
                labels={"mean_annual_percent_slope": "Mean annual slope (%/year)"},
            )
            plot_blocks.append(
                pio.to_html(figure, include_plotlyjs=True, full_html=False)
            )
        if len(comparison_summary):
            figure = px.bar(
                comparison_summary,
                x="cohort",
                y="mean_pair_sign_agreement",
                color="method",
                facet_row="subject_diagnosis",
                barmode="group",
                title="Reconstruction versus ground-truth volume-change sign",
                range_y=[0, 1],
            )
            plot_blocks.append(
                pio.to_html(figure, include_plotlyjs=False, full_html=False)
            )
    except ImportError:
        plot_blocks.append("<p>Plotly is unavailable; CSV tables were still generated.</p>")

    latent_group = pd.DataFrame()
    if len(latent_subjects):
        latent_group = (
            latent_subjects.groupby(
                ["cohort", "method", "subject_diagnosis"], dropna=False
            )
            .agg(
                subjects=("subject_id", "nunique"),
                subjects_ge3=("successive_delta_cosine", "count"),
                mean_successive_delta_cosine=("successive_delta_cosine", "mean"),
                mean_latent_speed=("mean_latent_speed", "mean"),
            )
            .reset_index()
        )
    style = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #202124; }
    h1, h2 { font-weight: 600; }
    .note { max-width: 1000px; line-height: 1.5; }
    table.dataframe { border-collapse: collapse; font-size: 12px; }
    table.dataframe th, table.dataframe td { border: 1px solid #ddd; padding: 5px 7px; }
    table.dataframe th { background: #f3f4f6; position: sticky; top: 0; }
    """
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ADNI Representation Volume Audit</title>
<style>{style}</style></head><body>
<h1>ADNI Representation Volume Audit</h1>
<p class="note">Test A measures ground-truth longitudinal volume trends. Test B
compares independently reconstructed SIREN and DeepSDF scan volumes with the
same ground-truth trends. Every group statistic is formed from subject-level
summaries so subjects with many visits do not dominate.</p>
<h2>Availability</h2>{dataframe_html(availability)}
<h2>Subject-weighted group trends</h2>{dataframe_html(group_summary)}
<h2>Reconstruction versus ground truth</h2>{dataframe_html(comparison_summary)}
<h2>Latent longitudinal structure</h2>{dataframe_html(latent_group)}
{''.join(plot_blocks)}
</body></html>"""
    output_path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mesh_cache_path = Path(args.mesh_cache).expanduser().resolve()
    large_siren_mesh_dir = Path(args.large_siren_mesh_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_cache = load_mesh_cache(mesh_cache_path)
    require_files(
        [LARGE_MANIFEST, LARGE_STABLE_MANIFEST, LARGE_QC_MANIFEST, OLD_MANIFEST]
    )

    large_all = read_manifest(LARGE_MANIFEST, "large_all")
    large_stable = read_manifest(LARGE_STABLE_MANIFEST, "large_stable")
    large_qc = read_manifest(LARGE_QC_MANIFEST, "large_qc")
    old_small = read_manifest(OLD_MANIFEST, "old_small")
    cohorts = {
        "large_all": large_all,
        "large_stable": large_stable,
        "large_qc": large_qc,
        "old_small": old_small,
    }

    ground_truth = compute_ground_truth_volumes(
        large_all,
        old_small,
        cache=mesh_cache,
        cache_path=mesh_cache_path,
        refresh=args.refresh_mesh_cache,
    )
    write_csv_atomic(ground_truth, output_dir / "source_mesh_volumes.csv")
    old_siren = old_reconstruction_volumes(
        old_small,
        "siren",
        OLD_SIREN_DIR,
        cache=mesh_cache,
        cache_path=mesh_cache_path,
        refresh=args.refresh_mesh_cache,
    )
    old_deepsdf = old_reconstruction_volumes(
        old_small,
        "deepsdf",
        OLD_DEEPSDF_DIR,
        cache=mesh_cache,
        cache_path=mesh_cache_path,
        refresh=args.refresh_mesh_cache,
    )
    large_siren = large_siren_reconstruction_volumes(
        large_all,
        full_mesh_dir=large_siren_mesh_dir,
        cache=mesh_cache,
        cache_path=mesh_cache_path,
        refresh=args.refresh_mesh_cache,
    )
    flush_mesh_cache(mesh_cache, mesh_cache_path)

    scan_volumes = build_scan_volume_table(
        cohorts,
        ground_truth,
        large_siren,
        old_siren,
        old_deepsdf,
    )
    pair_rates = build_pair_rates(scan_volumes)
    subject_summary = build_subject_summaries(scan_volumes, pair_rates)
    group_summary = build_group_summary(
        subject_summary,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
    )
    representation_subject, representation_group = build_representation_comparisons(
        pair_rates, subject_summary
    )

    large_siren_latents = load_latent_archives(LARGE_SIREN_DIR)
    old_siren_latents = load_latent_archives(OLD_SIREN_DIR)
    old_deepsdf_latents = load_latent_archives(OLD_DEEPSDF_DIR)
    latent_maps: dict[tuple[str, str], Mapping[str, np.ndarray]] = {
        ("large_all", "siren"): large_siren_latents,
        ("large_stable", "siren"): large_siren_latents,
        ("large_qc", "siren"): large_siren_latents,
        ("old_small", "siren"): old_siren_latents,
        ("old_small", "deepsdf"): old_deepsdf_latents,
    }
    ground_truth_volume_maps = {
        source: frame.set_index("scan_id")["ground_truth_volume"].to_dict()
        for source, frame in ground_truth.groupby("source_dataset", sort=False)
    }
    latent_pair_frames: list[pd.DataFrame] = []
    latent_subject_frames: list[pd.DataFrame] = []
    for (cohort, method), latent_map in latent_maps.items():
        pair_frame, subject_frame = latent_diagnostics_for_cohort(
            cohorts[cohort],
            latent_map,
            ground_truth_volume_maps[
                "old" if cohort == "old_small" else "large"
            ],
            method,
            cohort,
        )
        latent_pair_frames.append(pair_frame)
        latent_subject_frames.append(subject_frame)
    latent_pairs = pd.concat(latent_pair_frames, ignore_index=True)
    latent_subjects = pd.concat(latent_subject_frames, ignore_index=True)
    availability = build_availability(cohorts, scan_volumes, latent_maps)

    outputs = {
        "scan_volumes.csv": scan_volumes,
        "pair_volume_rates.csv": pair_rates,
        "subject_volume_summary.csv": subject_summary,
        "group_volume_summary.csv": group_summary,
        "representation_subject_comparison.csv": representation_subject,
        "representation_group_comparison.csv": representation_group,
        "latent_pair_diagnostics.csv": latent_pairs,
        "latent_subject_diagnostics.csv": latent_subjects,
        "availability.csv": availability,
    }
    for name, frame in outputs.items():
        write_csv_atomic(frame, output_dir / name)

    write_html_report(
        output_dir / "report.html",
        availability,
        group_summary,
        representation_group,
        latent_subjects,
    )
    run_summary = {
        "output_dir": str(output_dir),
        "analysis_stage": "existing_meshes_cpu",
        "mesh_cache": str(mesh_cache_path),
        "large_siren_mesh_dir": str(large_siren_mesh_dir),
        "decoder_loaded": False,
        "large_deepsdf_available": False,
        "new_latents_created": False,
        "large_siren_artifact_counts": {
            str(source): int(count)
            for source, count in large_siren.loc[
                large_siren["reconstruction_status"] == "ok", "artifact_source"
            ].value_counts().items()
        },
        "cohorts": {
            name: {
                "scans": int(len(frame)),
                "subjects": int(frame["subject_id"].nunique()),
            }
            for name, frame in cohorts.items()
        },
        "output_files": [str(output_dir / name) for name in outputs]
        + [str(output_dir / "report.html")],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
