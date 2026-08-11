"""Shared utilities for QC and longitudinal analysis of SynthSeg meshes.

The functions in this module are deliberately read-only with respect to the
SynthSeg source tree.  They assemble its manifest and mesh-QC provenance into
analysis records, calculate longitudinal volume changes, and use the final
physical-unit correspondence meshes for vertex-wise change measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import json
import re

import numpy as np
import pandas as pd


DEFAULT_SYNTHSEG_ROOT = Path(
    "/home/jakaria/ADNI/ADNI_1_GO_Large/synthseg_minimal_correspondence/full"
)
STRUCTURES = ("left_hippocampus", "left_lateral_ventricle")
STRUCTURE_LABELS = {
    "left_hippocampus": "Left hippocampus",
    "left_lateral_ventricle": "Left lateral ventricle",
}
STRUCTURE_SHORT = {"left_hippocampus": "hippocampus", "left_lateral_ventricle": "ventricle"}
DIAGNOSES = ("CN", "MCI", "AD")
COHORT_FILTERS = ("strict_no_mci", "all")
DIAGNOSIS_COLORS = {"CN": "#2E86DE", "MCI": "#F39C12", "AD": "#C0392B"}


@dataclass(frozen=True)
class LoadedMesh:
    """A mesh loaded without any repair or processing."""

    vertices: np.ndarray
    faces: np.ndarray
    volume: float
    area: float
    watertight: bool
    winding_consistent: bool
    component_count: int
    finite_vertices: bool


def terminal_progress(stage: str, index: int, total: int, *, every: int = 100) -> None:
    """Emit simple progress that remains visible in redirected terminal logs."""

    if total <= 0 or index == 1 or index == total or index % max(1, every) == 0:
        percent = 100.0 * index / max(1, total)
        print(f"[{stage}] {index:,}/{total:,} ({percent:5.1f}%)", flush=True)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(to_builtin(dict(payload)), indent=2), encoding="utf-8")


def parse_structures(value: str | Sequence[str]) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else value
    selected = tuple(item.strip() for item in items if item.strip())
    unknown = sorted(set(selected).difference(STRUCTURES))
    if unknown:
        raise ValueError(f"Unsupported structures: {unknown}. Choices: {list(STRUCTURES)}")
    if not selected:
        raise ValueError("At least one structure is required.")
    return selected


def normalize_diagnosis(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    if normalized in DIAGNOSES:
        return normalized
    return None


def truthy(value: object) -> bool:
    """Interpret CSV boolean values without treating missing data as true."""

    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def visit_month(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"bl", "sc", "screening", "baseline"}:
        return 0.0
    match = re.fullmatch(r"m(\d+)", text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        return np.nan


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def safe_percent_error(observed: float, expected: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(expected) or expected == 0.0:
        return np.nan
    return 100.0 * (observed - expected) / expected


def structure_paths(input_root: Path, structure: str, scan_id: str) -> dict[str, Path]:
    base = input_root / structure
    return {
        "raw_mesh_path": base / "raw_ply" / f"{scan_id}.ply",
        "smooth_mesh_path": base / "minimal_smooth_ply" / f"{scan_id}.ply",
        "correspondence_mesh_path": base
        / "minimal_smooth_correspondence"
        / "final_ply_mm"
        / f"{scan_id}.ply",
        "correspondence_normalized_mesh_path": base
        / "minimal_smooth_correspondence"
        / "final_ply"
        / f"{scan_id}.ply",
    }


def load_mesh(path: str | Path) -> LoadedMesh:
    """Load a mesh for inspection without changing it on disk or in memory."""

    import trimesh

    mesh_path = Path(path)
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("empty mesh scene")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    components = len(loaded.split(only_watertight=False))
    return LoadedMesh(
        vertices=vertices,
        faces=faces,
        volume=abs(float(loaded.volume)),
        area=float(loaded.area),
        watertight=bool(loaded.is_watertight),
        winding_consistent=bool(loaded.is_winding_consistent),
        component_count=int(components),
        finite_vertices=bool(np.isfinite(vertices).all()),
    )


def mesh_topology_hash(faces: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    return sha256(canonical.tobytes()).hexdigest()


def _manifest_frame(input_root: Path) -> pd.DataFrame:
    path = input_root / "manifests" / "selected_scans.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Selected SynthSeg manifest is missing: {path}")
    frame = pd.read_csv(path, dtype={"RID": str, "VISCODE": str, "scan_id": str})
    required = {"scan_id", "RID", "VISCODE"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Manifest is missing required columns: {missing}")
    frame["subject_id"] = frame["RID"].astype(str)
    frame["visit_month"] = (
        pd.to_numeric(frame.get("month_from_viscode"), errors="coerce")
        if "month_from_viscode" in frame
        else frame["VISCODE"].map(visit_month)
    )
    frame["visit_month"] = frame["visit_month"].fillna(frame["VISCODE"].map(visit_month))
    frame["age_years"] = pd.to_numeric(
        frame.get("age_numeric", frame.get("AGE", np.nan)), errors="coerce"
    )
    for source, destination in (
        ("baseline_dx_3class", "baseline_diagnosis"),
        ("visit_dx_3class", "visit_diagnosis"),
    ):
        frame[destination] = frame.get(source, pd.Series(index=frame.index, dtype=object)).map(
            normalize_diagnosis
        )
    frame["sex"] = frame.get("PTGENDER", pd.Series(index=frame.index, dtype=object)).astype(str)
    return frame


def load_analysis_records(
    input_root: str | Path,
    structures: Sequence[str] = STRUCTURES,
    *,
    cohort_label: str = "baseline",
    cohort_filter: str = "strict_no_mci",
) -> pd.DataFrame:
    """Return one row per scan and structure with volume/provenance metadata.

    The default ``cohort_filter='strict_no_mci'`` selects only baseline-CN/AD
    participants with no MCI-labelled meshed visit. It excludes baseline-MCI
    converters/reverters as well as unknown baseline groups.
    """

    root = Path(input_root).expanduser().resolve()
    if cohort_label not in {"baseline", "visit"}:
        raise ValueError("cohort_label must be 'baseline' or 'visit'")
    if cohort_filter not in COHORT_FILTERS:
        raise ValueError(f"cohort_filter must be one of {COHORT_FILTERS}")
    manifest = _manifest_frame(root)
    diagnosis_column = f"{cohort_label}_diagnosis"
    manifest["cohort_diagnosis"] = manifest[diagnosis_column]
    manifest["strict_subject_no_mci"] = manifest.get(
        "strict_subject_no_mci", pd.Series(False, index=manifest.index)
    ).map(truthy)
    if cohort_filter == "strict_no_mci":
        manifest = manifest.loc[
            manifest["baseline_diagnosis"].isin(("CN", "AD"))
            & manifest["strict_subject_no_mci"]
        ].copy()
    if manifest.empty:
        raise RuntimeError(f"No scans remain after applying cohort filter {cohort_filter!r}.")
    manifest["cohort_filter"] = cohort_filter
    all_frames: list[pd.DataFrame] = []
    for structure in structures:
        qc_path = root / structure / "mesh_qc.csv"
        if not qc_path.is_file():
            raise FileNotFoundError(f"Mesh QC provenance is missing: {qc_path}")
        qc = pd.read_csv(qc_path, dtype={"scan_id": str, "RID": str, "VISCODE": str})
        if "scan_id" not in qc:
            raise KeyError(f"mesh_qc.csv has no scan_id column: {qc_path}")
        keep = [
            column
            for column in (
                "scan_id",
                "mask_volume_mm3",
                "raw_mesh_volume_mm3",
                "raw_surface_area_mm2",
                "raw_mesh_components",
                "raw_watertight",
                "smooth_mesh_volume_mm3",
                "smooth_surface_area_mm2",
                "smooth_mesh_components",
                "smooth_watertight",
                "smooth_vs_mask_volume_pct",
                "status",
            )
            if column in qc
        ]
        merged = manifest.merge(qc.loc[:, keep], on="scan_id", how="left", validate="one_to_one")
        merged["structure"] = structure
        merged["structure_display"] = STRUCTURE_LABELS[structure]
        paths = merged["scan_id"].astype(str).map(lambda scan_id: structure_paths(root, structure, scan_id))
        for key in next(iter(paths), {}).keys() if len(paths) else []:
            merged[key] = paths.map(lambda value: str(value[key]))
        merged["mesh_pipeline_status"] = merged.get("status", "missing").fillna("missing").astype(str)
        all_frames.append(merged)
    records = pd.concat(all_frames, ignore_index=True, sort=False)
    numeric_columns = [
        "visit_month",
        "age_years",
        "mask_volume_mm3",
        "raw_mesh_volume_mm3",
        "raw_surface_area_mm2",
        "smooth_mesh_volume_mm3",
        "smooth_surface_area_mm2",
        "smooth_vs_mask_volume_pct",
    ]
    for column in numeric_columns:
        if column in records:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    records = records.sort_values(
        ["structure", "subject_id", "visit_month", "VISCODE", "scan_id"], kind="stable"
    ).reset_index(drop=True)
    return records


def build_adjacent_pairs(records: pd.DataFrame) -> pd.DataFrame:
    """Calculate adjacent-visit physical-volume rates for each structure."""

    rows: list[dict[str, Any]] = []
    order_columns = ["visit_month", "age_years", "VISCODE", "scan_id"]
    for (structure, subject_id), group in records.groupby(["structure", "subject_id"], sort=True):
        ordered = group.sort_values(order_columns, kind="stable")
        for index in range(1, len(ordered)):
            source = ordered.iloc[index - 1]
            target = ordered.iloc[index]
            source_month = safe_float(source["visit_month"])
            target_month = safe_float(target["visit_month"])
            delta_years = (target_month - source_month) / 12.0
            if not np.isfinite(delta_years) or delta_years <= 0.0:
                continue
            source_volume = safe_float(source.get("smooth_mesh_volume_mm3"))
            target_volume = safe_float(target.get("smooth_mesh_volume_mm3"))
            log_rate = (
                100.0 * np.log(target_volume / source_volume) / delta_years
                if source_volume > 0.0 and target_volume > 0.0
                else np.nan
            )
            rows.append(
                {
                    "structure": structure,
                    "structure_display": source["structure_display"],
                    "subject_id": str(subject_id),
                    "cohort_diagnosis": source.get("cohort_diagnosis"),
                    "baseline_diagnosis": source.get("baseline_diagnosis"),
                    "source_scan_id": str(source["scan_id"]),
                    "target_scan_id": str(target["scan_id"]),
                    "source_visit": str(source["VISCODE"]),
                    "target_visit": str(target["VISCODE"]),
                    "source_visit_diagnosis": source.get("visit_diagnosis"),
                    "target_visit_diagnosis": target.get("visit_diagnosis"),
                    "source_age_years": safe_float(source.get("age_years")),
                    "target_age_years": safe_float(target.get("age_years")),
                    "midpoint_age_years": np.nanmean(
                        [safe_float(source.get("age_years")), safe_float(target.get("age_years"))]
                    ),
                    "source_month": source_month,
                    "target_month": target_month,
                    "delta_years": delta_years,
                    "source_volume_mm3": source_volume,
                    "target_volume_mm3": target_volume,
                    "delta_volume_mm3": target_volume - source_volume,
                    "signed_volume_change_pct_per_year": log_rate,
                    "absolute_volume_change_pct_per_year": abs(log_rate) if np.isfinite(log_rate) else np.nan,
                    "source_mask_volume_mm3": safe_float(source.get("mask_volume_mm3")),
                    "target_mask_volume_mm3": safe_float(target.get("mask_volume_mm3")),
                    "source_raw_volume_mm3": safe_float(source.get("raw_mesh_volume_mm3")),
                    "target_raw_volume_mm3": safe_float(target.get("raw_mesh_volume_mm3")),
                    "source_surface_area_mm2": safe_float(source.get("smooth_surface_area_mm2")),
                    "target_surface_area_mm2": safe_float(target.get("smooth_surface_area_mm2")),
                    "source_correspondence_mesh_path": source.get("correspondence_mesh_path"),
                    "target_correspondence_mesh_path": target.get("correspondence_mesh_path"),
                }
            )
    return pd.DataFrame.from_records(rows)


def subject_rate_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    metric_columns = [
        "signed_volume_change_pct_per_year",
        "absolute_volume_change_pct_per_year",
        "delta_volume_mm3",
        "midpoint_age_years",
    ]
    grouped = pairs.groupby(["structure", "structure_display", "subject_id", "cohort_diagnosis"], dropna=False)
    result = grouped[metric_columns].mean().reset_index()
    result = result.rename(columns={"midpoint_age_years": "mean_interval_age_years"})
    result["adjacent_pair_count"] = grouped.size().to_numpy()
    return result


def kabsch_align_points(target_vertices: np.ndarray, source_vertices: np.ndarray) -> np.ndarray:
    """Rigidly align source points to target points without scaling."""

    if target_vertices.shape != source_vertices.shape:
        raise ValueError(
            f"Correspondence vertex shape mismatch: target={target_vertices.shape}, source={source_vertices.shape}"
        )
    source_center = source_vertices.mean(axis=0)
    target_center = target_vertices.mean(axis=0)
    source_zero = source_vertices - source_center
    target_zero = target_vertices - target_center
    u_matrix, _, vt_matrix = np.linalg.svd(source_zero.T @ target_zero)
    rotation = u_matrix @ vt_matrix
    if np.linalg.det(rotation) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix
    return source_zero @ rotation + target_center


def pair_shape_measurements(
    pair: Mapping[str, Any], mesh_cache: dict[str, LoadedMesh] | None = None
) -> dict[str, Any]:
    """Measure correspondence displacement for one adjacent pair in mm/year."""

    cache = mesh_cache if mesh_cache is not None else {}

    def cached(path_value: object) -> LoadedMesh:
        path = str(path_value)
        if path not in cache:
            cache[path] = load_mesh(path)
        return cache[path]

    source = cached(pair["source_correspondence_mesh_path"])
    target = cached(pair["target_correspondence_mesh_path"])
    if not np.array_equal(source.faces, target.faces):
        raise ValueError("correspondence meshes do not have identical face connectivity")
    source_aligned = kabsch_align_points(target.vertices, source.vertices)
    displacement = target.vertices - source_aligned
    distance = np.linalg.norm(displacement, axis=1)
    delta_years = safe_float(pair["delta_years"])
    if not np.isfinite(delta_years) or delta_years <= 0.0:
        raise ValueError("pair has nonpositive elapsed time")
    return {
        "shape_vertex_count": int(len(target.vertices)),
        "shape_face_count": int(len(target.faces)),
        "shape_rms_displacement_mm_per_year": float(np.sqrt(np.mean(distance**2)) / delta_years),
        "shape_mean_displacement_mm_per_year": float(np.mean(distance) / delta_years),
        "shape_p95_displacement_mm_per_year": float(np.quantile(distance, 0.95) / delta_years),
        "shape_max_displacement_mm_per_year": float(np.max(distance) / delta_years),
    }


def add_pair_shape_measurements(
    pairs: pd.DataFrame,
    *,
    progress_name: str = "shape QC",
    every: int = 100,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()
    records: list[dict[str, Any]] = []
    cache: dict[str, LoadedMesh] = {}
    total = len(pairs)
    for index, row in enumerate(pairs.to_dict("records"), start=1):
        terminal_progress(progress_name, index, total, every=every)
        try:
            result = pair_shape_measurements(row, cache)
            result["shape_measurement_error"] = ""
        except Exception as exc:  # Keep processing even if one mesh is unreadable.
            result = {
                "shape_vertex_count": np.nan,
                "shape_face_count": np.nan,
                "shape_rms_displacement_mm_per_year": np.nan,
                "shape_mean_displacement_mm_per_year": np.nan,
                "shape_p95_displacement_mm_per_year": np.nan,
                "shape_max_displacement_mm_per_year": np.nan,
                "shape_measurement_error": f"{type(exc).__name__}: {exc}",
            }
        records.append(result)
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def median_mad_bounds(values: pd.Series, multiplier: float = 5.0) -> tuple[float, float, float, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return (np.nan, np.nan, np.nan, np.nan)
    median = float(finite.median())
    mad = float(np.median(np.abs(finite.to_numpy() - median)))
    scale = 1.4826 * mad
    if scale <= 1.0e-12:
        return (median, mad, -np.inf, np.inf)
    return (median, mad, median - multiplier * scale, median + multiplier * scale)


def bootstrap_mean_difference(
    group_a: Sequence[float], group_b: Sequence[float], *, seed: int = 42, draws: int = 2000
) -> dict[str, float]:
    first = np.asarray(group_a, dtype=float)
    second = np.asarray(group_b, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if len(first) < 2 or len(second) < 2:
        return {"mean_difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "cohens_d": np.nan}
    difference = float(second.mean() - first.mean())
    pooled_variance = ((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)) / (
        len(first) + len(second) - 2
    )
    cohens_d = difference / np.sqrt(pooled_variance) if pooled_variance > 0.0 else np.nan
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=float)
    for index in range(draws):
        sampled[index] = (
            rng.choice(second, size=len(second), replace=True).mean()
            - rng.choice(first, size=len(first), replace=True).mean()
        )
    return {
        "mean_difference": difference,
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "cohens_d": float(cohens_d),
    }


def pairwise_rate_statistics(subject_rates: pd.DataFrame, *, bootstrap_draws: int = 2000) -> pd.DataFrame:
    """CN/MCI/AD subject-level rate contrasts, separately for each structure."""

    rows: list[dict[str, Any]] = []
    for structure, frame in subject_rates.groupby("structure", sort=True):
        available_diagnoses = set(frame["cohort_diagnosis"].dropna().astype(str))
        comparisons = tuple(
            pair for pair in (("CN", "MCI"), ("MCI", "AD"), ("CN", "AD")) if set(pair).issubset(available_diagnoses)
        )
        for metric in ("signed_volume_change_pct_per_year", "absolute_volume_change_pct_per_year"):
            for diagnosis_a, diagnosis_b in comparisons:
                values_a = frame.loc[frame["cohort_diagnosis"].eq(diagnosis_a), metric]
                values_b = frame.loc[frame["cohort_diagnosis"].eq(diagnosis_b), metric]
                stats = bootstrap_mean_difference(values_a, values_b, draws=bootstrap_draws)
                rows.append(
                    {
                        "structure": structure,
                        "metric": metric,
                        "group_a": diagnosis_a,
                        "group_b": diagnosis_b,
                        "n_a": int(values_a.notna().sum()),
                        "n_b": int(values_b.notna().sum()),
                        **stats,
                    }
                )
    return pd.DataFrame.from_records(rows)


def save_shape_maps(
    pairs: pd.DataFrame,
    output_path: Path,
    *,
    every: int = 100,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Create diagnosis-average signed and absolute local shape-speed maps.

    Subject-average maps are used before group averaging so participants with
    many visits do not dominate a cohort.  Positive signed speed denotes
    outward change relative to the target mesh normals after rigid alignment.
    """

    valid = pairs.loc[
        pairs["cohort_diagnosis"].isin(DIAGNOSES)
        & pairs["delta_years"].gt(0.0)
        & pairs["source_correspondence_mesh_path"].notna()
        & pairs["target_correspondence_mesh_path"].notna()
    ].copy()
    mesh_cache: dict[str, LoadedMesh] = {}
    subject_maps: dict[tuple[str, str, str], list[np.ndarray]] = {}
    faces_by_structure: dict[str, np.ndarray] = {}
    vertices_by_structure: dict[str, list[np.ndarray]] = {}
    total = len(valid)
    for index, row in enumerate(valid.to_dict("records"), start=1):
        terminal_progress("local shape maps", index, total, every=every)
        try:
            source_path = str(row["source_correspondence_mesh_path"])
            target_path = str(row["target_correspondence_mesh_path"])
            if source_path not in mesh_cache:
                mesh_cache[source_path] = load_mesh(source_path)
            if target_path not in mesh_cache:
                mesh_cache[target_path] = load_mesh(target_path)
            source = mesh_cache[source_path]
            target = mesh_cache[target_path]
            if not np.array_equal(source.faces, target.faces):
                raise ValueError("faces differ")
            source_aligned = kabsch_align_points(target.vertices, source.vertices)
            displacement = target.vertices - source_aligned
            # trimesh is used only to obtain the target normals; no repair is performed.
            import trimesh

            target_mesh = trimesh.Trimesh(vertices=target.vertices, faces=target.faces, process=False)
            normals = np.asarray(target_mesh.vertex_normals, dtype=float)
            signed_speed = np.einsum("ij,ij->i", displacement, normals) / float(row["delta_years"])
            key = (str(row["structure"]), str(row["cohort_diagnosis"]), str(row["subject_id"]))
            subject_maps.setdefault(key, []).append(signed_speed)
            faces_by_structure.setdefault(str(row["structure"]), target.faces)
            vertices_by_structure.setdefault(str(row["structure"]), []).append(target.vertices)
        except Exception as exc:
            print(f"[local shape maps] skipped {row['source_scan_id']} -> {row['target_scan_id']}: {exc}", flush=True)

    arrays: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    for structure in sorted(faces_by_structure):
        arrays[f"{structure}__faces"] = faces_by_structure[structure]
        arrays[f"{structure}__vertices"] = np.mean(np.stack(vertices_by_structure[structure]), axis=0)
        for diagnosis in DIAGNOSES:
            maps = [
                np.mean(np.stack(values), axis=0)
                for (map_structure, map_diagnosis, _), values in subject_maps.items()
                if map_structure == structure and map_diagnosis == diagnosis
            ]
            if not maps:
                continue
            signed = np.mean(np.stack(maps), axis=0)
            absolute = np.mean(np.abs(np.stack(maps)), axis=0)
            arrays[f"{structure}__{diagnosis}__signed_speed"] = signed
            arrays[f"{structure}__{diagnosis}__absolute_speed"] = absolute
            threshold = float(np.quantile(np.clip(-signed, 0.0, None), 0.90))
            hotspot = np.clip(-signed, 0.0, None) >= threshold
            summary_rows.append(
                {
                    "structure": structure,
                    "diagnosis": diagnosis,
                    "subject_count": len(maps),
                    "mean_signed_speed_mm_per_year": float(np.mean(signed)),
                    "mean_absolute_speed_mm_per_year": float(np.mean(absolute)),
                    "peak_inward_speed_mm_per_year": float(np.max(np.clip(-signed, 0.0, None))),
                    "inward_hotspot_vertex_fraction": float(np.mean(hotspot)),
                }
            )
    ensure_directory(output_path.parent)
    np.savez_compressed(output_path, **arrays)
    return pd.DataFrame.from_records(summary_rows), arrays


def input_validation_summary(input_root: Path, records: pd.DataFrame) -> dict[str, Any]:
    summary_path = input_root / "reports" / "validation_summary.json"
    prior_validation: dict[str, Any] = {}
    if summary_path.is_file():
        prior_validation = json.loads(summary_path.read_text(encoding="utf-8"))
    cohort_counts = (
        records.drop_duplicates(["structure", "scan_id"])["cohort_diagnosis"].value_counts(dropna=False).to_dict()
        if not records.empty
        else {}
    )
    return {
        "input_root": str(input_root),
        "records": int(len(records)),
        "subjects": int(records["subject_id"].nunique()) if not records.empty else 0,
        "structures": sorted(records["structure"].unique().tolist()) if not records.empty else [],
        "cohort_diagnosis_counts_per_structure": cohort_counts,
        "pipeline_validation": prior_validation,
    }
