#!/usr/bin/env python3
"""Build a leakage-controlled reference for longitudinal surface velocity.

This script does not claim to recover directly measured instantaneous motion.
It compares interval differences with smoother subject-trajectory derivatives,
selects the trajectory estimator on validation subjects, freezes that choice,
and evaluates it once on test subjects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
DEFAULT_CONFIG = TASK / "configs" / "velocity_reference_audit.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from velocity_core import (
    Candidate,
    align_training_and_holdout,
    bootstrap_mean,
    candidates_from_config,
    consecutive_cosines,
    fit_trajectory,
    generalized_rigid_alignment,
    mesh_volume,
    prediction_metrics,
    subject_group_mean,
    vector_cosine,
    vertex_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-subjects", type=int, default=None, help="Per split; intended only for a smoke test")
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--skip-surface-distances", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(root: Path, force: bool) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    manifest = root / "manifest.json"
    if manifest.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite completed output without --force: {root}")
    tables, arrays = root / "tables", root / "arrays"
    tables.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)
    return tables, arrays


def choose_subjects(archive: dict[str, np.ndarray], limit: int | None, minimum_visits: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    subjects = archive["subject_ids"].astype(str)
    visit_subjects = archive["visit_subject_ids"].astype(str)
    counts = pd.Series(visit_subjects).value_counts()
    eligible = np.asarray([subject for subject in subjects if int(counts.get(subject, 0)) >= minimum_visits], dtype=str)
    excluded = [
        {"subject_id": subject, "reason": f"fewer_than_{minimum_visits}_visits", "visits": int(counts.get(subject, 0))}
        for subject in subjects
        if int(counts.get(subject, 0)) < minimum_visits
    ]
    if limit is None or limit >= len(eligible):
        return eligible, excluded
    diagnoses = archive["subject_diagnoses"].astype(str)
    diagnosis_by_subject = dict(zip(subjects, diagnoses))
    selected: list[str] = []
    for diagnosis in ("CN", "AD"):
        quota = max(1, limit // 2)
        selected.extend([subject for subject in eligible if diagnosis_by_subject[subject] == diagnosis][:quota])
    for subject in eligible:
        if len(selected) >= limit:
            break
        if subject not in selected:
            selected.append(subject)
    return np.asarray(selected[:limit], dtype=str), excluded


def load_inputs(config: dict[str, Any], split: str, limit: int | None) -> dict[str, Any]:
    task128 = resolve(config["source_task_128"])
    registry = read_json(task128 / "configs" / "representations.json")
    bulk = resolve(registry["output_root"])
    archive_path = bulk / "representations" / "pca128" / f"{split}_subject_sequences_128.npz"
    vertices_path = resolve(registry["ae_bulk_root"]) / "cache" / f"adni_{split}_V.npy"
    pca_root = resolve(config["pca_model_root"])
    with np.load(archive_path, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    vertices_all = np.load(vertices_path, mmap_mode="r")
    selected_subjects, excluded_subjects = choose_subjects(archive, limit, int(config["minimum_visits"]))
    mask = np.isin(archive["visit_subject_ids"].astype(str), selected_subjects)
    visit_indices = np.flatnonzero(mask)
    vertices = np.asarray(vertices_all[visit_indices], dtype=np.float64)
    metadata = pd.DataFrame({
        "scan_id": archive["visit_scan_ids"][mask].astype(str),
        "subject_id": archive["visit_subject_ids"][mask].astype(str),
        "split": archive["visit_splits"][mask].astype(str),
        "diagnosis": archive["visit_diagnoses"][mask].astype(str),
        "label_ad": archive["visit_label_ad"][mask].astype(np.int64),
        "visit_order": archive["visit_orders"][mask].astype(np.int64),
        "age_years": archive["visit_age_years"][mask].astype(np.float64),
        "volume_archive_mm3": archive["visit_volume_mm3"][mask].astype(np.float64),
    })
    faces = np.load(pca_root / "faces.npy", allow_pickle=False).astype(np.int64)
    pca_mean = np.load(pca_root / "mean.npy", allow_pickle=False).astype(np.float64)
    pca_components = np.load(pca_root / "components_150.npy", allow_pickle=False).astype(np.float64)
    manifest = pd.read_csv(resolve(config["source_manifest"]), dtype={"scan_id": str, "subject_id": str})
    manifest = manifest.set_index("scan_id").loc[metadata.scan_id].reset_index()
    if not np.array_equal(manifest.scan_id.astype(str), metadata.scan_id.astype(str)):
        raise ValueError(f"{split}: source manifest scan order mismatch")
    if not np.array_equal(manifest.split.astype(str), metadata.split.astype(str)):
        raise ValueError(f"{split}: split mismatch between archive and source manifest")
    if not np.array_equal(manifest.subject_id.astype(str), metadata.subject_id.astype(str)):
        raise ValueError(f"{split}: subject mismatch between archive and source manifest")
    if vertices.ndim != 3 or vertices.shape[1:] != (len(pca_mean) // 3, 3):
        raise ValueError(f"{split}: unexpected vertex shape {vertices.shape}")
    if not np.isfinite(vertices).all() or not np.isfinite(metadata.age_years).all():
        raise ValueError(f"{split}: non-finite physical input")
    if set(metadata.diagnosis) - {"CN", "AD"}:
        raise ValueError(f"{split}: diagnoses outside CN/AD")
    for subject, frame in metadata.groupby("subject_id", sort=False):
        ages = frame.age_years.to_numpy()
        if len(ages) < int(config["minimum_visits"]) or not np.all(np.diff(ages) > 0.0):
            raise ValueError(f"{split}: invalid visit sequence for {subject}: ages={ages.tolist()}")
    computed = np.asarray([mesh_volume(mesh, faces) for mesh in vertices])
    relative = np.abs(computed - metadata.volume_archive_mm3.to_numpy()) / np.maximum(computed, 1.0)
    if float(np.quantile(relative, 0.99)) > 1.0e-4:
        raise ValueError(f"{split}: mesh/archive volume mismatch, 99th percentile={np.quantile(relative, 0.99):.6g}")
    metadata["volume_mm3"] = computed
    return {
        "archive": archive,
        "archive_path": archive_path,
        "vertices_path": vertices_path,
        "vertices": vertices,
        "faces": faces,
        "metadata": metadata,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "excluded_subjects": excluded_subjects,
    }


def subject_blocks(metadata: pd.DataFrame):
    for subject, indices in metadata.groupby("subject_id", sort=False).groups.items():
        yield str(subject), np.asarray(list(indices), dtype=np.int64)


def evaluate_leave_one_out(
    split: str,
    data: dict[str, Any],
    candidates: list[Candidate],
    huber_delta: float,
    surface_distances: bool,
) -> pd.DataFrame:
    vertices = data["vertices"]
    metadata = data["metadata"]
    rows: list[dict[str, Any]] = []
    for subject, indices in subject_blocks(metadata):
        ages = metadata.loc[indices, "age_years"].to_numpy(dtype=np.float64)
        meshes = vertices[indices]
        diagnosis = str(metadata.loc[indices[0], "diagnosis"])
        for held_local in range(1, len(indices) - 1):
            keep = np.ones(len(indices), dtype=bool)
            keep[held_local] = False
            train_raw, held_raw = meshes[keep], meshes[held_local]
            train_rigid, held_rigid = align_training_and_holdout(train_raw, held_raw)
            base = {
                "split": split,
                "subject_id": subject,
                "diagnosis": diagnosis,
                "scan_id": str(metadata.loc[indices[held_local], "scan_id"]),
                "held_age_years": float(ages[held_local]),
                "n_subject_visits": len(indices),
                "n_fit_visits": int(keep.sum()),
                "previous_gap_years": float(ages[held_local] - ages[held_local - 1]),
                "next_gap_years": float(ages[held_local + 1] - ages[held_local]),
            }
            previous = meshes[held_local - 1]
            fraction = (ages[held_local] - ages[held_local - 1]) / (ages[held_local + 1] - ages[held_local - 1])
            interpolation = meshes[held_local - 1] + fraction * (meshes[held_local + 1] - meshes[held_local - 1])
            for name, label, predicted in (
                ("baseline_no_change", "No change from previous visit", previous),
                ("baseline_neighbor_interpolation", "Neighbour interpolation", interpolation),
            ):
                row = dict(base, candidate=name, candidate_label=label, candidate_kind="baseline")
                row.update(prediction_metrics(predicted, held_raw, data["faces"], surface_distances))
                rows.append(row)
            for candidate in candidates:
                if int(keep.sum()) < candidate.minimum_fit_visits:
                    continue
                train = train_raw if candidate.alignment == "raw" else train_rigid
                observed = held_raw if candidate.alignment == "raw" else held_rigid
                model = fit_trajectory(
                    ages[keep], train, candidate, data["pca_mean"], data["pca_components"], huber_delta
                )
                predicted = model.predict(ages[held_local])[0]
                row = dict(base, candidate=candidate.name, candidate_label=candidate.label, candidate_kind="trajectory")
                row.update(prediction_metrics(predicted, observed, data["faces"], surface_distances))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_folds(folds: pd.DataFrame, candidates: list[Candidate]) -> pd.DataFrame:
    metric_columns = [column for column in ("vertex_rmse_mm", "mean_vertex_error_mm", "assd_mm", "hd95_mm", "volume_abs_error_mm3", "log_volume_abs_error") if column in folds]
    subject = folds.groupby(["candidate", "candidate_label", "candidate_kind", "subject_id"], as_index=False)[metric_columns].mean()
    rows = []
    for keys, frame in subject.groupby(["candidate", "candidate_label", "candidate_kind"], sort=False):
        candidate, label, kind = keys
        source = folds[folds.candidate.eq(candidate)]
        row = {
            "candidate": candidate,
            "candidate_label": label,
            "candidate_kind": kind,
            "eligible_subjects": int(frame.subject_id.nunique()),
            "held_out_visits": len(source),
            "subject_mean_vertex_rmse_mm": float(frame.vertex_rmse_mm.mean()),
            "visit_mean_vertex_rmse_mm": float(source.vertex_rmse_mm.mean()),
        }
        for metric in metric_columns[1:]:
            row[f"subject_mean_{metric}"] = float(frame[metric].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    order = {candidate.name: index for index, candidate in enumerate(candidates)}
    result["configured_order"] = result.candidate.map(order).fillna(10_000).astype(int)
    return result.sort_values(["candidate_kind", "subject_mean_vertex_rmse_mm", "configured_order"]).reset_index(drop=True)


def select_winner(summary: pd.DataFrame, candidates: list[Candidate]) -> Candidate:
    allowed = summary[summary.candidate_kind.eq("trajectory")].copy()
    maximum = int(allowed.eligible_subjects.max())
    allowed = allowed[allowed.eligible_subjects.eq(maximum)]
    winner_name = str(allowed.sort_values(["subject_mean_vertex_rmse_mm", "configured_order"]).iloc[0].candidate)
    return next(candidate for candidate in candidates if candidate.name == winner_name)


def interval_fields(meshes: np.ndarray, ages: np.ndarray, faces: np.ndarray) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    rows: list[dict[str, float]] = []
    fields: list[np.ndarray] = []
    for index in range(len(meshes) - 1):
        gap = float(ages[index + 1] - ages[index])
        velocity = (meshes[index + 1] - meshes[index]) / gap
        midpoint = (meshes[index + 1] + meshes[index]) / 2.0
        normals, area, volume = vertex_geometry(midpoint, faces)
        normal = np.sum(velocity * normals, axis=1)
        fields.append(normal)
        weight = area / max(area.sum(), 1.0e-12)
        rows.append({
            "interval_index": index,
            "age_mid_years": float((ages[index + 1] + ages[index]) / 2.0),
            "gap_years": gap,
            "vector_rms_mm_per_year": float(np.sqrt(np.mean(np.sum(velocity**2, axis=1)))),
            "normal_mean_mm_per_year": float(np.sum(weight * normal)),
            "normal_abs_mean_mm_per_year": float(np.sum(weight * np.abs(normal))),
            "normal_rms_mm_per_year": float(np.sqrt(np.sum(weight * normal**2))),
            "log_volume_rate_percent_per_year": float(100.0 * (np.log(mesh_volume(meshes[index + 1], faces)) - np.log(mesh_volume(meshes[index], faces))) / gap),
            "reference_volume_mm3": float(volume),
        })
    return rows, fields


def scan_difference_field(meshes: np.ndarray, ages: np.ndarray, index: int) -> np.ndarray:
    if index == 0:
        left, right = 0, 1
    elif index == len(meshes) - 1:
        left, right = len(meshes) - 2, len(meshes) - 1
    else:
        left, right = index - 1, index + 1
    return (meshes[right] - meshes[left]) / float(ages[right] - ages[left])


def build_reference(
    split: str,
    data: dict[str, Any],
    winner: Candidate,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    metadata = data["metadata"]
    vertices = data["vertices"]
    faces = data["faces"]
    scan_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    fields_fit = np.empty((len(metadata), len(vertices[0])), dtype=np.float32)
    fields_adjacent = np.empty_like(fields_fit)
    fields_endpoint = np.empty_like(fields_fit)
    fitted_positions = np.empty_like(vertices, dtype=np.float32)
    aligned_observed = np.empty_like(vertices, dtype=np.float32)
    for subject, indices in subject_blocks(metadata):
        ages = metadata.loc[indices, "age_years"].to_numpy(dtype=np.float64)
        raw = vertices[indices]
        rigid, _ = generalized_rigid_alignment(raw)
        working = raw if winner.alignment == "raw" else rigid
        aligned_observed[indices] = working.astype(np.float32)
        model = fit_trajectory(ages, working, winner, data["pca_mean"], data["pca_components"], config["huber_delta"])
        predicted = model.predict(ages)
        derivative = model.derivative(ages)
        fitted_positions[indices] = predicted.astype(np.float32)
        endpoint = (working[-1] - working[0]) / float(ages[-1] - ages[0])
        raw_interval, raw_fields = interval_fields(raw, ages, faces)
        rigid_interval, rigid_fields = interval_fields(rigid, ages, faces)
        raw_vectors = [(raw[i + 1] - raw[i]) / (ages[i + 1] - ages[i]) for i in range(len(raw) - 1)]
        rigid_vectors = [(rigid[i + 1] - rigid[i]) / (ages[i + 1] - ages[i]) for i in range(len(rigid) - 1)]
        raw_cosines = consecutive_cosines(raw_vectors)
        rigid_cosines = consecutive_cosines(rigid_vectors)
        for local, (raw_row, rigid_row) in enumerate(zip(raw_interval, rigid_interval)):
            interval_rows.append({
                "split": split,
                "subject_id": subject,
                "diagnosis": str(metadata.loc[indices[0], "diagnosis"]),
                "source_scan_id": str(metadata.loc[indices[local], "scan_id"]),
                "target_scan_id": str(metadata.loc[indices[local + 1], "scan_id"]),
                **{f"registered_{key}": value for key, value in raw_row.items()},
                **{f"rigid_removed_{key}": value for key, value in rigid_row.items()},
                "centroid_speed_mm_per_year": float(np.linalg.norm(raw[local + 1].mean(axis=0) - raw[local].mean(axis=0)) / raw_row["gap_years"]),
                "rigid_removed_to_registered_speed_ratio": float(rigid_row["vector_rms_mm_per_year"] / max(raw_row["vector_rms_mm_per_year"], 1.0e-12)),
                "next_registered_velocity_cosine": float(raw_cosines[local]) if local < len(raw_cosines) else np.nan,
                "next_rigid_removed_velocity_cosine": float(rigid_cosines[local]) if local < len(rigid_cosines) else np.nan,
            })
        for local, global_index in enumerate(indices):
            normals, area, volume = vertex_geometry(predicted[local], faces)
            weight = area / max(area.sum(), 1.0e-12)
            adjacent_vector = scan_difference_field(working, ages, local)
            fit_normal = np.sum(derivative[local] * normals, axis=1)
            adjacent_normal = np.sum(adjacent_vector * normals, axis=1)
            endpoint_normal = np.sum(endpoint * normals, axis=1)
            fields_fit[global_index] = fit_normal.astype(np.float32)
            fields_adjacent[global_index] = adjacent_normal.astype(np.float32)
            fields_endpoint[global_index] = endpoint_normal.astype(np.float32)
            scan_rows.append({
                "split": split,
                "subject_id": subject,
                "scan_id": str(metadata.loc[global_index, "scan_id"]),
                "diagnosis": str(metadata.loc[global_index, "diagnosis"]),
                "label_ad": int(metadata.loc[global_index, "label_ad"]),
                "age_years": float(ages[local]),
                "visit_order": int(metadata.loc[global_index, "visit_order"]),
                "selected_estimator": winner.name,
                "selected_estimator_label": winner.label,
                "reference_label": "Observed — fitted trajectory",
                "fit_vector_rms_mm_per_year": float(np.sqrt(np.mean(np.sum(derivative[local] ** 2, axis=1)))),
                "fit_normal_mean_mm_per_year": float(np.sum(weight * fit_normal)),
                "fit_normal_abs_mean_mm_per_year": float(np.sum(weight * np.abs(fit_normal))),
                "fit_normal_rms_mm_per_year": float(np.sqrt(np.sum(weight * fit_normal**2))),
                "adjacent_vector_rms_mm_per_year": float(np.sqrt(np.mean(np.sum(adjacent_vector**2, axis=1)))),
                "adjacent_normal_rms_mm_per_year": float(np.sqrt(np.sum(weight * adjacent_normal**2))),
                "endpoint_vector_rms_mm_per_year": float(np.sqrt(np.mean(np.sum(endpoint**2, axis=1)))),
                "endpoint_normal_rms_mm_per_year": float(np.sqrt(np.sum(weight * endpoint_normal**2))),
                "adjacent_to_fit_speed_ratio": float(np.sqrt(np.mean(np.sum(adjacent_vector**2, axis=1))) / max(np.sqrt(np.mean(np.sum(derivative[local]**2, axis=1))), 1.0e-12)),
                "adjacent_fit_vector_cosine": vector_cosine(adjacent_vector, derivative[local]),
                "fit_surface_integral_log_volume_rate_percent_per_year": float(100.0 * np.sum(area * fit_normal) / max(volume, 1.0e-12)),
                "observed_volume_mm3": float(metadata.loc[global_index, "volume_mm3"]),
                "fitted_volume_mm3": float(mesh_volume(predicted[local], faces)),
            })
        if len(indices) >= 4:
            full_at_ages = derivative
            for held_local in range(len(indices)):
                keep = np.ones(len(indices), dtype=bool)
                keep[held_local] = False
                reduced = fit_trajectory(ages[keep], working[keep], winner, data["pca_mean"], data["pca_components"], config["huber_delta"])
                held_derivative = reduced.derivative(ages[held_local])[0]
                full_derivative = full_at_ages[held_local]
                stability_rows.append({
                    "split": split,
                    "subject_id": subject,
                    "diagnosis": str(metadata.loc[indices[0], "diagnosis"]),
                    "held_scan_id": str(metadata.loc[indices[held_local], "scan_id"]),
                    "n_subject_visits": len(indices),
                    "derivative_cosine": vector_cosine(held_derivative, full_derivative),
                    "relative_derivative_error": float(np.linalg.norm(held_derivative - full_derivative) / max(np.linalg.norm(full_derivative), 1.0e-12)),
                    "speed_ratio": float(np.linalg.norm(held_derivative) / max(np.linalg.norm(full_derivative), 1.0e-12)),
                })
    arrays = {
        "faces": faces.astype(np.int32),
        "template_vertices": fitted_positions.mean(axis=0).astype(np.float32),
        "fitted_positions": fitted_positions,
        "aligned_observed": aligned_observed,
        "fitted_normal_velocity": fields_fit,
        "adjacent_normal_velocity": fields_adjacent,
        "endpoint_normal_velocity": fields_endpoint,
        "scan_ids": metadata.scan_id.to_numpy(dtype="U32"),
        "subject_ids": metadata.subject_id.to_numpy(dtype="U32"),
        "diagnoses": metadata.diagnosis.to_numpy(dtype="U4"),
        "ages_years": metadata.age_years.to_numpy(dtype=np.float32),
    }
    for diagnosis in ("CN", "AD"):
        arrays[f"fitted_group_{diagnosis.lower()}"] = subject_group_mean(fields_fit, arrays["subject_ids"], arrays["diagnoses"], diagnosis).astype(np.float32)
        arrays[f"adjacent_group_{diagnosis.lower()}"] = subject_group_mean(fields_adjacent, arrays["subject_ids"], arrays["diagnoses"], diagnosis).astype(np.float32)
        arrays[f"endpoint_group_{diagnosis.lower()}"] = subject_group_mean(fields_endpoint, arrays["subject_ids"], arrays["diagnoses"], diagnosis).astype(np.float32)
    return pd.DataFrame(scan_rows), pd.DataFrame(interval_rows), pd.DataFrame(stability_rows), arrays


def group_and_volume_tables(scan: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    slopes = []
    for (split, subject), frame in scan.groupby(["split", "subject_id"], sort=False):
        ages = frame.age_years.to_numpy(dtype=np.float64)
        centered = ages - ages.mean()
        observed_log = np.log(frame.observed_volume_mm3.to_numpy(dtype=np.float64))
        fitted_log = np.log(frame.fitted_volume_mm3.to_numpy(dtype=np.float64))
        denominator = float(np.sum(centered**2))
        slopes.append({
            "split": split,
            "subject_id": subject,
            "diagnosis": str(frame.diagnosis.iloc[0]),
            "n_visits": len(frame),
            "age_span_years": float(ages.max() - ages.min()),
            "observed_log_volume_slope_percent_per_year": float(100.0 * np.sum(centered * (observed_log - observed_log.mean())) / denominator),
            "fitted_log_volume_slope_percent_per_year": float(100.0 * np.sum(centered * (fitted_log - fitted_log.mean())) / denominator),
        })
    subject = pd.DataFrame(slopes)
    rows = []
    for split, frame in subject.groupby("split"):
        for metric in ("observed_log_volume_slope_percent_per_year", "fitted_log_volume_slope_percent_per_year"):
            values = {}
            for diagnosis in ("CN", "AD"):
                values[diagnosis] = frame.loc[frame.diagnosis.eq(diagnosis), metric].to_numpy()
                mean, low, high = bootstrap_mean(values[diagnosis], bootstrap, seed + len(rows))
                rows.append({"split": split, "metric": metric, "contrast": diagnosis, "estimate": mean, "ci95_low": low, "ci95_high": high, "subjects": len(values[diagnosis])})
            generator = np.random.default_rng(seed + 100 + len(rows))
            draws = []
            for _ in range(bootstrap):
                ad = generator.choice(values["AD"], size=len(values["AD"]), replace=True).mean()
                cn = generator.choice(values["CN"], size=len(values["CN"]), replace=True).mean()
                draws.append(ad - cn)
            rows.append({"split": split, "metric": metric, "contrast": "AD minus CN", "estimate": float(values["AD"].mean() - values["CN"].mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)), "subjects": len(frame)})
    return subject, pd.DataFrame(rows)


def mixed_effects_table(scan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return pd.DataFrame([{"status": "statsmodels unavailable"}])
    for split, frame in scan.groupby("split"):
        work = frame.copy()
        work["log_volume"] = np.log(work.observed_volume_mm3)
        work["age_between"] = work.groupby("subject_id").age_years.transform("mean")
        work["age_between"] -= work.age_between.mean()
        work["age_within"] = work.age_years - work.groupby("subject_id").age_years.transform("mean")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.mixedlm(
                    "log_volume ~ age_within * label_ad + age_between + label_ad",
                    work,
                    groups=work["subject_id"],
                    re_formula="1",
                ).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            for term in result.params.index:
                if term.startswith("Group"):
                    continue
                rows.append({
                    "split": split,
                    "term": term,
                    "estimate": float(result.params[term]),
                    "standard_error": float(result.bse[term]),
                    "p_value": float(result.pvalues[term]),
                    "converged": bool(result.converged),
                    "subjects": int(work.subject_id.nunique()),
                    "scans": len(work),
                })
        except Exception as error:
            rows.append({"split": split, "term": "fit_error", "estimate": np.nan, "standard_error": np.nan, "p_value": np.nan, "converged": False, "subjects": int(work.subject_id.nunique()), "scans": len(work), "message": str(error)})
    return pd.DataFrame(rows)


def latent_reference_table(config: dict[str, Any], winner: Candidate, selected_scans: set[str] | None = None) -> pd.DataFrame:
    task128 = resolve(config["source_task_128"])
    registry = read_json(task128 / "configs" / "representations.json")
    bulk = resolve(registry["output_root"])
    sources = []
    for representation, method, label in (
        ("pca128", "pca", "PCA"),
        ("spiralnet128", "spiral", "Spiral"),
        ("adaptive128", "adaptive", "Adaptive"),
    ):
        sources.append((bulk / "representations" / representation / "test_subject_sequences_128.npz", "visit_latent_standardized_128", method, label))
    sources.append((resolve(config["inr_test_archive"]), "visit_latent_standardized_256", "inr", "INR"))
    rows = []
    latent_candidate = Candidate("latent_fitted", "Fitted latent trajectory", "raw", "surface", winner.degree, robust=winner.robust, ridge=winner.ridge)
    for path, key, method, label in sources:
        with np.load(path, allow_pickle=False) as loaded:
            scans = loaded["visit_scan_ids"].astype(str)
            subjects = loaded["visit_subject_ids"].astype(str)
            diagnoses = loaded["visit_diagnoses"].astype(str)
            ages = loaded["visit_age_years"].astype(np.float64)
            latent = loaded[key].astype(np.float64)
        for subject in np.unique(subjects):
            indices = np.flatnonzero(subjects == subject)
            model = fit_trajectory(ages[indices], latent[indices], latent_candidate, huber_delta=config["huber_delta"])
            fitted = model.derivative(ages[indices]).reshape(len(indices), -1)
            endpoint = (latent[indices[-1]] - latent[indices[0]]) / (ages[indices[-1]] - ages[indices[0]])
            for local, index in enumerate(indices):
                if selected_scans is not None and scans[index] not in selected_scans:
                    continue
                adjacent = scan_difference_field(latent[indices, :, None], ages[indices], local).reshape(-1)
                rows.append({
                    "method": method,
                    "method_label": label,
                    "subject_id": subject,
                    "scan_id": scans[index],
                    "diagnosis": diagnoses[index],
                    "age_years": float(ages[index]),
                    "fitted_rms_per_coordinate_per_year": float(np.sqrt(np.mean(fitted[local] ** 2))),
                    "adjacent_rms_per_coordinate_per_year": float(np.sqrt(np.mean(adjacent**2))),
                    "endpoint_rms_per_coordinate_per_year": float(np.sqrt(np.mean(endpoint**2))),
                    "adjacent_fit_cosine": vector_cosine(adjacent, fitted[local]),
                    "coordinate_contract": "train-standardized latent coordinates; compare only within representation",
                })
    return pd.DataFrame(rows)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    print(f"saved {path} ({len(frame)} rows)", flush=True)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    root = resolve(args.output_root if args.output_root is not None else config["output_root"])
    tables, arrays_dir = prepare_output(root, args.force)
    bootstrap = int(args.bootstrap if args.bootstrap is not None else config["bootstrap_repetitions"])
    if bootstrap < 100:
        raise ValueError("Use at least 100 subject-level bootstrap repetitions")
    if args.max_subjects is not None and args.max_subjects < 4:
        raise ValueError("Smoke test needs at least four subjects per split")
    candidates = candidates_from_config(config)
    input_data = {split: load_inputs(config, split, args.max_subjects) for split in config["splits"]}
    folds = {}
    summaries = {}
    for split, data in input_data.items():
        print(f"leave-one-interior-visit-out: {split}", flush=True)
        folds[split] = evaluate_leave_one_out(split, data, candidates, config["huber_delta"], not args.skip_surface_distances)
        summaries[split] = summarize_folds(folds[split], candidates)
        save_table(folds[split], tables / f"{split}_heldout_visit_predictions.csv")
        save_table(summaries[split], tables / f"{split}_estimator_summary.csv")
    winner = select_winner(summaries["val"], candidates)
    (root / "selected_estimator.json").write_text(json.dumps({
        "selected_on": "validation subjects only",
        "selection_metric": config["selection_metric"],
        "candidate": winner.name,
        "candidate_label": winner.label,
        "alignment": winner.alignment,
        "degree": winner.degree,
        "basis": winner.basis,
        "components": winner.components,
        "robust": winner.robust,
        "test_was_not_used_for_selection": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"frozen validation winner: {winner.label}", flush=True)
    all_scan = []
    all_interval = []
    all_stability = []
    arrays = {}
    for split, data in input_data.items():
        scan, interval, stability, split_arrays = build_reference(split, data, winner, config)
        all_scan.append(scan)
        all_interval.append(interval)
        all_stability.append(stability)
        np.savez_compressed(arrays_dir / f"{split}_velocity_reference.npz", **split_arrays)
        arrays[split] = split_arrays
    scan = pd.concat(all_scan, ignore_index=True)
    interval = pd.concat(all_interval, ignore_index=True)
    stability = pd.concat(all_stability, ignore_index=True)
    save_table(scan, tables / "velocity_reference_per_scan.csv")
    save_table(interval, tables / "adjacent_interval_diagnostics.csv")
    save_table(stability, tables / "leave_one_visit_out_derivative_stability.csv")
    subject_volume, group_volume = group_and_volume_tables(scan, bootstrap, int(config["random_seed"]))
    save_table(subject_volume, tables / "subject_volume_slopes.csv")
    save_table(group_volume, tables / "diagnosis_volume_slope_bootstrap.csv")
    save_table(mixed_effects_table(scan), tables / "volume_within_between_mixed_effects.csv")
    selected_scans = set(arrays["test"]["scan_ids"].astype(str))
    save_table(latent_reference_table(config, winner, selected_scans), tables / "latent_velocity_reference_per_scan.csv")
    input_hashes = {
        "config": sha256(config_path),
        "pca_faces": sha256(resolve(config["pca_model_root"]) / "faces.npy"),
        "pca_components": sha256(resolve(config["pca_model_root"]) / "components_150.npy"),
    }
    manifest = {
        "status": "complete",
        "mode": "smoke" if args.max_subjects is not None else "full",
        "max_subjects_per_split": args.max_subjects,
        "selected_estimator": winner.name,
        "selected_estimator_label": winner.label,
        "selection_split": "val",
        "evaluation_split": "test",
        "bootstrap_repetitions": bootstrap,
        "surface_distances_computed": not args.skip_surface_distances,
        "input_hashes": input_hashes,
        "counts": {split: {"subjects": int(data["metadata"].subject_id.nunique()), "scans": len(data["metadata"])} for split, data in input_data.items()},
        "excluded_subjects": {split: data["excluded_subjects"] for split, data in input_data.items()},
        "interpretation": {
            "instantaneous_ground_truth_exists": False,
            "observed_fitted_trajectory": "A smoother estimate inferred from repeated registered meshes; it is the primary reference, not directly measured instantaneous motion.",
            "observed_adjacent_interval": "An interval-average change divided by elapsed years; it contains differenced visit-specific error and is not an instantaneous ground truth.",
            "observed_first_to_last": "A long-interval average rate; stable but unable to show within-subject acceleration.",
        },
        "source_meshes_modified": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
