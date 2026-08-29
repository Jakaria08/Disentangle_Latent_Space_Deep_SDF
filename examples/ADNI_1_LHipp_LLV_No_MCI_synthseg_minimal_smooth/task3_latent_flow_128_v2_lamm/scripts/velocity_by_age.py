#!/usr/bin/env python3
"""Compare N3 ensemble instantaneous surface velocity with observed age references.

The model quantity is the zero-horizon direct-C4 velocity pushed through each frozen
decoder with a Jacobian-vector product and averaged in mesh space.  The primary observed
reference is the derivative of the validation-selected subject trajectory smoother.  The
adjacent-scan finite difference is shown separately and is never called instantaneous GT.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _bootstrap import activate
from evaluate_n3_ensemble import age_norm_per_year, load_member, parse_assignment

activate()
import common as C


AGE_EDGES = np.asarray([-np.inf, 70.0, 75.0, 80.0, 85.0, np.inf])
AGE_LABELS = ("<70", "70-75", "75-80", "80-85", "85+")
AGE_CENTERS = np.asarray([67.5, 72.5, 77.5, 82.5, 87.5])
SERIES = {
    "model": "LAMM N3 instantaneous",
    "fitted": "Observed - fitted trajectory",
    "adjacent": "Observed - adjacent interval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def vertex_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_area = np.linalg.norm(cross, axis=1) * 0.5
    normals = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(normals, faces[:, 0], cross)
    np.add.at(normals, faces[:, 1], cross)
    np.add.at(normals, faces[:, 2], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    signed = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum() / 6.0
    if signed < 0.0:
        normals *= -1.0
    area = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(area, faces.reshape(-1), np.repeat(face_area / 3.0, 3))
    return normals, area


def assert_visit_alignment(members) -> None:
    first = members[0].archive
    for member in members[1:]:
        for key in (
            "subject_ids", "subject_diagnoses", "subject_visit_offsets", "visit_scan_ids",
            "visit_subject_ids", "visit_diagnoses", "visit_label_ad", "visit_orders",
        ):
            if not np.array_equal(first[key], member.archive[key]):
                raise ValueError(f"Visit alignment failed at {key}: {member.name}")
        for key in ("visit_age_years", "visit_age_norm_train"):
            if not np.allclose(first[key], member.archive[key], atol=1.0e-7, rtol=0.0):
                raise ValueError(f"Visit age alignment failed at {key}: {member.name}")


def all_visit_model_velocity(members, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_visit_alignment(members)
    archive = members[0].archive
    scale, fit_residual = age_norm_per_year(members[0].train_archive)
    member_vertices: list[torch.Tensor] = []
    member_velocity: list[torch.Tensor] = []
    count = len(archive["visit_scan_ids"])
    for member in members:
        vertices_chunks: list[torch.Tensor] = []
        velocity_chunks: list[torch.Tensor] = []
        for start in range(0, count, batch_size):
            z = member.values["z"][start : start + batch_size]
            age = member.values["age"][start : start + batch_size]
            label = member.values["label"][start : start + batch_size]
            with torch.no_grad():
                velocity_year = member.flow.average_velocity(z, age, age, label) * scale
            z_input = z.detach().requires_grad_(True)
            _, vertex_velocity = torch.autograd.functional.jvp(
                member.geometry.vertices,
                z_input,
                velocity_year.detach(),
                create_graph=False,
                strict=False,
            )
            with torch.no_grad():
                vertices_chunks.append(member.geometry.vertices(z).detach().cpu())
                velocity_chunks.append(vertex_velocity.detach().cpu())
        member_vertices.append(torch.cat(vertices_chunks))
        member_velocity.append(torch.cat(velocity_chunks))
    vertices = torch.stack(member_vertices).mean(0)
    velocity = torch.stack(member_velocity).mean(0)
    faces = members[0].geometry.faces.detach().cpu().numpy()

    volume_geometry = members[0].geometry.cpu()
    _, volume_velocity = torch.autograd.functional.jvp(
        volume_geometry.volume_from_vertices,
        vertices.detach().requires_grad_(True),
        velocity,
        create_graph=False,
        strict=False,
    )
    with torch.no_grad():
        volume = volume_geometry.volume_from_vertices(vertices)
        volume_rate = (100.0 * volume_velocity / volume).numpy()

    vertices_np = vertices.numpy().astype(np.float64)
    velocity_np = velocity.numpy().astype(np.float64)
    vector_rms = np.sqrt(np.mean(np.sum(velocity_np ** 2, axis=2), axis=1))
    coordinate_rms = vector_rms / np.sqrt(3.0)
    normal_mean = np.empty(count, dtype=np.float64)
    normal_rms = np.empty(count, dtype=np.float64)
    for index in range(count):
        normals, area = vertex_geometry(vertices_np[index], faces)
        weight = area / max(float(area.sum()), 1.0e-12)
        normal_velocity = np.sum(velocity_np[index] * normals, axis=1)
        normal_mean[index] = float(np.sum(weight * normal_velocity))
        normal_rms[index] = float(np.sqrt(np.sum(weight * normal_velocity ** 2)))

    frame = pd.DataFrame({
        "split": np.repeat(str(archive["visit_splits"][0]), count),
        "subject_id": archive["visit_subject_ids"].astype(str),
        "scan_id": archive["visit_scan_ids"].astype(str),
        "diagnosis": archive["visit_diagnoses"].astype(str),
        "age_years": archive["visit_age_years"].astype(np.float64),
        "visit_order": archive["visit_orders"].astype(np.int64),
        "model_vector_rms_mm_per_year": vector_rms,
        "model_coordinate_rms_mm_per_year": coordinate_rms,
        "model_normal_mean_mm_per_year": normal_mean,
        "model_normal_rms_mm_per_year": normal_rms,
        "model_signed_volume_rate_pct_per_year": volume_rate,
    })
    metadata = {
        "visits_evaluated": count,
        "subjects_evaluated": int(frame.subject_id.nunique()),
        "age_norm_per_year": scale,
        "age_normalization_max_linear_residual": fit_residual,
        "members": [member.name for member in members],
        "aggregation": "equal mean of decoder-JVP mesh velocity vectors",
    }
    return frame, metadata


def merge_reference(model: pd.DataFrame, path: Path, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = pd.read_csv(path, dtype={"scan_id": str, "subject_id": str})
    reference = reference[reference["split"] == split].copy()
    columns = [
        "subject_id", "scan_id", "diagnosis", "age_years", "visit_order",
        "selected_estimator", "selected_estimator_label", "reference_label",
        "fit_vector_rms_mm_per_year", "fit_normal_mean_mm_per_year",
        "fit_normal_rms_mm_per_year", "adjacent_vector_rms_mm_per_year",
        "adjacent_normal_rms_mm_per_year",
        "fit_surface_integral_log_volume_rate_percent_per_year",
    ]
    merged = model.merge(
        reference[columns],
        on=["subject_id", "scan_id", "diagnosis", "visit_order"],
        how="inner",
        suffixes=("_model", "_reference"),
        validate="one_to_one",
    )
    if not np.allclose(merged.age_years_model, merged.age_years_reference, atol=1.0e-3):
        raise ValueError("Model/reference ages do not align")
    merged["age_years"] = merged.pop("age_years_model")
    merged = merged.drop(columns=["age_years_reference"])
    merged["model_to_fitted_vector_speed_ratio"] = (
        merged.model_vector_rms_mm_per_year / merged.fit_vector_rms_mm_per_year.clip(lower=1.0e-12)
    )
    merged["model_to_fitted_normal_speed_ratio"] = (
        merged.model_normal_rms_mm_per_year / merged.fit_normal_rms_mm_per_year.clip(lower=1.0e-12)
    )
    metadata = {
        "model_visits": len(model),
        "reference_visits": len(reference),
        "matched_visits": len(merged),
        "matched_subjects": int(merged.subject_id.nunique()),
        "excluded_model_visits_without_fitted_reference": int(len(model) - len(merged)),
        "selected_estimators": sorted(merged.selected_estimator.unique().tolist()),
    }
    return merged, metadata


def bootstrap_summary(frame: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    metrics = {
        "vector_rms": {
            "model": "model_vector_rms_mm_per_year",
            "fitted": "fit_vector_rms_mm_per_year",
            "adjacent": "adjacent_vector_rms_mm_per_year",
        },
        "normal_rms": {
            "model": "model_normal_rms_mm_per_year",
            "fitted": "fit_normal_rms_mm_per_year",
            "adjacent": "adjacent_normal_rms_mm_per_year",
        },
    }
    work = frame.copy()
    work["age_bin_index"] = np.digitize(work.age_years, AGE_EDGES[1:-1], right=False)
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for diagnosis in ("CN", "AD"):
        for bin_index, (label, center) in enumerate(zip(AGE_LABELS, AGE_CENTERS)):
            selected = work[(work.diagnosis == diagnosis) & (work.age_bin_index == bin_index)]
            for metric, series_columns in metrics.items():
                for series, column in series_columns.items():
                    subject_values = selected.groupby("subject_id")[column].mean().dropna().to_numpy()
                    if len(subject_values) == 0:
                        continue
                    draws = rng.choice(
                        subject_values, size=(samples, len(subject_values)), replace=True
                    ).mean(axis=1)
                    output.append({
                        "diagnosis": diagnosis,
                        "age_bin": label,
                        "age_center_years": float(center),
                        "metric": metric,
                        "series": series,
                        "series_label": SERIES[series],
                        "subjects": int(len(subject_values)),
                        "visits": int(len(selected)),
                        "subject_mean": float(subject_values.mean()),
                        "subject_median": float(np.median(subject_values)),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                    })
    return pd.DataFrame(output)


def overall_summary(frame: pd.DataFrame, samples: int, seed: int) -> list[dict[str, Any]]:
    columns = {
        "model_vector_rms_mm_per_year": "model_vector_rms_mm_per_year",
        "fitted_vector_rms_mm_per_year": "fit_vector_rms_mm_per_year",
        "adjacent_vector_rms_mm_per_year": "adjacent_vector_rms_mm_per_year",
        "model_normal_rms_mm_per_year": "model_normal_rms_mm_per_year",
        "fitted_normal_rms_mm_per_year": "fit_normal_rms_mm_per_year",
        "adjacent_normal_rms_mm_per_year": "adjacent_normal_rms_mm_per_year",
        "model_signed_volume_rate_pct_per_year": "model_signed_volume_rate_pct_per_year",
        "fitted_signed_volume_rate_pct_per_year": "fit_surface_integral_log_volume_rate_percent_per_year",
    }
    rng = np.random.default_rng(seed)
    rows = []
    for diagnosis in ("CN", "AD", "overall"):
        selected = frame if diagnosis == "overall" else frame[frame.diagnosis == diagnosis]
        for metric, column in columns.items():
            subject_values = selected.groupby("subject_id")[column].mean().dropna().to_numpy()
            draws = rng.choice(subject_values, size=(samples, len(subject_values)), replace=True).mean(axis=1)
            rows.append({
                "diagnosis": diagnosis,
                "metric": metric,
                "subjects": len(subject_values),
                "visits": len(selected),
                "subject_mean": float(subject_values.mean()),
                "subject_median": float(np.median(subject_values)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
            })
    return rows


def make_plot(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"model": "#D62728", "fitted": "#111111", "adjacent": "#7F7F7F"}
    styles = {"model": "-", "fitted": "-", "adjacent": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    for column, diagnosis in enumerate(("CN", "AD")):
        for row_index, metric in enumerate(("vector_rms", "normal_rms")):
            ax = axes[row_index, column]
            current = summary[(summary.diagnosis == diagnosis) & (summary.metric == metric)]
            for series in ("fitted", "model", "adjacent"):
                line = current[current.series == series].sort_values("age_center_years")
                x = line.age_center_years.to_numpy()
                y = line.subject_mean.to_numpy()
                low = line.ci95_low.to_numpy()
                high = line.ci95_high.to_numpy()
                ax.plot(
                    x, y, marker="o", linewidth=2.2, linestyle=styles[series],
                    color=colors[series], label=SERIES[series],
                )
                ax.fill_between(x, low, high, color=colors[series], alpha=0.10)
            ax.set_title(f"{diagnosis}: {'full-vector' if metric == 'vector_rms' else 'surface-normal'} RMS speed")
            ax.set_ylabel("mm/year")
            ax.grid(alpha=0.25)
            ax.set_xticks(AGE_CENTERS, AGE_LABELS)
            if row_index == 1:
                ax.set_xlabel("Age bin (years)")
            if row_index == 0 and column == 0:
                ax.legend(frameon=False, loc="upper left")
    fig.suptitle("LAMM N3 instantaneous velocity versus longitudinal observed references", fontsize=15)
    fig.text(
        0.5, 0.005,
        "Lines are equal-subject means; bands are 95% subject-bootstrap intervals. "
        "Fitted trajectory is the primary GT estimate; adjacent interval is observed but not instantaneous.",
        ha="center", fontsize=9,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("Test evaluation requires --allow-test")
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    device = C.choose_device(args.device)
    members = [
        load_member(label, path, args.split, device)
        for label, path in (parse_assignment(value, "--member") for value in args.member)
    ]
    model, model_metadata = all_visit_model_velocity(members, args.batch_size)
    merged, merge_metadata = merge_reference(model, args.reference_csv, args.split)
    by_age = bootstrap_summary(merged, args.bootstrap_samples, args.seed)
    overall = overall_summary(merged, args.bootstrap_samples, args.seed + 1)
    report = {
        "schema_version": 1,
        "split": args.split,
        "instantaneous_velocity_definition": "decoder JVP of zero-horizon direct-C4 phi(z,t,t,d), converted from normalized age to years",
        "primary_reference": "Observed - fitted trajectory derivative; validation-selected smoother; estimate, not directly measured instantaneous GT",
        "secondary_reference": "Observed - adjacent interval finite difference; interval-average and noise-sensitive",
        "model": model_metadata,
        "matching": merge_metadata,
        "overall_subject_weighted": overall,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "subject within diagnosis/age bin",
        "source_meshes_modified": False,
    }
    compact = {
        "matching": merge_metadata,
        "overall": [row for row in overall if row["metric"] in {
            "model_vector_rms_mm_per_year", "fitted_vector_rms_mm_per_year",
            "adjacent_vector_rms_mm_per_year", "model_normal_rms_mm_per_year",
            "fitted_normal_rms_mm_per_year", "adjacent_normal_rms_mm_per_year",
        }],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    if args.dry_run:
        print("DRY RUN PASSED - no files written")
        return 0
    output = C.require_bulk_path(args.output_dir, "velocity-by-age output")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    C.atomic_json(output / "report.json", report)
    model.to_csv(output / "model_velocity_all_visits.csv", index=False)
    merged.to_csv(output / "model_vs_reference_per_scan.csv", index=False)
    by_age.to_csv(output / "velocity_by_age_summary.csv", index=False)
    write_csv(output / "overall_subject_weighted.csv", overall)
    make_plot(by_age, output / "instantaneous_vs_gt_by_age.png")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
