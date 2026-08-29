#!/usr/bin/env python3
"""Age-stratified model instantaneous velocity versus observed longitudinal velocity."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-task5-direct-age-velocity")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/task5-direct-age-xdg-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C
import data as D
from conditional_spiral_unet import ConditionalSpiralUNet, vertex_normals
from train import build_model, validate_config


REFERENCE_DESCRIPTION = (
    "Observed velocity is the derivative of a smooth trajectory fitted to each subject's "
    "registered longitudinal visits after rigid removal. It is not a directly measured "
    "continuous-time ground-truth derivative."
)

BOOTSTRAP_METRICS = (
    "predicted_speed_mm_per_year",
    "observed_speed_mm_per_year",
    "speed_difference_mm_per_year",
    "speed_ratio",
    "predicted_inward_normal_mm_per_year",
    "observed_inward_normal_mm_per_year",
    "inward_normal_difference_mm_per_year",
    "vector_rmse_mm_per_year",
    "vector_error_to_zero_ratio",
    "normal_rmse_mm_per_year",
    "normal_error_to_zero_ratio",
    "vector_cosine",
    "normal_pearson",
    "normal_sign_agreement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--age-bins",
        type=float,
        nargs="+",
        default=(70.0, 75.0, 80.0, 95.0),
        metavar="AGE",
        help="Ordered bin edges. The final edge is inclusive; default: 70 75 80 95.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=1701)
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Balanced inference limit for a wiring smoke test; omit for the scientific analysis.",
    )
    return parser.parse_args()


def validate_age_edges(values: Iterable[float]) -> np.ndarray:
    edges = np.asarray(list(values), dtype=np.float64)
    if len(edges) < 2 or not np.isfinite(edges).all():
        raise ValueError("At least two finite age-bin edges are required")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("Age-bin edges must be strictly increasing")
    return edges


def age_bin_index(age: float, edges: np.ndarray) -> int | None:
    if age < float(edges[0]) or age > float(edges[-1]):
        return None
    if age == float(edges[-1]):
        return len(edges) - 2
    index = int(np.searchsorted(edges, age, side="right") - 1)
    return index if 0 <= index < len(edges) - 1 else None


def age_bin_label(index: int, edges: np.ndarray) -> str:
    left = float(edges[index])
    right = float(edges[index + 1])
    suffix = "]" if index == len(edges) - 2 else ")"
    return f"[{left:g}, {right:g}{suffix}"


def balanced_velocity_indices(split: D.PreparedSplit, max_scans: int | None) -> list[int]:
    candidates = [
        index
        for index in range(len(split.scan_ids))
        if float(split.velocity_reference_weight[index].cpu()) > 0.0
    ]
    if max_scans is None or int(max_scans) <= 0 or int(max_scans) >= len(candidates):
        return candidates
    groups = {
        diagnosis: [index for index in candidates if str(split.diagnoses[index]) == diagnosis]
        for diagnosis in ("CN", "AD")
    }
    output: list[int] = []
    cursor = {"CN": 0, "AD": 0}
    while len(output) < int(max_scans):
        changed = False
        for diagnosis in ("CN", "AD"):
            values = groups[diagnosis]
            if cursor[diagnosis] < len(values):
                output.append(values[cursor[diagnosis]])
                cursor[diagnosis] += 1
                changed = True
                if len(output) == int(max_scans):
                    break
        if not changed:
            break
    return output


@torch.no_grad()
def velocity_records(
    model: ConditionalSpiralUNet,
    split: D.PreparedSplit,
    batch_size: int,
    max_scans: int | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    selected = balanced_velocity_indices(split, max_scans)
    records: list[dict[str, Any]] = []
    for chunk in C.chunked(selected, int(batch_size)):
        indices = torch.as_tensor(chunk, dtype=torch.long, device=split.vertices.device)
        vertices = split.vertices[indices]
        observed = split.velocity_reference[indices]
        predicted = model.instantaneous_velocity(vertices, split.ages[indices], split.labels[indices])
        normals = vertex_normals(vertices, model.faces)
        predicted_normal = torch.sum(predicted * normals, dim=-1)
        observed_normal = torch.sum(observed * normals, dim=-1)

        vector_error = torch.sqrt(torch.mean((predicted - observed).square(), dim=(1, 2)))
        vector_zero = torch.sqrt(torch.mean(observed.square(), dim=(1, 2)))
        normal_error = torch.sqrt(torch.mean((predicted_normal - observed_normal).square(), dim=1))
        normal_zero = torch.sqrt(torch.mean(observed_normal.square(), dim=1))
        predicted_flat = predicted.reshape(len(chunk), -1)
        observed_flat = observed.reshape(len(chunk), -1)
        vector_cosine = torch.sum(predicted_flat * observed_flat, dim=1) / (
            torch.linalg.vector_norm(predicted_flat, dim=1)
            * torch.linalg.vector_norm(observed_flat, dim=1)
        ).clamp_min(1.0e-8)
        predicted_centered = predicted_normal - predicted_normal.mean(dim=1, keepdim=True)
        observed_centered = observed_normal - observed_normal.mean(dim=1, keepdim=True)
        normal_pearson = torch.sum(predicted_centered * observed_centered, dim=1) / (
            torch.linalg.vector_norm(predicted_centered, dim=1)
            * torch.linalg.vector_norm(observed_centered, dim=1)
        ).clamp_min(1.0e-8)
        predicted_speed = torch.linalg.vector_norm(predicted, dim=-1).mean(dim=1)
        observed_speed = torch.linalg.vector_norm(observed, dim=-1).mean(dim=1)
        sign_agreement = (torch.sign(predicted_normal) == torch.sign(observed_normal)).float().mean(dim=1)

        for local_index, split_index in enumerate(chunk):
            records.append(
                {
                    "scan_id": str(split.scan_ids[split_index]),
                    "subject_id": str(split.subject_ids[split_index]),
                    "diagnosis": str(split.diagnoses[split_index]),
                    "age_years": float(split.ages[split_index].cpu()),
                    "reference_reliability": float(split.velocity_reference_weight[split_index].cpu()),
                    "predicted_speed_mm_per_year": float(predicted_speed[local_index].cpu()),
                    "observed_speed_mm_per_year": float(observed_speed[local_index].cpu()),
                    # Outward mesh normals make negative normal velocity inward shrinkage.
                    "predicted_inward_normal_mm_per_year": float(-predicted_normal[local_index].mean().cpu()),
                    "observed_inward_normal_mm_per_year": float(-observed_normal[local_index].mean().cpu()),
                    "vector_rmse_mm_per_year": float(vector_error[local_index].cpu()),
                    "zero_vector_rmse_mm_per_year": float(vector_zero[local_index].cpu()),
                    "normal_rmse_mm_per_year": float(normal_error[local_index].cpu()),
                    "zero_normal_rmse_mm_per_year": float(normal_zero[local_index].cpu()),
                    "vector_cosine": float(vector_cosine[local_index].cpu()),
                    "normal_pearson": float(normal_pearson[local_index].cpu()),
                    "normal_sign_agreement": float(sign_agreement[local_index].cpu()),
                }
            )
    return records


def weighted_value(records: list[dict[str, Any]], name: str) -> float:
    weights = np.asarray([float(row["reference_reliability"]) for row in records], dtype=np.float64)
    values = np.asarray([float(row[name]) for row in records], dtype=np.float64)
    if not float(weights.sum()) > 0.0:
        raise ValueError("Velocity group has zero total reference reliability")
    return float(np.average(values, weights=weights))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty age/diagnosis group")
    predicted_speed = weighted_value(records, "predicted_speed_mm_per_year")
    observed_speed = weighted_value(records, "observed_speed_mm_per_year")
    predicted_inward = weighted_value(records, "predicted_inward_normal_mm_per_year")
    observed_inward = weighted_value(records, "observed_inward_normal_mm_per_year")
    vector_error = weighted_value(records, "vector_rmse_mm_per_year")
    vector_zero = weighted_value(records, "zero_vector_rmse_mm_per_year")
    normal_error = weighted_value(records, "normal_rmse_mm_per_year")
    normal_zero = weighted_value(records, "zero_normal_rmse_mm_per_year")
    return {
        "visits": len(records),
        "subjects": len({str(row["subject_id"]) for row in records}),
        "reference_weight_sum": float(sum(float(row["reference_reliability"]) for row in records)),
        "reference_reliability_mean": float(
            np.mean([float(row["reference_reliability"]) for row in records])
        ),
        "predicted_speed_mm_per_year": predicted_speed,
        "observed_speed_mm_per_year": observed_speed,
        "speed_difference_mm_per_year": float(predicted_speed - observed_speed),
        "speed_ratio": float(predicted_speed / max(observed_speed, 1.0e-8)),
        "predicted_inward_normal_mm_per_year": predicted_inward,
        "observed_inward_normal_mm_per_year": observed_inward,
        "inward_normal_difference_mm_per_year": float(predicted_inward - observed_inward),
        "vector_rmse_mm_per_year": vector_error,
        "zero_vector_rmse_mm_per_year": vector_zero,
        "vector_error_to_zero_ratio": float(vector_error / max(vector_zero, 1.0e-8)),
        "normal_rmse_mm_per_year": normal_error,
        "zero_normal_rmse_mm_per_year": normal_zero,
        "normal_error_to_zero_ratio": float(normal_error / max(normal_zero, 1.0e-8)),
        "vector_cosine": weighted_value(records, "vector_cosine"),
        "normal_pearson": weighted_value(records, "normal_pearson"),
        "normal_sign_agreement": weighted_value(records, "normal_sign_agreement"),
    }


def subject_bootstrap_intervals(
    records: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, float]:
    subjects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        subjects[str(row["subject_id"])].append(row)
    subject_ids = sorted(subjects)
    output: dict[str, float] = {}
    if int(samples) <= 0 or len(subject_ids) < 2:
        for metric in BOOTSTRAP_METRICS:
            output[f"{metric}_ci95_low"] = float("nan")
            output[f"{metric}_ci95_high"] = float("nan")
        return output
    rng = np.random.default_rng(int(seed))
    distributions = {metric: np.empty(int(samples), dtype=np.float64) for metric in BOOTSTRAP_METRICS}
    for sample_index in range(int(samples)):
        sampled = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        current = [row for subject in sampled for row in subjects[str(subject)]]
        summary = summarize_records(current)
        for metric in BOOTSTRAP_METRICS:
            distributions[metric][sample_index] = float(summary[metric])
    for metric, values in distributions.items():
        low, high = np.quantile(values, (0.025, 0.975))
        output[f"{metric}_ci95_low"] = float(low)
        output[f"{metric}_ci95_high"] = float(high)
    return output


def age_summary(
    records: list[dict[str, Any]], edges: np.ndarray, bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for diagnosis_index, diagnosis in enumerate(("CN", "AD")):
        for index in range(len(edges) - 1):
            current = [
                row
                for row in records
                if str(row["diagnosis"]) == diagnosis
                and age_bin_index(float(row["age_years"]), edges) == index
            ]
            if not current:
                continue
            summary = {
                "diagnosis": diagnosis,
                "age_bin_index": index,
                "age_bin": age_bin_label(index, edges),
                "age_start_years": float(edges[index]),
                "age_end_years": float(edges[index + 1]),
                "age_midpoint_years": float(0.5 * (edges[index] + edges[index + 1])),
                **summarize_records(current),
            }
            summary.update(
                subject_bootstrap_intervals(
                    current,
                    int(bootstrap_samples),
                    int(seed) + 1009 * diagnosis_index + 97 * index,
                )
            )
            output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _errorbar(axis, x, rows, metric: str, label: str, color: str, marker: str) -> None:
    values = np.asarray([float(row[metric]) for row in rows])
    low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in rows])
    high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in rows])
    errors = np.vstack((np.maximum(values - low, 0.0), np.maximum(high - values, 0.0)))
    if np.isfinite(errors).all():
        axis.errorbar(x, values, yerr=errors, label=label, color=color, marker=marker, capsize=3)
    else:
        axis.plot(x, values, label=label, color=color, marker=marker)


def _rows_for(summary: list[dict[str, Any]], diagnosis: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in summary if str(row["diagnosis"]) == diagnosis],
        key=lambda row: int(row["age_bin_index"]),
    )


def plot_speed(summary: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, diagnosis in zip(axes, ("CN", "AD")):
        rows = _rows_for(summary, diagnosis)
        x = np.arange(len(rows))
        _errorbar(axis, x, rows, "observed_speed_mm_per_year", "Observed velocity", "#202020", "o")
        _errorbar(axis, x, rows, "predicted_speed_mm_per_year", "Model instantaneous velocity", "#e76f00", "s")
        axis.set_xticks(x, [str(row["age_bin"]) for row in rows])
        axis.set_xlabel("Age interval (years)")
        axis.set_title(f"{diagnosis}: mean surface speed")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean vertex speed (mm/year)")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Instantaneous mesh velocity across age\n"
        "Observed = fitted longitudinal-visit derivative; error bars = subject-bootstrap 95% CI",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "speed_by_age.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "speed_by_age.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_inward_normal(summary: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, diagnosis in zip(axes, ("CN", "AD")):
        rows = _rows_for(summary, diagnosis)
        x = np.arange(len(rows))
        _errorbar(
            axis,
            x,
            rows,
            "observed_inward_normal_mm_per_year",
            "Observed velocity",
            "#202020",
            "o",
        )
        _errorbar(
            axis,
            x,
            rows,
            "predicted_inward_normal_mm_per_year",
            "Model instantaneous velocity",
            "#5b4cc4",
            "s",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, [str(row["age_bin"]) for row in rows])
        axis.set_xlabel("Age interval (years)")
        axis.set_title(f"{diagnosis}: mean inward-normal velocity")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Inward-normal velocity (mm/year; positive = shrinkage)")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Direction of hippocampal surface change across age\n"
        "Observed = fitted longitudinal-visit derivative; error bars = subject-bootstrap 95% CI",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "inward_normal_velocity_by_age.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "inward_normal_velocity_by_age.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_agreement(summary: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = (
        ("vector_error_to_zero_ratio", "Vector error / zero-velocity error", 1.0),
        ("vector_cosine", "Vector direction cosine", 0.0),
        ("normal_pearson", "Normal-velocity spatial correlation", 0.0),
    )
    colors = {"CN": "#2b6cb0", "AD": "#c53030"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, (metric, label, reference) in zip(axes, metrics):
        for diagnosis, marker in (("CN", "o"), ("AD", "s")):
            rows = _rows_for(summary, diagnosis)
            x = np.arange(len(rows))
            _errorbar(axis, x, rows, metric, diagnosis, colors[diagnosis], marker)
        labels = [str(row["age_bin"]) for row in _rows_for(summary, "CN")]
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.axhline(reference, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Age interval (years)")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Below 1 beats no change")
    axes[1].set_title("1 means identical vector direction")
    axes[2].set_title("1 means identical spatial pattern")
    axes[-1].legend(frameon=False)
    fig.suptitle(
        "Instantaneous-velocity agreement by diagnosis and age\n"
        "Intervals show subject-bootstrap 95% CI; the observed reference is fitted from longitudinal visits",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_agreement_by_age.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "velocity_agreement_by_age.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    edges = validate_age_edges(args.age_bins)
    root = C.output_root(args.output_root)
    device = C.choose_device(args.device)
    checkpoint_path = C.resolve_path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    validate_config(config)
    if bool(payload.get("test_data_loaded", False)):
        raise ValueError("Refusing a checkpoint whose training payload reports test-data access")
    split = D.load_split(args.split, root, device)
    model, _ = build_model(config, root, device)
    model.load_state_dict(payload["model_state_dict"])
    batch_size = int(args.batch_size or config["training"]["evaluation_batch_size"])
    records = velocity_records(model, split, batch_size, args.max_scans)
    summary = age_summary(records, edges, int(args.bootstrap_samples), int(args.bootstrap_seed))
    if not summary:
        raise ValueError("No positive-reliability visits fall inside the requested age range")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent.parent / "evaluation" / args.split / "age_velocity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_visit_velocity.csv", records)
    write_csv(output_dir / "age_velocity_summary.csv", summary)
    plot_speed(summary, output_dir)
    plot_inward_normal(summary, output_dir)
    plot_agreement(summary, output_dir)

    included = sum(
        age_bin_index(float(row["age_years"]), edges) is not None for row in records
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "operator": str(config["model"]["operator"]),
        "split": str(args.split),
        "age_bin_edges_years": edges.tolist(),
        "reference_description": REFERENCE_DESCRIPTION,
        "weighting": "Reference-reliability weighted within each diagnosis/age interval",
        "uncertainty": "Paired visit records resampled by subject with replacement",
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "positive_reliability_visits": len(records),
        "included_visits": int(included),
        "excluded_outside_age_range": int(len(records) - included),
        "max_scans_smoke_limit": args.max_scans,
        "test_data_loaded_by_training": False,
        "age_groups": summary,
    }
    C.atomic_json(output_dir / "summary.json", report)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "operator": report["operator"],
                "split": report["split"],
                "included_visits": report["included_visits"],
                "age_groups": len(summary),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
