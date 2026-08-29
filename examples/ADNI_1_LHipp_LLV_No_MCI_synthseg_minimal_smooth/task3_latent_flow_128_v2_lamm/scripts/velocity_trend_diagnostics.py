#!/usr/bin/env python3
"""Reproducible age/disease diagnostics for the LAMM N3 C4 velocity field.

This analysis deliberately separates four estimands:

1. the model-defined zero-horizon velocity, evaluated with a decoder JVP;
2. a one-year finite-horizon velocity, which is inside the C4 training support;
3. a validation-selected population longitudinal surface derivative; and
4. the noise-sensitive adjacent-scan finite difference.

The population reference uses subject fixed effects so its age curve is driven by
within-subject change rather than cross-sectional differences between subjects.
Polynomial degree is selected independently for CN and AD on the validation split
and then frozen before the test reference is fitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from _bootstrap import activate
from evaluate_n3_ensemble import age_norm_per_year, load_member, parse_assignment
from velocity_by_age import assert_visit_alignment, vertex_geometry

activate()
import common as C


AGE_CENTER = 75.0
AGE_SCALE = 10.0
DIAGNOSES = ("CN", "AD")
SERIES_LABELS = {
    "population_gt": "Observed population trajectory",
    "individual_fitted": "Observed individual fitted trajectory",
    "instantaneous": "LAMM N3 instantaneous",
    "one_year": "LAMM N3 one-year",
    "adjacent": "Observed adjacent interval",
}


@dataclass
class FixedEffectFit:
    degree: int
    beta: np.ndarray
    alpha: np.ndarray


@dataclass
class BootstrapCurve:
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    draws: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--horizon-years", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--age-min", type=float, default=65.0)
    parser.add_argument("--age-max", type=float, default=90.0)
    parser.add_argument("--age-step", type=float, default=0.5)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--reference-array-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    signed = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum() / 6.0
    return abs(float(signed))


def velocity_scalars(
    vertices: np.ndarray,
    velocity: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float]:
    normals, area = vertex_geometry(vertices, faces)
    weight = area / max(float(area.sum()), 1.0e-12)
    normal_velocity = np.sum(velocity * normals, axis=1)
    volume = mesh_volume(vertices, faces)
    return {
        "vector_rms_mm_per_year": float(np.sqrt(np.mean(np.sum(velocity**2, axis=1)))),
        "normal_rms_mm_per_year": float(np.sqrt(np.sum(weight * normal_velocity**2))),
        "normal_mean_mm_per_year": float(np.sum(weight * normal_velocity)),
        "inward_normal_mean_mm_per_year": float(-np.sum(weight * normal_velocity)),
        "signed_volume_rate_pct_per_year": float(
            100.0 * np.sum(area * normal_velocity) / max(volume, 1.0e-12)
        ),
    }


def compute_model_velocity(
    members: list[Any],
    batch_size: int,
    horizon_years: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_visit_alignment(members)
    archive = members[0].archive
    count = len(archive["visit_scan_ids"])
    scale, residual = age_norm_per_year(members[0].train_archive)
    member_vertices: list[torch.Tensor] = []
    member_instantaneous: list[torch.Tensor] = []
    member_horizon_velocity: list[torch.Tensor] = []
    member_horizon_vertices: list[torch.Tensor] = []
    for member in members:
        vertices_chunks: list[torch.Tensor] = []
        instantaneous_chunks: list[torch.Tensor] = []
        horizon_velocity_chunks: list[torch.Tensor] = []
        horizon_vertices_chunks: list[torch.Tensor] = []
        for start in range(0, count, batch_size):
            z = member.values["z"][start : start + batch_size]
            age = member.values["age"][start : start + batch_size]
            label = member.values["label"][start : start + batch_size]
            with torch.no_grad():
                instantaneous_latent = member.flow.average_velocity(z, age, age, label) * scale
                transported = member.flow.transport(
                    z, age, age + float(horizon_years) * scale, label
                )
                vertices = member.geometry.vertices(z)
                horizon_vertices = member.geometry.vertices(transported)
            z_input = z.detach().requires_grad_(True)
            _, instantaneous_vertices = torch.autograd.functional.jvp(
                member.geometry.vertices,
                z_input,
                instantaneous_latent.detach(),
                create_graph=False,
                strict=False,
            )
            vertices_chunks.append(vertices.detach().cpu())
            instantaneous_chunks.append(instantaneous_vertices.detach().cpu())
            horizon_vertices_chunks.append(horizon_vertices.detach().cpu())
            horizon_velocity_chunks.append(
                ((horizon_vertices - vertices) / float(horizon_years)).detach().cpu()
            )
        member_vertices.append(torch.cat(vertices_chunks))
        member_instantaneous.append(torch.cat(instantaneous_chunks))
        member_horizon_velocity.append(torch.cat(horizon_velocity_chunks))
        member_horizon_vertices.append(torch.cat(horizon_vertices_chunks))

    vertices = torch.stack(member_vertices).mean(0).numpy().astype(np.float64)
    instantaneous = torch.stack(member_instantaneous).mean(0).numpy().astype(np.float64)
    horizon_velocity = torch.stack(member_horizon_velocity).mean(0).numpy().astype(np.float64)
    horizon_vertices = torch.stack(member_horizon_vertices).mean(0).numpy().astype(np.float64)
    faces = members[0].geometry.faces.detach().cpu().numpy().astype(np.int64)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        instant = velocity_scalars(vertices[index], instantaneous[index], faces)
        finite = velocity_scalars(vertices[index], horizon_velocity[index], faces)
        finite["signed_volume_rate_pct_per_year"] = float(
            100.0
            * (
                np.log(mesh_volume(horizon_vertices[index], faces))
                - np.log(mesh_volume(vertices[index], faces))
            )
            / float(horizon_years)
        )
        row: dict[str, Any] = {
            "split": str(archive["visit_splits"][index]),
            "subject_id": str(archive["visit_subject_ids"][index]),
            "scan_id": str(archive["visit_scan_ids"][index]),
            "diagnosis": str(archive["visit_diagnoses"][index]),
            "age_years": float(archive["visit_age_years"][index]),
            "visit_order": int(archive["visit_orders"][index]),
        }
        row.update({f"instantaneous_{key}": value for key, value in instant.items()})
        row.update({f"one_year_{key}": value for key, value in finite.items()})
        rows.append(row)
    metadata = {
        "members": [member.name for member in members],
        "visits": count,
        "subjects": int(len(np.unique(archive["visit_subject_ids"]))),
        "age_norm_per_year": float(scale),
        "age_normalization_max_residual": float(residual),
        "finite_horizon_years": float(horizon_years),
        "ensemble_aggregation": "equal mean of member velocity vectors in mesh space",
    }
    return pd.DataFrame(rows), metadata


def polynomial(age: np.ndarray, degree: int) -> np.ndarray:
    x = (np.asarray(age, dtype=np.float64) - AGE_CENTER) / AGE_SCALE
    if degree == 0:
        return np.empty((len(x), 0), dtype=np.float64)
    return np.stack([x**power for power in range(1, degree + 1)], axis=1)


def polynomial_derivative(age: np.ndarray, degree: int) -> np.ndarray:
    x = (np.asarray(age, dtype=np.float64) - AGE_CENTER) / AGE_SCALE
    return np.stack(
        [power * x ** (power - 1) / AGE_SCALE for power in range(1, degree + 1)],
        axis=1,
    )


def subject_demean(
    predictors: np.ndarray,
    values: np.ndarray,
    subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for subject in np.unique(subjects):
        selected = subjects == subject
        p_chunks.append(predictors[selected] - predictors[selected].mean(axis=0))
        y_chunks.append(values[selected] - values[selected].mean(axis=0))
    return np.concatenate(p_chunks), np.concatenate(y_chunks)


def fixed_effect_fit(
    age: np.ndarray,
    subjects: np.ndarray,
    values: np.ndarray,
    degree: int,
) -> FixedEffectFit:
    values = np.asarray(values, dtype=np.float64)
    original_shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    predictors = polynomial(age, degree)
    centered_x, centered_y = subject_demean(predictors, flat, subjects)
    beta = np.linalg.pinv(centered_x, rcond=1.0e-12) @ centered_y
    subject_alpha = []
    for subject in np.unique(subjects):
        selected = subjects == subject
        subject_alpha.append(
            (flat[selected] - predictors[selected] @ beta).mean(axis=0)
        )
    alpha = np.mean(subject_alpha, axis=0)
    return FixedEffectFit(
        degree=degree,
        beta=beta.reshape((degree,) + original_shape),
        alpha=alpha.reshape(original_shape),
    )


def fixed_effect_level_curve(fit: FixedEffectFit, ages: np.ndarray) -> np.ndarray:
    return fit.alpha + np.einsum("gd,d...->g...", polynomial(ages, fit.degree), fit.beta)


def fixed_effect_derivative_curve(fit: FixedEffectFit, ages: np.ndarray) -> np.ndarray:
    return np.einsum(
        "gd,d...->g...", polynomial_derivative(ages, fit.degree), fit.beta
    )


def loo_score(
    age: np.ndarray,
    subjects: np.ndarray,
    values: np.ndarray,
    degree: int,
) -> tuple[float, float]:
    unique = np.unique(subjects)
    lookup = {subject: index for index, subject in enumerate(unique)}
    indicators = np.zeros((len(age), len(unique)), dtype=np.float64)
    indicators[np.arange(len(age)), [lookup[subject] for subject in subjects]] = 1.0
    design = np.concatenate((indicators, polynomial(age, degree)), axis=1)
    flat = np.asarray(values, dtype=np.float64).reshape(len(age), -1)
    inverse = np.linalg.pinv(design, rcond=1.0e-12)
    residual = flat - design @ (inverse @ flat)
    leverage = np.diag(design @ inverse)
    loo = residual / np.maximum(1.0 - leverage[:, None], 1.0e-8)
    scan_rmse = np.sqrt(np.mean(loo**2, axis=1))
    table = pd.DataFrame({"subject_id": subjects.astype(str), "rmse": scan_rmse})
    subject_mean = float(table.groupby("subject_id").rmse.mean().mean())
    return subject_mean, float(scan_rmse.mean())


def select_reference_degrees(
    validation_array: dict[str, np.ndarray],
    validation_table: pd.DataFrame,
) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, int]] = {"surface": {}, "log_volume": {}}
    for diagnosis in DIAGNOSES:
        array_mask = validation_array["diagnoses"].astype(str) == diagnosis
        surface_values = validation_array["aligned_observed"][array_mask]
        surface_age = validation_array["ages_years"][array_mask].astype(np.float64)
        surface_subjects = validation_array["subject_ids"][array_mask].astype(str)
        volume_frame = validation_table[validation_table.diagnosis == diagnosis]
        volume_age = volume_frame.age_years.to_numpy(dtype=np.float64)
        volume_subjects = volume_frame.subject_id.to_numpy(dtype=str)
        volume_values = np.log(volume_frame.observed_volume_mm3.to_numpy(dtype=np.float64))[:, None]
        for target, age, subjects, values in (
            ("surface", surface_age, surface_subjects, surface_values),
            ("log_volume", volume_age, volume_subjects, volume_values),
        ):
            current = []
            for degree in (1, 2, 3):
                subject_score, visit_score = loo_score(age, subjects, values, degree)
                scale = 100.0 if target == "log_volume" else 1.0
                row = {
                    "diagnosis": diagnosis,
                    "target": target,
                    "degree": degree,
                    "subject_balanced_loo_rmse": subject_score * scale,
                    "visit_balanced_loo_rmse": visit_score * scale,
                    "units": "log-volume percent" if target == "log_volume" else "coordinate mm",
                }
                rows.append(row)
                current.append(row)
            winner = min(current, key=lambda item: item["subject_balanced_loo_rmse"])
            selected[target][diagnosis] = int(winner["degree"])
    return selected, pd.DataFrame(rows)


def subject_aggregates(
    age: np.ndarray,
    subjects: np.ndarray,
    values: np.ndarray,
    degree: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    original_shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    predictors = polynomial(age, degree)
    a_values = []
    b_values = []
    mean_p = []
    mean_y = []
    for subject in np.unique(subjects):
        selected = subjects == subject
        p = predictors[selected]
        y = flat[selected]
        pd = p - p.mean(axis=0)
        yd = y - y.mean(axis=0)
        a_values.append(pd.T @ pd)
        b_values.append(pd.T @ yd)
        mean_p.append(p.mean(axis=0))
        mean_y.append(y.mean(axis=0))
    return {
        "a": np.stack(a_values),
        "b": np.stack(b_values),
        "mean_p": np.stack(mean_p),
        "mean_y": np.stack(mean_y),
        "value_shape": np.asarray(original_shape, dtype=np.int64),
    }


def bootstrap_fixed_effect_curve(
    age: np.ndarray,
    subjects: np.ndarray,
    values: np.ndarray,
    degree: int,
    grid: np.ndarray,
    samples: int,
    seed: int,
    transform: Callable[[FixedEffectFit, np.ndarray], np.ndarray],
) -> BootstrapCurve:
    fit = fixed_effect_fit(age, subjects, values, degree)
    mean = np.asarray(transform(fit, grid), dtype=np.float64)
    aggregates = subject_aggregates(age, subjects, values, degree)
    count = len(aggregates["a"])
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(grid)), dtype=np.float64)
    for draw_index in range(samples):
        sampled = rng.integers(0, count, count)
        weights = np.bincount(sampled, minlength=count).astype(np.float64)
        matrix = np.einsum("s,sde->de", weights, aggregates["a"])
        right = np.einsum("s,sdk->dk", weights, aggregates["b"])
        beta_flat = np.linalg.pinv(matrix, rcond=1.0e-12) @ right
        mean_p = np.einsum("s,sd->d", weights, aggregates["mean_p"]) / count
        mean_y = np.einsum("s,sk->k", weights, aggregates["mean_y"]) / count
        alpha_flat = mean_y - mean_p @ beta_flat
        value_shape = tuple(int(value) for value in aggregates["value_shape"])
        sampled_fit = FixedEffectFit(
            degree=degree,
            beta=beta_flat.reshape((degree,) + value_shape),
            alpha=alpha_flat.reshape(value_shape),
        )
        draws[draw_index] = np.asarray(transform(sampled_fit, grid), dtype=np.float64)
    return BootstrapCurve(
        mean=mean,
        low=np.quantile(draws, 0.025, axis=0),
        high=np.quantile(draws, 0.975, axis=0),
        draws=draws,
    )


def surface_speed_transform(fit: FixedEffectFit, grid: np.ndarray) -> np.ndarray:
    velocity = fixed_effect_derivative_curve(fit, grid)
    return np.sqrt(np.mean(np.sum(velocity**2, axis=2), axis=1))


def scalar_level_transform(fit: FixedEffectFit, grid: np.ndarray) -> np.ndarray:
    return fixed_effect_level_curve(fit, grid).reshape(-1)


def positive_scalar_level_transform(fit: FixedEffectFit, grid: np.ndarray) -> np.ndarray:
    return np.exp(fixed_effect_level_curve(fit, grid).reshape(-1))


def negative_scalar_derivative_transform(fit: FixedEffectFit, grid: np.ndarray) -> np.ndarray:
    return -100.0 * fixed_effect_derivative_curve(fit, grid).reshape(-1)


def add_curve_rows(
    rows: list[dict[str, Any]],
    diagnosis: str,
    metric: str,
    series: str,
    grid: np.ndarray,
    curve: BootstrapCurve,
) -> None:
    for index, age in enumerate(grid):
        rows.append({
            "diagnosis": diagnosis,
            "metric": metric,
            "series": series,
            "series_label": SERIES_LABELS[series],
            "age_years": float(age),
            "mean": float(curve.mean[index]),
            "ci95_low": float(curve.low[index]),
            "ci95_high": float(curve.high[index]),
        })


def curve_agreement(
    diagnosis: str,
    metric: str,
    model_name: str,
    grid: np.ndarray,
    model: BootstrapCurve,
    reference: BootstrapCurve,
) -> dict[str, Any]:
    model_delta = model.mean[-1] - model.mean[0]
    reference_delta = reference.mean[-1] - reference.mean[0]
    centered_model = model.mean - model.mean.mean()
    centered_reference = reference.mean - reference.mean.mean()
    denominator = np.linalg.norm(centered_model) * np.linalg.norm(centered_reference)
    correlation = (
        float(np.dot(centered_model, centered_reference) / denominator)
        if denominator > 1.0e-12
        else None
    )
    model_draw_delta = model.draws[:, -1] - model.draws[:, 0]
    reference_draw_delta = reference.draws[:, -1] - reference.draws[:, 0]
    reference_tolerance = 1.0e-8 * max(float(np.mean(np.abs(reference.mean))), 1.0)
    reference_trend_identifiable = bool(
        np.quantile(np.abs(reference_draw_delta), 0.95) > reference_tolerance
    )
    same_direction_probability = None
    if reference_trend_identifiable:
        same_direction_probability = float(
            np.mean(np.sign(model_draw_delta) == np.sign(reference_draw_delta))
        )
    return {
        "diagnosis": diagnosis,
        "metric": metric,
        "model_series": model_name,
        "age_min": float(grid[0]),
        "age_max": float(grid[-1]),
        "model_mean": float(model.mean.mean()),
        "reference_mean": float(reference.mean.mean()),
        "mean_magnitude_ratio": float(model.mean.mean() / max(reference.mean.mean(), 1.0e-12)),
        "model_endpoint_change": float(model_delta),
        "model_endpoint_change_ci95_low": float(np.quantile(model_draw_delta, 0.025)),
        "model_endpoint_change_ci95_high": float(np.quantile(model_draw_delta, 0.975)),
        "model_probability_endpoint_increase": float(np.mean(model_draw_delta > 0.0)),
        "reference_endpoint_change": float(reference_delta),
        "reference_endpoint_change_ci95_low": float(np.quantile(reference_draw_delta, 0.025)),
        "reference_endpoint_change_ci95_high": float(np.quantile(reference_draw_delta, 0.975)),
        "reference_probability_endpoint_increase": float(np.mean(reference_draw_delta > 0.0)),
        "reference_trend_identifiable": reference_trend_identifiable,
        "curve_centered_correlation": correlation,
        "curve_rmse": float(np.sqrt(np.mean((model.mean - reference.mean) ** 2))),
        "bootstrap_same_endpoint_direction_probability": same_direction_probability,
    }


def subject_slope_summary(
    frame: pd.DataFrame,
    columns: dict[str, str],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for diagnosis in DIAGNOSES:
        selected = frame[frame.diagnosis == diagnosis]
        for series, column in columns.items():
            blocks = []
            for _, group in selected.groupby("subject_id"):
                x = group.age_years.to_numpy(dtype=np.float64)
                y = group[column].to_numpy(dtype=np.float64)
                xc = x - x.mean()
                yc = y - y.mean()
                blocks.append((float(np.sum(xc * yc)), float(np.sum(xc**2))))
            numerator = np.asarray([item[0] for item in blocks])
            denominator = np.asarray([item[1] for item in blocks])
            slope = float(numerator.sum() / denominator.sum())
            indices = rng.integers(0, len(blocks), (samples, len(blocks)))
            draws = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
            rows.append({
                "diagnosis": diagnosis,
                "series": series,
                "series_label": SERIES_LABELS[series],
                "metric": column,
                "subjects": len(blocks),
                "slope_per_age_year": slope,
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "probability_positive": float(np.mean(draws > 0.0)),
            })
    return pd.DataFrame(rows)


def subject_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "instantaneous_vector_rms_mm_per_year",
        "one_year_vector_rms_mm_per_year",
        "fit_vector_rms_mm_per_year",
        "instantaneous_signed_volume_rate_pct_per_year",
        "one_year_signed_volume_rate_pct_per_year",
        "fit_surface_integral_log_volume_rate_percent_per_year",
    ]
    return frame.groupby(["subject_id", "diagnosis"], as_index=False)[columns].mean()


def plot_population_velocity(
    curves: pd.DataFrame,
    subject_counts: dict[str, int],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"population_gt": "#111111", "instantaneous": "#D62728", "one_year": "#FF8C00"}
    styles = {"population_gt": "-", "instantaneous": "-", "one_year": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey="row")
    for column, diagnosis in enumerate(DIAGNOSES):
        current = curves[(curves.diagnosis == diagnosis) & (curves.metric == "vector_rms")]
        baseline_values: dict[str, float] = {}
        for series in ("population_gt", "instantaneous", "one_year"):
            line = current[current.series == series].sort_values("age_years")
            x = line.age_years.to_numpy()
            y = line["mean"].to_numpy()
            low = line.ci95_low.to_numpy()
            high = line.ci95_high.to_numpy()
            axes[0, column].plot(x, y, color=colors[series], linestyle=styles[series], linewidth=2.3, label=SERIES_LABELS[series])
            axes[0, column].fill_between(x, low, high, color=colors[series], alpha=0.10)
            anchor = int(np.argmin(np.abs(x - AGE_CENTER)))
            baseline_values[series] = max(float(y[anchor]), 1.0e-12)
            axes[1, column].plot(x, y / baseline_values[series], color=colors[series], linestyle=styles[series], linewidth=2.3, label=SERIES_LABELS[series])
        axes[0, column].set_title(f"{diagnosis} (n={subject_counts[diagnosis]} subjects)")
        axes[1, column].axhline(1.0, color="#AAAAAA", linewidth=1.0)
        axes[1, column].set_xlabel("Age (years)")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Full-vector RMS speed (mm/year)")
    axes[1, 0].set_ylabel("Speed relative to age 75")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("LAMM N3 velocity versus validation-selected population longitudinal reference")
    fig.text(0.5, 0.012, "Top: absolute calibration on a shared CN/AD scale. Bottom: age-trend shape only. Bands use subject bootstrap.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_directed_atrophy(curves: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"population_gt": "#111111", "individual_fitted": "#111111", "instantaneous": "#D62728", "one_year": "#FF8C00"}
    styles = {"population_gt": "-", "individual_fitted": ":", "instantaneous": "-", "one_year": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey="row")
    for column, diagnosis in enumerate(DIAGNOSES):
        for row_index, metric in enumerate(("volume_atrophy", "inward_normal")):
            current = curves[(curves.diagnosis == diagnosis) & (curves.metric == metric)]
            reference_name = "population_gt" if metric == "volume_atrophy" else "individual_fitted"
            for series in (reference_name, "instantaneous", "one_year"):
                line = current[current.series == series].sort_values("age_years")
                x = line.age_years.to_numpy()
                y = line["mean"].to_numpy()
                axes[row_index, column].plot(x, y, color=colors[series], linestyle=styles[series], linewidth=2.3, label=SERIES_LABELS[series])
                axes[row_index, column].fill_between(x, line.ci95_low, line.ci95_high, color=colors[series], alpha=0.10)
            axes[row_index, column].grid(alpha=0.25)
            axes[row_index, column].set_title(diagnosis if row_index == 0 else "")
            if row_index == 1:
                axes[row_index, column].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Volume atrophy magnitude (%/year)")
    axes[1, 0].set_ylabel("Inward normal mean (mm/year)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[1, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Directed hippocampal atrophy by age")
    fig.text(0.5, 0.012, "Positive values denote inward/atrophic change. Population log-volume GT and individual fitted-normal GT are shown separately.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_and_adjacent(
    calibration: pd.DataFrame,
    curves: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diagnosis_colors = {"CN": "#1F77B4", "AD": "#D62728"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for diagnosis in DIAGNOSES:
        selected = calibration[calibration.diagnosis == diagnosis]
        axes[0].scatter(selected.fit_vector_rms_mm_per_year, selected.instantaneous_vector_rms_mm_per_year, s=38, alpha=0.75, color=diagnosis_colors[diagnosis], label=diagnosis)
        axes[1].scatter(-selected.fit_surface_integral_log_volume_rate_percent_per_year, -selected.instantaneous_signed_volume_rate_pct_per_year, s=38, alpha=0.75, color=diagnosis_colors[diagnosis], label=diagnosis)
        line = curves[(curves.diagnosis == diagnosis) & (curves.metric == "adjacent_vector") & (curves.series == "adjacent")].sort_values("age_years")
        axes[2].plot(line.age_years, line["mean"], linewidth=2.3, color=diagnosis_colors[diagnosis], label=diagnosis)
        axes[2].fill_between(line.age_years, line.ci95_low, line.ci95_high, color=diagnosis_colors[diagnosis], alpha=0.12)
    for axis, xname, yname in (
        (axes[0], "Individual fitted speed (mm/year)", "LAMM instantaneous speed (mm/year)"),
        (axes[1], "Individual fitted atrophy (%/year)", "LAMM instantaneous atrophy (%/year)"),
    ):
        low = min(axis.get_xlim()[0], axis.get_ylim()[0])
        high = max(axis.get_xlim()[1], axis.get_ylim()[1])
        axis.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1.2)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel(xname)
        axis.set_ylabel(yname)
        axis.grid(alpha=0.25)
    axes[2].set_xlabel("Age (years)")
    axes[2].set_ylabel("Adjacent-scan RMS speed (mm/year)")
    axes[2].set_title("Noise-sensitive adjacent reference (log trend)")
    axes[2].grid(alpha=0.25)
    axes[0].set_title("Shape-speed calibration")
    axes[1].set_title("Atrophy-rate calibration")
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_within_subject_acceleration(slopes: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"instantaneous": "#D62728", "one_year": "#FF8C00", "individual_fitted": "#111111"}
    labels = {
        "instantaneous": "LAMM instantaneous",
        "one_year": "LAMM one-year",
        "individual_fitted": "Individual fitted GT",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    series_order = ("instantaneous", "one_year", "individual_fitted")
    y = np.arange(len(series_order))[::-1]
    for axis, diagnosis in zip(axes, DIAGNOSES):
        current = slopes[slopes.diagnosis == diagnosis].set_index("series")
        for position, series in zip(y, series_order):
            row = current.loc[series]
            axis.errorbar(
                row.slope_per_age_year,
                position,
                xerr=np.asarray([[
                    row.slope_per_age_year - row.ci95_low
                ], [
                    row.ci95_high - row.slope_per_age_year
                ]]),
                fmt="o",
                color=colors[series],
                capsize=4,
                markersize=7,
            )
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1.2)
        axis.set_title(diagnosis)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Within-subject change in speed (mm/year²)")
    axes[0].set_yticks(y, [labels[series] for series in series_order])
    fig.suptitle("Does velocity accelerate within a subject?")
    fig.text(
        0.5,
        0.015,
        "Points are subject-fixed-effect slopes; bars are 95% subject-bootstrap intervals. The individual linear fitted reference is zero by construction.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.allow_test:
        raise ValueError("Test analysis requires --allow-test")
    if args.horizon_years <= 0.0:
        raise ValueError("--horizon-years must be positive")
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    if args.age_max <= args.age_min or args.age_step <= 0.0:
        raise ValueError("Invalid age grid")
    grid = np.arange(args.age_min, args.age_max + 0.5 * args.age_step, args.age_step)
    device = C.choose_device(args.device)
    members = [
        load_member(label, path, args.split, device)
        for label, path in (parse_assignment(value, "--member") for value in args.member)
    ]
    model, model_metadata = compute_model_velocity(members, args.batch_size, args.horizon_years)

    reference = pd.read_csv(args.reference_csv, dtype={"subject_id": str, "scan_id": str})
    validation_table = reference[reference.split == "val"].copy()
    test_table = reference[reference.split == args.split].copy()
    validation_array_file = args.reference_array_root / "val_velocity_reference.npz"
    test_array_file = args.reference_array_root / f"{args.split}_velocity_reference.npz"
    with np.load(validation_array_file, allow_pickle=False) as loaded:
        validation_array = {key: loaded[key] for key in loaded.files}
    with np.load(test_array_file, allow_pickle=False) as loaded:
        test_array = {key: loaded[key] for key in loaded.files}
    selected_degrees, selection_table = select_reference_degrees(validation_array, validation_table)

    merged = model.merge(
        test_table,
        on=["split", "subject_id", "scan_id", "diagnosis", "visit_order"],
        how="inner",
        suffixes=("_model", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(test_table):
        raise ValueError(f"Expected {len(test_table)} matched visits, found {len(merged)}")
    if not np.allclose(
        merged.age_years_model,
        merged.age_years_reference,
        atol=1.0e-3,
        rtol=0.0,
    ):
        raise ValueError("Model/reference table ages do not align")
    merged["age_years"] = merged.pop("age_years_model")
    merged = merged.drop(columns=["age_years_reference"])
    merged["instantaneous_volume_atrophy_pct_per_year"] = -merged.instantaneous_signed_volume_rate_pct_per_year
    merged["one_year_volume_atrophy_pct_per_year"] = -merged.one_year_signed_volume_rate_pct_per_year
    merged["fitted_volume_atrophy_pct_per_year"] = -merged.fit_surface_integral_log_volume_rate_percent_per_year
    merged["fitted_inward_normal_mean_mm_per_year"] = -merged.fit_normal_mean_mm_per_year

    array_order = {str(scan): index for index, scan in enumerate(test_array["scan_ids"])}
    order = np.asarray([array_order[str(scan)] for scan in merged.scan_id], dtype=np.int64)
    if not np.allclose(merged.age_years, test_array["ages_years"][order], atol=1.0e-3):
        raise ValueError("Test reference array/table age mismatch")

    curves: dict[tuple[str, str, str], BootstrapCurve] = {}
    curve_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for diagnosis in DIAGNOSES:
        array_mask = test_array["diagnoses"].astype(str) == diagnosis
        array_age = test_array["ages_years"][array_mask].astype(np.float64)
        array_subjects = test_array["subject_ids"][array_mask].astype(str)
        surface = test_array["aligned_observed"][array_mask].astype(np.float64)
        surface_gt = bootstrap_fixed_effect_curve(
            array_age,
            array_subjects,
            surface,
            selected_degrees["surface"][diagnosis],
            grid,
            args.bootstrap_samples,
            args.seed + seed_offset,
            surface_speed_transform,
        )
        seed_offset += 1
        curves[(diagnosis, "vector_rms", "population_gt")] = surface_gt
        add_curve_rows(curve_rows, diagnosis, "vector_rms", "population_gt", grid, surface_gt)

        group = merged[merged.diagnosis == diagnosis].copy()
        age = group.age_years.to_numpy(dtype=np.float64)
        subjects = group.subject_id.to_numpy(dtype=str)
        model_columns = {
            "vector_rms": {
                "instantaneous": "instantaneous_vector_rms_mm_per_year",
                "one_year": "one_year_vector_rms_mm_per_year",
            },
            "volume_atrophy": {
                "instantaneous": "instantaneous_volume_atrophy_pct_per_year",
                "one_year": "one_year_volume_atrophy_pct_per_year",
            },
            "inward_normal": {
                "instantaneous": "instantaneous_inward_normal_mean_mm_per_year",
                "one_year": "one_year_inward_normal_mean_mm_per_year",
            },
        }
        for metric, series_columns in model_columns.items():
            for series, column in series_columns.items():
                curve = bootstrap_fixed_effect_curve(
                    age,
                    subjects,
                    group[column].to_numpy(dtype=np.float64)[:, None],
                    1,
                    grid,
                    args.bootstrap_samples,
                    args.seed + seed_offset,
                    scalar_level_transform,
                )
                seed_offset += 1
                curves[(diagnosis, metric, series)] = curve
                add_curve_rows(curve_rows, diagnosis, metric, series, grid, curve)

        volume_degree = selected_degrees["log_volume"][diagnosis]
        volume_gt = bootstrap_fixed_effect_curve(
            age,
            subjects,
            np.log(group.observed_volume_mm3.to_numpy(dtype=np.float64))[:, None],
            volume_degree,
            grid,
            args.bootstrap_samples,
            args.seed + seed_offset,
            negative_scalar_derivative_transform,
        )
        seed_offset += 1
        curves[(diagnosis, "volume_atrophy", "population_gt")] = volume_gt
        add_curve_rows(curve_rows, diagnosis, "volume_atrophy", "population_gt", grid, volume_gt)

        fitted_normal = bootstrap_fixed_effect_curve(
            age,
            subjects,
            group.fitted_inward_normal_mean_mm_per_year.to_numpy(dtype=np.float64)[:, None],
            1,
            grid,
            args.bootstrap_samples,
            args.seed + seed_offset,
            scalar_level_transform,
        )
        seed_offset += 1
        curves[(diagnosis, "inward_normal", "individual_fitted")] = fitted_normal
        add_curve_rows(curve_rows, diagnosis, "inward_normal", "individual_fitted", grid, fitted_normal)

        adjacent = bootstrap_fixed_effect_curve(
            age,
            subjects,
            np.log(group.adjacent_vector_rms_mm_per_year.to_numpy(dtype=np.float64))[:, None],
            1,
            grid,
            args.bootstrap_samples,
            args.seed + seed_offset,
            positive_scalar_level_transform,
        )
        seed_offset += 1
        curves[(diagnosis, "adjacent_vector", "adjacent")] = adjacent
        add_curve_rows(curve_rows, diagnosis, "adjacent_vector", "adjacent", grid, adjacent)

    agreement_rows = []
    reference_for_metric = {
        "vector_rms": "population_gt",
        "volume_atrophy": "population_gt",
        "inward_normal": "individual_fitted",
    }
    for diagnosis in DIAGNOSES:
        for metric, reference_name in reference_for_metric.items():
            for model_name in ("instantaneous", "one_year"):
                agreement_rows.append(curve_agreement(
                    diagnosis,
                    metric,
                    model_name,
                    grid,
                    curves[(diagnosis, metric, model_name)],
                    curves[(diagnosis, metric, reference_name)],
                ))

    for metric, reference_name in reference_for_metric.items():
        reference_cn = curves[("CN", metric, reference_name)]
        reference_ad = curves[("AD", metric, reference_name)]
        reference_gap = BootstrapCurve(
            mean=reference_ad.mean - reference_cn.mean,
            low=np.quantile(reference_ad.draws - reference_cn.draws, 0.025, axis=0),
            high=np.quantile(reference_ad.draws - reference_cn.draws, 0.975, axis=0),
            draws=reference_ad.draws - reference_cn.draws,
        )
        for model_name in ("instantaneous", "one_year"):
            model_cn = curves[("CN", metric, model_name)]
            model_ad = curves[("AD", metric, model_name)]
            model_gap = BootstrapCurve(
                mean=model_ad.mean - model_cn.mean,
                low=np.quantile(model_ad.draws - model_cn.draws, 0.025, axis=0),
                high=np.quantile(model_ad.draws - model_cn.draws, 0.975, axis=0),
                draws=model_ad.draws - model_cn.draws,
            )
            agreement_rows.append(curve_agreement(
                "AD_minus_CN",
                metric,
                model_name,
                grid,
                model_gap,
                reference_gap,
            ))

    slope_columns = {
        "instantaneous": "instantaneous_vector_rms_mm_per_year",
        "one_year": "one_year_vector_rms_mm_per_year",
        "individual_fitted": "fit_vector_rms_mm_per_year",
        "adjacent": "adjacent_vector_rms_mm_per_year",
    }
    slopes = subject_slope_summary(merged, slope_columns, args.bootstrap_samples, args.seed + 500)
    calibration = subject_calibration(merged)
    curves_frame = pd.DataFrame(curve_rows)
    agreement_frame = pd.DataFrame(agreement_rows)

    age_75 = int(np.argmin(np.abs(grid - AGE_CENTER)))
    property_summary = {}
    for metric, reference_name in reference_for_metric.items():
        property_summary[metric] = {}
        for series in ("instantaneous", "one_year", reference_name):
            cn = curves[("CN", metric, series)].mean[age_75]
            ad = curves[("AD", metric, series)].mean[age_75]
            property_summary[metric][series] = {
                "cn_at_age_75": float(cn),
                "ad_at_age_75": float(ad),
                "ad_minus_cn_at_age_75": float(ad - cn),
                "ad_to_cn_ratio_at_age_75": float(ad / max(cn, 1.0e-12)),
            }
    report = {
        "schema_version": 1,
        "split": args.split,
        "model": model_metadata,
        "reference": {
            "surface_definition": "subject-fixed-effect population polynomial on rigid-aligned observed vertices",
            "volume_definition": "subject-fixed-effect population polynomial on observed log volume",
            "degree_selection": "validation subject-balanced leave-one-visit-out RMSE; frozen before test fitting",
            "selected_degrees": selected_degrees,
            "direct_instantaneous_gt_available": False,
        },
        "age_grid": {"minimum": float(grid[0]), "maximum": float(grid[-1]), "step": float(args.age_step)},
        "bootstrap": {"samples": args.bootstrap_samples, "unit": "subject within diagnosis"},
        "matching": {
            "model_visits": int(len(model)),
            "reference_visits": int(len(test_table)),
            "matched_visits": int(len(merged)),
            "matched_subjects": int(merged.subject_id.nunique()),
        },
        "cn_ad_property_at_age_75": property_summary,
        "trend_agreement": agreement_rows,
        "scientific_caveats": [
            "The zero-horizon field is outside the directly supervised 0.5-10 year interval support.",
            "The population reference is an estimate from discrete MRI visits, not directly measured instantaneous ground truth.",
            "Chronological age is not disease stage; the C4 condition is binary CN/AD.",
            "Individual fitted and adjacent references answer different questions and are plotted separately.",
        ],
        "source_meshes_modified": False,
    }

    compact = {
        "selected_degrees": selected_degrees,
        "matching": report["matching"],
        "cn_ad_property_at_age_75": property_summary,
        "trend_agreement": agreement_rows,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    if args.dry_run:
        print("DRY RUN PASSED - no files written")
        return 0

    output = C.require_bulk_path(args.output_dir, "velocity trend diagnostics output")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    C.atomic_json(output / "report.json", report)
    model.to_csv(output / "model_velocity_all_visits.csv", index=False)
    merged.to_csv(output / "model_vs_reference_per_scan.csv", index=False)
    selection_table.to_csv(output / "reference_degree_selection.csv", index=False)
    curves_frame.to_csv(output / "continuous_age_curves.csv", index=False)
    agreement_frame.to_csv(output / "trend_agreement.csv", index=False)
    slopes.to_csv(output / "within_subject_acceleration.csv", index=False)
    calibration.to_csv(output / "subject_calibration.csv", index=False)
    plot_population_velocity(
        curves_frame,
        {diagnosis: int(merged[merged.diagnosis == diagnosis].subject_id.nunique()) for diagnosis in DIAGNOSES},
        output / "model_vs_population_gt_by_age.png",
    )
    plot_directed_atrophy(curves_frame, output / "directed_atrophy_by_age.png")
    plot_calibration_and_adjacent(calibration, curves_frame, output / "calibration_and_adjacent_diagnostic.png")
    plot_within_subject_acceleration(slopes, output / "within_subject_acceleration.png")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
