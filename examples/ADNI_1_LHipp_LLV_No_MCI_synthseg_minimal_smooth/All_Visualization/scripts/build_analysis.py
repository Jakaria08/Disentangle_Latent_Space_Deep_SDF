#!/usr/bin/env python3
"""Build compact all-method tables and figures from completed evaluation caches."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-all-visualization")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/all-visualization-xdg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DEFAULT_REGISTRY, atomic_json, ensure_output_tree, output_root, registry, resolve


LABELS = {
    "pca128": "Latent PCA",
    "spiralnet128": "Latent Spiral",
    "adaptive128": "Latent Adaptive",
    "pca_corrective128": "Latent PCA + corrective",
    "inr256": "Latent INR",
    "lamm_global_256": "Direct LAMM global-256",
    "lamm_global_384": "Direct LAMM global-384",
    "lamm_regional_tokens": "Direct LAMM regional tokens",
    "spiral_mesh_direct": "Direct Spiral",
    "adaptive_spiral_mesh_direct": "Direct Adaptive",
    "lamm_latent_flow_128": "Latent LAMM single",
    "mesh_spiral": "Direct Spiral",
    "mesh_adaptive": "Direct Adaptive",
    "latent_pca": "Latent PCA",
    "latent_spiral": "Latent Spiral",
    "latent_adaptive": "Latent Adaptive",
    "latent_inr": "Latent INR",
    "lamm_n3": "Latent LAMM N3 ensemble",
    "pca_plain_ode": "PCA plain ODE",
    "pca_brainode": "PCA BrainODE",
}
NONSELECTED_LAMM_METHODS = {
    "lamm_global_256",
    "lamm_global_384",
    "lamm_regional_tokens",
    "lamm_latent_flow_128",
}
DIRECT_MESH_METHODS = {
    "adaptive_spiral_mesh_direct",
    "spiral_mesh_direct",
    "mesh_adaptive",
    "mesh_spiral",
}


def one_lamm_for_plot(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the validation-retained latent LAMM N3 in report figures."""
    if "method" not in frame:
        return frame
    return frame[~frame.method.astype(str).isin(NONSELECTED_LAMM_METHODS)].copy()


def without_direct_mesh(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the same comparison after omitting both direct mesh methods."""
    if "method" not in frame:
        return frame
    return frame[~frame.method.astype(str).isin(DIRECT_MESH_METHODS)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--require-central-test", action="store_true")
    return parser.parse_args()


def weighted_mean(frame: pd.DataFrame, column: str, weight: str = "reference_reliability") -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(frame[weight], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else math.nan


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def finish(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def central_rows(root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / f"central_{split}"
    subjects: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    if not base.is_dir():
        return pd.DataFrame(), pd.DataFrame()
    for directory in sorted(base.iterdir()):
        per_subject = directory / "per_subject.csv"
        summary_path = directory / "summary.json"
        if not per_subject.is_file() or not summary_path.is_file():
            continue
        current = pd.read_csv(per_subject)
        current["method"] = directory.name
        current["method_label"] = LABELS.get(directory.name, directory.name)
        current["split"] = split
        subjects.append(current)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for diagnosis, group in payload["surface"].items():
            endpoint = payload["endpoint"].get(diagnosis, {})
            velocity = payload["instantaneous_velocity"]["groups"].get(diagnosis, {})
            summaries.append(
                {
                    "method": directory.name,
                    "method_label": LABELS.get(directory.name, directory.name),
                    "split": split,
                    "diagnosis": diagnosis,
                    **group,
                    **{f"endpoint_{key}": value for key, value in endpoint.items()},
                    **{f"velocity_{key}": value for key, value in velocity.items()},
                    **payload.get("cocycle", {}),
                }
            )
    return (pd.concat(subjects, ignore_index=True) if subjects else pd.DataFrame(), pd.DataFrame(summaries))


def strict_prediction_table(config: dict[str, Any], root: Path, central_test: pd.DataFrame) -> pd.DataFrame:
    comparison = resolve(config["existing_caches"]["corrective_pca"]) / "five_model_test_30k"
    old = pd.read_csv(comparison / "matched_five_model_summary.csv")
    old = old[old.diagnosis.isin(["CN", "AD", "overall"])].copy()
    old["method"] = old["representation"]
    old["method_label"] = old.method.map(LABELS).fillna(old.display_name)
    old["model_family"] = np.where(old.representation.eq("inr256"), "implicit latent cocycle", "fixed-topology latent cocycle")
    old["comparison_cohort"] = "strict matched test cohort"
    old["surface_samples"] = 30000
    rename = {
        "subjects": "subjects",
        "prediction_mean_vertex_euclidean_mm_mean": "mean_vertex_error_mm",
        "prediction_coordinate_rmse_mm_mean": "vertex_rmse_mm",
        "prediction_assd_mm_mean": "assd_mm",
        "prediction_hd95_mm_mean": "hd95_mm",
        "prediction_chamfer_l1_mm_mean": "chamfer_l1_mm",
        "prediction_chamfer_l2_squared_mm2_mean": "chamfer_l2_squared_mm2",
        "prediction_volume_relative_error_mean": "volume_relative_error",
        "prediction_normal_signed_cosine_mean": "normal_signed_cosine",
        "prediction_flipped_face_fraction_vs_ground_truth_mean": "flipped_face_fraction",
        "nochange_assd_mm_mean": "nochange_assd_mm",
        "nochange_hd95_mm_mean": "nochange_hd95_mm",
        "nochange_volume_relative_error_mean": "nochange_volume_relative_error",
        "observed_signed_log_volume_rate_per_year_mean": "observed_log_volume_rate_per_year",
        "prediction_signed_log_volume_rate_raw_anchor_per_year_mean": "predicted_log_volume_rate_per_year",
        "prediction_log_volume_rate_absolute_error_per_year_mean": "volume_rate_abs_error_per_year",
    }
    columns = ["method", "method_label", "model_family", "comparison_cohort", "surface_samples", "diagnosis"] + list(rename)
    old = old[[column for column in columns if column in old.columns]].rename(columns=rename)

    matched = pd.read_csv(comparison / "matched_per_subject.csv", dtype={"subject_id": str})
    strict_ids = set(matched.subject_id.astype(str))
    generated: list[dict[str, Any]] = []
    lamm_path = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/ensemble/n3_s42_test/pair_metrics.csv")
    if lamm_path.is_file():
        lamm = pd.read_csv(lamm_path, dtype={"subject": str})
        lamm = lamm[lamm.subject.astype(str).isin(strict_ids)].copy()
        first_source = lamm.groupby("subject").source_index.transform("min")
        last_target = lamm.groupby("subject").target_index.transform("max")
        lamm = lamm[lamm.source_index.eq(first_source) & lamm.target_index.eq(last_target)]
        for diagnosis in ("CN", "AD", "overall"):
            group = lamm if diagnosis == "overall" else lamm[lamm.diagnosis.eq(diagnosis)]
            if group.empty:
                continue
            generated.append({
                "method": "lamm_n3", "method_label": LABELS["lamm_n3"],
                "model_family": "latent cocycle ensemble", "comparison_cohort": "strict matched test cohort",
                "surface_samples": np.nan, "diagnosis": diagnosis,
                "subjects": int(group.subject.nunique()),
                "mean_vertex_error_mm": group.end_to_end_euclidean.mean(),
                "vertex_rmse_mm": group.end_to_end_coordinate_rmse.mean(),
                "assd_mm": np.nan, "hd95_mm": np.nan, "chamfer_l1_mm": np.nan,
                "chamfer_l2_squared_mm2": np.nan, "normal_signed_cosine": np.nan,
                "flipped_face_fraction": np.nan,
                "volume_relative_error": group.volume_relative.mean(),
                "nochange_assd_mm": np.nan, "nochange_hd95_mm": np.nan,
                "nochange_volume_relative_error": group.nochange_volume_relative.mean(),
                "observed_log_volume_rate_per_year": group.observed_signed_rate.mean(),
                "predicted_log_volume_rate_per_year": group.predicted_signed_rate.mean(),
                "volume_rate_abs_error_per_year": group.rate.mean(),
            })
    if not central_test.empty:
        central_test = central_test[central_test.subject.astype(str).isin(strict_ids)].copy()
        for method, method_frame in central_test.groupby("method"):
            for diagnosis in ("CN", "AD", "overall"):
                group = method_frame if diagnosis == "overall" else method_frame[method_frame.diagnosis.eq(diagnosis)]
                if group.empty:
                    continue
                generated.append(
                    {
                        "method": method,
                        "method_label": LABELS.get(method, method),
                        "model_family": "latent cocycle" if method == "lamm_latent_flow_128" else "direct mesh cocycle",
                        "comparison_cohort": "strict matched test cohort",
                        "surface_samples": 10000,
                        "diagnosis": diagnosis,
                        "subjects": int(group.subject.astype(str).nunique()),
                        "mean_vertex_error_mm": group.mean_vertex_error_mm.mean(),
                        "vertex_rmse_mm": group.vertex_rmse_mm.mean(),
                        "assd_mm": group.prediction_assd_mm.mean(),
                        "hd95_mm": group.prediction_hd95_mm.mean(),
                        "chamfer_l1_mm": group.prediction_chamfer_l1_mm.mean(),
                        "chamfer_l2_squared_mm2": group.prediction_chamfer_l2_squared_mm2.mean(),
                        "volume_relative_error": group.volume_relative_error.mean(),
                        "normal_signed_cosine": group.prediction_normal_signed_cosine.mean(),
                        "flipped_face_fraction": group.prediction_flipped_face_fraction_vs_ground_truth.mean(),
                        "nochange_assd_mm": group.nochange_assd_mm.mean(),
                        "nochange_hd95_mm": group.nochange_hd95_mm.mean(),
                        "nochange_volume_relative_error": group.nochange_volume_relative_error.mean(),
                        "observed_log_volume_rate_per_year": group.observed_log_volume_rate_per_year.mean(),
                        "predicted_log_volume_rate_per_year": group.predicted_log_volume_rate_per_year.mean(),
                        "volume_rate_abs_error_per_year": group.volume_rate_abs_error_per_year.mean(),
                    }
                )
    ode_root = resolve(config["pca_ode_baselines"]["surface_output"])
    ode_summary_path = ode_root / "summary.csv"
    ode_manifest_path = ode_root / "summary.json"
    if not ode_summary_path.is_file() or not ode_manifest_path.is_file():
        raise FileNotFoundError(
            f"Matched PCA ODE surface evaluation is missing: {ode_root}; "
            "run the PCA ODE surface command in RUNBOOK.md"
        )
    ode_manifest = json.loads(ode_manifest_path.read_text(encoding="utf-8"))
    evaluated_ids = set(map(str, ode_manifest.get("matched_subject_filter") or []))
    if evaluated_ids != strict_ids:
        raise ValueError(
            "PCA ODE surface cohort does not match the strict comparison cohort: "
            f"expected={len(strict_ids)}, actual={len(evaluated_ids)}"
        )
    ode = pd.read_csv(ode_summary_path)
    ode = ode[ode.diagnosis.isin(["CN", "AD", "overall"])].copy()
    ode["method"] = ode["representation"]
    ode["method_label"] = ode.method.map(LABELS)
    if ode.method_label.isna().any():
        raise ValueError(f"Unexpected PCA ODE surface methods: {sorted(ode.loc[ode.method_label.isna(), 'method'].unique())}")
    ode["model_family"] = "matched PCA ODE baseline"
    ode["comparison_cohort"] = "strict matched test cohort"
    ode["surface_samples"] = int(config["pca_ode_baselines"]["surface_points"])
    ode_rename = {
        "prediction_mean_vertex_euclidean_mm_mean": "mean_vertex_error_mm",
        "prediction_coordinate_rmse_mm_mean": "vertex_rmse_mm",
        "prediction_assd_mm_mean": "assd_mm",
        "prediction_hd95_mm_mean": "hd95_mm",
        "prediction_chamfer_l1_mm_mean": "chamfer_l1_mm",
        "prediction_chamfer_l2_squared_mm2_mean": "chamfer_l2_squared_mm2",
        "prediction_volume_relative_error_mean": "volume_relative_error",
        "prediction_normal_signed_cosine_mean": "normal_signed_cosine",
        "prediction_flipped_face_fraction_vs_ground_truth_mean": "flipped_face_fraction",
        "nochange_assd_mm_mean": "nochange_assd_mm",
        "nochange_hd95_mm_mean": "nochange_hd95_mm",
        "nochange_volume_relative_error_mean": "nochange_volume_relative_error",
        "observed_signed_log_volume_rate_per_year_mean": "observed_log_volume_rate_per_year",
        "prediction_signed_log_volume_rate_raw_anchor_per_year_mean": "predicted_log_volume_rate_per_year",
        "prediction_log_volume_rate_absolute_error_per_year_mean": "volume_rate_abs_error_per_year",
    }
    ode_columns = [
        "method", "method_label", "model_family", "comparison_cohort", "surface_samples", "diagnosis", "subjects",
        *ode_rename,
    ]
    ode = ode[[column for column in ode_columns if column in ode.columns]].rename(columns=ode_rename)
    return pd.concat([old, pd.DataFrame(generated), ode], ignore_index=True, sort=False)


def validation_ranking(central_val: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if central_val.empty:
        return pd.DataFrame()
    for method, frame in central_val.groupby("method"):
        disease = frame[frame.diagnosis.isin(["CN", "AD"])]
        if len(disease) != 2:
            continue
        rows.append(
            {
                "method": method,
                "method_label": LABELS.get(method, method),
                "validation_macro_assd_mm": disease.prediction_assd_mm.mean(),
                "validation_macro_volume_rate_mae_per_year": disease.endpoint_volume_rate_abs_error_per_year.mean(),
                "validation_macro_velocity_normal_rmse_mm_per_year": disease.velocity_normal_rmse_mm_per_year.mean(),
                "maximum_flipped_face_fraction": disease.prediction_flipped_face_fraction_vs_ground_truth.max(),
                "relative_cocycle_defect_mean": disease.relative_cocycle_defect_mean.mean(),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["structural_gate_pass"] = output.maximum_flipped_face_fraction.le(0.001)
    eligible = output.structural_gate_pass
    output["surface_rank"] = np.nan
    output["volume_rank"] = np.nan
    output.loc[eligible, "surface_rank"] = output.loc[eligible, "validation_macro_assd_mm"].rank(method="min")
    output.loc[eligible, "volume_rank"] = output.loc[eligible, "validation_macro_volume_rate_mae_per_year"].rank(method="min")
    output["selection_rank_sum"] = output.surface_rank + output.volume_rank
    output["selected_for_ood"] = False
    if eligible.any():
        winner = output.loc[eligible].sort_values(
            ["selection_rank_sum", "validation_macro_assd_mm", "relative_cocycle_defect_mean", "method"]
        ).index[0]
        output.loc[winner, "selected_for_ood"] = True
    return output.sort_values(["selected_for_ood", "selection_rank_sum"], ascending=[False, True])


def inr_reconstruction(config: dict[str, Any]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for spec in config["inr_representations"]:
        run = resolve(spec["run_dir"])
        candidates = sorted((run / "periodic_evaluation").glob("*/per_scan_metrics.csv"))
        if not candidates:
            continue
        frame = pd.read_csv(candidates[-1])
        frame = frame[frame.method.eq("inr")].copy()
        frame["representation"] = spec["key"]
        frame["representation_label"] = spec["label"]
        records.append(frame)
    if not records:
        return pd.DataFrame()
    combined = pd.concat(records, ignore_index=True)
    rows: list[dict[str, Any]] = []
    metrics = ["assd_mm", "hd95_mm", "chamfer_l1_mm", "volume_relative_error", "normal_signed_cosine", "fscore_0_5mm"]
    for (representation, label, split), frame in combined.groupby(["representation", "representation_label", "split"]):
        for diagnosis in ("CN", "AD", "overall"):
            group = frame if diagnosis == "overall" else frame[frame.diagnosis.eq(diagnosis)]
            if group.empty:
                continue
            row: dict[str, Any] = {
                "representation": representation,
                "representation_label": label,
                "split": split,
                "diagnosis": diagnosis,
                "scans": len(group),
            }
            row.update({metric: pd.to_numeric(group[metric], errors="coerce").mean() for metric in metrics})
            rows.append(row)
    return pd.DataFrame(rows)


def representation_floor(config: dict[str, Any]) -> pd.DataFrame:
    source = resolve(config["existing_caches"]["corrective_pca"]) / "five_model_test_30k" / "matched_five_model_summary.csv"
    frame = pd.read_csv(source)
    frame = frame[frame.diagnosis.eq("overall")].copy()
    frame["method_label"] = frame.representation.map(LABELS).fillna(frame.display_name)
    columns = {
        "floor_mean_vertex_euclidean_mm_mean": "reconstruction_mean_vertex_error_mm",
        "floor_assd_mm_mean": "reconstruction_assd_mm",
        "floor_hd95_mm_mean": "reconstruction_hd95_mm",
        "floor_volume_relative_error_mean": "reconstruction_volume_relative_error",
    }
    output = frame[["representation", "method_label", "subjects"] + list(columns)].rename(columns=columns)
    output["evaluation_items"] = output["subjects"]
    output["scope"] = "strict matched test cohort; one source reconstruction per subject"
    lamm_summary = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/ensemble/n3_s42_test/summary.json")
    if lamm_summary.is_file():
        payload = json.loads(lamm_summary.read_text(encoding="utf-8"))
        values = payload["representation_floor"]
        pair_path = lamm_summary.parent / "pair_metrics.csv"
        lamm_subjects = pd.read_csv(pair_path, usecols=["subject"]).subject.astype(str).nunique()
        output = pd.concat([output, pd.DataFrame([{
            "representation": "lamm_n3", "method_label": LABELS["lamm_n3"],
            "subjects": lamm_subjects, "evaluation_items": values["scans"],
            "reconstruction_mean_vertex_error_mm": values["vertex_euclidean_mm_mean"],
            "reconstruction_assd_mm": np.nan, "reconstruction_hd95_mm": np.nan,
            "reconstruction_volume_relative_error": np.nan,
            "scope": "full test cohort; all reconstructed visits; exact surface metrics not cached",
        }])], ignore_index=True)
    return output


def load_velocity(config: dict[str, Any], root: Path) -> pd.DataFrame:
    base = resolve(config["existing_caches"]["age_velocity"]) / "tables" / "per_visit_velocity.csv"
    frames = [pd.read_csv(base, dtype={"subject_id": str})]
    ode_velocity = resolve(config["pca_ode_baselines"]["velocity_output"]) / "per_scan.csv"
    if not ode_velocity.is_file():
        raise FileNotFoundError(
            f"PCA ODE surface velocity is missing: {ode_velocity}; "
            "run extract_pca_ode_velocity.py first"
        )
    frames.append(pd.read_csv(ode_velocity, dtype={"subject_id": str}))
    direct_root = root / "direct_velocity_test"
    if direct_root.is_dir():
        for path in sorted(direct_root.glob("*/per_scan.csv")):
            current = pd.read_csv(path, dtype={"subject_id": str})
            frames.append(current)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["method_label"] = frame.method.map(LABELS).fillna(frame.method_label)
    frame["age_bin_start"] = (np.floor(frame.age_years / 5.0) * 5.0).astype(int)
    frame["age_bin"] = frame.age_bin_start.map(lambda value: f"{value}–{value + 5}")
    frame["age_year"] = np.floor(frame.age_years).astype(int)
    return frame


def summarize_velocity(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "predicted_speed_mm_per_year", "observed_speed_mm_per_year",
        "predicted_inward_normal_mm_per_year", "observed_inward_normal_mm_per_year",
        "vector_rmse_mm_per_year", "zero_vector_rmse_mm_per_year",
        "normal_rmse_mm_per_year", "zero_normal_rmse_mm_per_year",
        "vector_cosine", "normal_pearson", "normal_sign_agreement",
    ]
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["scans"] = len(group)
        row["subjects"] = group.subject_id.astype(str).nunique() if "subject_id" in group else math.nan
        row["reference_reliability_mean"] = pd.to_numeric(group.reference_reliability, errors="coerce").mean()
        for metric in metrics:
            row[metric] = weighted_mean(group, metric) if metric in group else math.nan
        row["vector_error_to_observed_ratio"] = row["vector_rmse_mm_per_year"] / max(row["zero_vector_rmse_mm_per_year"], 1e-12)
        row["normal_error_to_observed_ratio"] = row["normal_rmse_mm_per_year"] / max(row["zero_normal_rmse_mm_per_year"], 1e-12)
        row["speed_ratio"] = row["predicted_speed_mm_per_year"] / max(row["observed_speed_mm_per_year"], 1e-12)
        rows.append(row)
    output = pd.DataFrame(rows)
    if "diagnosis" in group_columns:
        reduced = [column for column in group_columns if column != "diagnosis"]
        overall = summarize_velocity(frame, reduced)
        overall["diagnosis"] = "overall"
        output = pd.concat([output, overall[output.columns]], ignore_index=True)
    return output


def prediction_plots(frame: pd.DataFrame, figure_dir: Path) -> None:
    overall = one_lamm_for_plot(frame[frame.diagnosis.eq("overall")]).sort_values("assd_mm")
    metrics = [
        ("assd_mm", "ASSD (mm)"), ("hd95_mm", "HD95 (mm)"),
        ("mean_vertex_error_mm", "Mean corresponding-vertex distance (mm)"),
        ("volume_relative_error", "Relative volume error"),
    ]
    variants = [
        (overall, "", "Matched held-out endpoint prediction: lower is better"),
        (
            without_direct_mesh(overall),
            "_without_direct_mesh",
            "Matched held-out endpoint prediction without direct mesh methods",
        ),
    ]
    for selected, suffix, title in variants:
        figure, axes = plt.subplots(2, 2, figsize=(14, 12))
        colors = np.where(selected.model_family.eq("direct mesh cocycle"), "#8c2d91", "#2a6fbb")
        for axis, (metric, label) in zip(axes.flat, metrics):
            current = selected.dropna(subset=[metric]).sort_values(metric, ascending=True)
            axis.barh(
                current.method_label,
                current[metric],
                color=[colors[selected.index.get_loc(i)] for i in current.index],
            )
            axis.set_xlabel(label)
            style_axis(axis)
        figure.suptitle(title, y=1.01)
        figure.tight_layout()
        finish(figure, figure_dir / f"prediction_metrics_strict_test{suffix}.png")

        figure, axes = plt.subplots(1, 2, figsize=(14, 6))
        for axis, metric, baseline, panel_title in (
            (axes[0], "assd_mm", "nochange_assd_mm", "ASSD relative to unchanged source"),
            (axes[1], "volume_relative_error", "nochange_volume_relative_error", "Volume error relative to unchanged source"),
        ):
            current = selected.dropna(subset=[metric, baseline]).copy()
            current["ratio"] = current[metric] / current[baseline].clip(lower=1e-12)
            current = current.sort_values("ratio")
            axis.barh(current.method_label, current.ratio, color="#4e79a7")
            axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_xlabel("Ratio; below 1 improves on unchanged source")
            axis.set_title(panel_title)
            style_axis(axis)
        figure.suptitle(
            "Endpoint improvement over unchanged source"
            + (" without direct mesh methods" if suffix else ""),
            y=1.02,
        )
        figure.tight_layout()
        finish(figure, figure_dir / f"prediction_vs_nochange{suffix}.png")


def inr_plot(frame: pd.DataFrame, figure_dir: Path) -> None:
    current = frame[frame.diagnosis.eq("overall")].copy()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, metric, label in (
        (axes[0], "assd_mm", "ASSD (mm)"),
        (axes[1], "hd95_mm", "HD95 (mm)"),
        (axes[2], "volume_relative_error", "Relative volume error"),
    ):
        pivot = current.pivot(index="split", columns="representation_label", values=metric)
        pivot.plot.bar(ax=axis, color=["#76b7b2", "#e15759"], rot=0)
        axis.set_ylabel(label)
        axis.set_xlabel("Held-out split")
        axis.get_legend().remove()
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("INR representation reconstruction only", y=1.03)
    figure.tight_layout()
    finish(figure, figure_dir / "inr_128_vs_256_reconstruction.png")


def floor_plot(frame: pd.DataFrame, figure_dir: Path) -> None:
    current = frame.sort_values("reconstruction_assd_mm")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    surface = current.dropna(subset=["reconstruction_assd_mm"])
    volume = current.dropna(subset=["reconstruction_volume_relative_error"])
    axes[0].barh(surface.method_label, surface.reconstruction_assd_mm, color="#59a14f")
    axes[0].set_xlabel("Reconstruction ASSD (mm)")
    axes[1].barh(volume.method_label, volume.reconstruction_volume_relative_error, color="#f28e2b")
    axes[1].set_xlabel("Reconstruction relative volume error")
    for axis in axes:
        style_axis(axis)
    figure.tight_layout()
    finish(figure, figure_dir / "representation_reconstruction_floor.png")


def volume_plot(frame: pd.DataFrame, figure_dir: Path) -> None:
    current = one_lamm_for_plot(frame[frame.diagnosis.isin(["CN", "AD"])]).copy()
    for selected, suffix in (
        (current, ""),
        (without_direct_mesh(current), "_without_direct_mesh"),
    ):
        methods = selected.groupby("method_label").assd_mm.mean().sort_values().index.tolist()
        figure, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.35 * len(methods))))
        for axis, diagnosis in zip(axes, ("CN", "AD")):
            group = selected[selected.diagnosis.eq(diagnosis)].set_index("method_label").reindex(methods)
            positions = np.arange(len(methods))
            axis.barh(positions - 0.18, 100 * group.observed_log_volume_rate_per_year, height=0.34, color="#555555", label="Observed longitudinal rate")
            axis.barh(positions + 0.18, 100 * group.predicted_log_volume_rate_per_year, height=0.34, color="#4e79a7", label="Model prediction")
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_yticks(positions, methods)
            axis.set_xlabel("Signed log-volume rate (%/year); negative = loss")
            axis.set_title(diagnosis)
            style_axis(axis)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        figure.suptitle(
            "Observed and predicted first-to-last volume trend"
            + (" without direct mesh methods" if suffix else ""),
            y=1.02,
        )
        figure.tight_layout()
        finish(figure, figure_dir / f"volume_rate_ad_cn{suffix}.png")


def ood_brainode_only_plot(root: Path, figure_dir: Path) -> None:
    """Companion OOD volume plot with both direct mesh methods omitted."""
    table_path = root / "tables" / "ood_best_vs_brainode_trajectory.csv"
    manifest_path = root / "ood_comparison_manifest.json"
    if not table_path.exists() or not manifest_path.exists():
        return
    frame = pd.read_csv(table_path).sort_values("target_age_years")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support_end = float(str(manifest["ood_definition"]).split(">")[-1].strip())
    observed = frame[frame.observed_mesh_available.astype(bool)]

    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(
        frame.target_age_years,
        frame.brainode_volume_mm3,
        marker="s",
        color="#4e79a7",
        linewidth=2.0,
        label="BrainODE prediction",
    )
    axis.scatter(
        observed.target_age_years,
        observed.observed_volume_mm3,
        color="black",
        marker="o",
        s=42,
        zorder=4,
        label="Observed scans",
    )
    axis.axvline(
        support_end,
        color="#e15759",
        linestyle="--",
        linewidth=1.3,
        label="End of training-age support",
    )
    axis.set_xlabel("Age (years)")
    axis.set_ylabel("Mesh volume (mm³)")
    axis.set_title("Observed and BrainODE volume; direct mesh methods omitted")
    style_axis(axis)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    figure.tight_layout()
    finish(figure, figure_dir / "ood_brainode_only_volume_without_direct_mesh.png")


def velocity_plots(binned: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    binned = one_lamm_for_plot(binned)
    summary = one_lamm_for_plot(summary)
    latent_methods = [
        method
        for method in binned.method.unique()
        if str(method).startswith("latent_") or method == "lamm_n3" or str(method).startswith("pca_")
    ]
    for family_name, methods in (
        ("latent", latent_methods),
        ("direct", [method for method in binned.method.unique() if method not in set(latent_methods)]),
    ):
        selected = binned[binned.method.isin(methods)].copy()
        if selected.empty:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
        for axis, diagnosis in zip(axes, ("CN", "AD")):
            group = selected[selected.diagnosis.eq(diagnosis)]
            reference = group.groupby("age_bin_start", as_index=False).observed_speed_mm_per_year.mean()
            axis.plot(reference.age_bin_start + 2.5, reference.observed_speed_mm_per_year, color="black", linestyle="--", linewidth=2.2, label="Observed — fitted trajectory")
            for method, line in group.groupby("method"):
                line = line.sort_values("age_bin_start")
                axis.plot(line.age_bin_start + 2.5, line.predicted_speed_mm_per_year, marker="o", linewidth=1.5, label=LABELS.get(method, method))
            axis.set_title(diagnosis)
            axis.set_xlabel("Age-bin midpoint (years)")
            axis.set_ylabel("Mean surface speed (mm/year)")
            style_axis(axis)
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.08))
        title = (
            "Instantaneous surface velocity by age: latent cocycles and PCA ODE baselines"
            if family_name == "latent"
            else "Cocycle diagonal velocity by age: direct mesh methods"
        )
        figure.suptitle(title, y=1.02)
        figure.tight_layout()
        finish(figure, figure_dir / f"velocity_speed_5year_{family_name}.png")

    current = summary[summary.diagnosis.eq("overall")].sort_values("normal_error_to_observed_ratio")
    figure, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.34 * len(current))))
    axes[0].barh(current.method_label, current.normal_error_to_observed_ratio, color="#e15759")
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Normal-velocity RMSE / zero-prediction RMSE")
    axes[1].barh(current.method_label, current.normal_sign_agreement, color="#76b7b2")
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Normal-direction sign agreement")
    for axis in axes:
        style_axis(axis)
    figure.tight_layout()
    finish(figure, figure_dir / "velocity_agreement_summary.png")

    inward = binned[binned.diagnosis.isin(["CN", "AD"])].pivot_table(
        index=["method", "method_label", "age_bin_start"], columns="diagnosis", values="predicted_inward_normal_mm_per_year"
    ).reset_index()
    if {"AD", "CN"}.issubset(inward.columns):
        inward["ad_minus_cn_inward_mm_per_year"] = inward.AD - inward.CN
        figure, axis = plt.subplots(figsize=(10, 5.5))
        for method, group in inward.groupby("method"):
            group = group.sort_values("age_bin_start")
            axis.plot(group.age_bin_start + 2.5, group.ad_minus_cn_inward_mm_per_year, marker="o", label=LABELS.get(method, method))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Age-bin midpoint (years)")
        axis.set_ylabel("Predicted inward speed: AD − CN (mm/year)")
        style_axis(axis)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        figure.tight_layout()
        finish(figure, figure_dir / "velocity_disease_contrast_5year.png")


def brainode_tables(config: dict[str, Any], table_dir: Path) -> None:
    cache = resolve(config["brainode"]["comparison_cache"])
    for name in (
        "metric_summary.csv", "selected_mesh_metrics.csv", "selected_volume_trajectories.csv",
        "counterfactual_trajectories.csv", "cohort_volume_trend_summary.csv", "cohort_volume_curve.csv",
    ):
        source = cache / "tables" / name
        if source.is_file():
            frame = pd.read_csv(source)
            save_table(frame, table_dir / f"brainode_{name}")


def pca_ode_consistency(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, specification in config["pca_ode_baselines"]["methods"].items():
        run = resolve(specification["run_dir"])
        for split in ("val", "test"):
            path = run / "evaluation" / split / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            defects = payload["consistency_defects"]
            rows.append(
                {
                    "method": method,
                    "method_label": LABELS[method],
                    "relative_cocycle_defect_mean": defects["relative_semigroup_defect_mean"],
                    "relative_cocycle_defect_p95": defects["relative_semigroup_defect_p95"],
                    "relative_inverse_defect_mean": defects["relative_inverse_defect_mean"],
                    "relative_inverse_defect_p95": defects["relative_inverse_defect_p95"],
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    config = registry(args.registry)
    root = output_root(config, args.output_root)
    ensure_output_tree(root)
    (root / "direct_velocity_test").mkdir(parents=True, exist_ok=True)
    table_dir, figure_dir = root / "tables", root / "figures"

    central_val_subjects, central_val_summary = central_rows(root, "val")
    central_test_subjects, central_test_summary = central_rows(root, "test")
    if args.require_central_test and central_test_subjects.empty:
        raise FileNotFoundError(root / "central_test")
    save_table(central_val_summary, table_dir / "central_validation_summary.csv")
    if not central_test_summary.empty:
        save_table(central_test_summary, table_dir / "central_test_summary.csv")

    prediction = strict_prediction_table(config, root, central_test_subjects)
    save_table(prediction, table_dir / "prediction_metrics_strict_test.csv")
    prediction_plots(prediction, figure_dir)

    ranking = validation_ranking(central_val_summary)
    save_table(ranking, table_dir / "validation_best_method_ranking.csv")
    selected = ranking[ranking.selected_for_ood]
    selected_payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "rule": "structural gate, then equal sum of macro ASSD rank and macro absolute log-volume-rate error rank",
        "method": selected.method.iloc[0] if len(selected) else None,
        "method_label": selected.method_label.iloc[0] if len(selected) else None,
        "test_metrics_used_for_selection": False,
    }
    atomic_json(root / "selected_best_method.json", selected_payload)

    reconstruction = representation_floor(config)
    save_table(reconstruction, table_dir / "representation_reconstruction_floor.csv")
    floor_plot(reconstruction, figure_dir)
    inr = inr_reconstruction(config)
    save_table(inr, table_dir / "inr_128_vs_256_reconstruction.csv")
    inr_plot(inr, figure_dir)

    volume_plot(prediction, figure_dir)
    volume_columns = [
        "method", "method_label", "diagnosis", "subjects",
        "observed_log_volume_rate_per_year", "predicted_log_volume_rate_per_year",
        "volume_rate_abs_error_per_year", "volume_relative_error",
    ]
    save_table(prediction[prediction.diagnosis.isin(["CN", "AD"])][volume_columns], table_dir / "volume_rate_ad_cn.csv")

    velocity = load_velocity(config, root)
    save_table(velocity, table_dir / "instantaneous_velocity_per_scan.csv")
    by_five = summarize_velocity(velocity, ["method", "method_label", "diagnosis", "age_bin_start", "age_bin"])
    annual = summarize_velocity(velocity, ["method", "method_label", "diagnosis", "age_year"])
    velocity_summary = summarize_velocity(velocity, ["method", "method_label", "diagnosis"])
    save_table(by_five, table_dir / "instantaneous_velocity_5year.csv")
    save_table(annual, table_dir / "instantaneous_velocity_annual.csv")
    save_table(velocity_summary, table_dir / "instantaneous_velocity_summary.csv")
    velocity_plots(by_five, velocity_summary, figure_dir)

    cocycle_rows: list[pd.DataFrame] = []
    for split, current in (("val", central_val_summary), ("test", central_test_summary)):
        if not current.empty:
            keep = current[current.diagnosis.eq("overall")][[
                "method", "method_label", "relative_cocycle_defect_mean", "relative_cocycle_defect_p95",
                "relative_inverse_defect_mean", "relative_inverse_defect_p95",
            ]].copy()
            keep["split"] = split
            cocycle_rows.append(keep)
    cocycle_rows.append(pca_ode_consistency(config))
    if cocycle_rows:
        save_table(pd.concat(cocycle_rows, ignore_index=True), table_dir / "cocycle_consistency.csv")

    brainode_tables(config, table_dir)
    ood_brainode_only_plot(root, figure_dir)
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "comparison_protocol": {
            "primary_prediction": "strict matched test subjects; first-to-last endpoint",
            "selection": "validation only",
            "surface_sampling": "earlier latent methods 30k; central direct/LAMM and matched PCA ODE methods 10k",
            "instantaneous_velocity": "cocycle diagonal or ODE vector field mapped to corresponding surface; compared with fitted observed longitudinal trajectory",
            "velocity_primary_age_bins_years": 5,
            "annual_velocity_table_also_written": True,
            "inr_128_256": "representation reconstruction only; only INR-256 has a trained cocycle flow",
            "direct_reconstruction": "not applicable because direct mesh transport has no autoencoder bottleneck",
            "latent_lamm_surface": "cached LAMM-N3 corresponding-vertex, volume, and velocity results are included; ASSD/HD95 remain unavailable because the pinned legacy evaluator rejected the changed builder hash",
            "pca_ode_baselines": "matched PCA plain ODE and singleton-attention PCA BrainODE use the identical PCA archive and strict subject cohort; their diagonal fields are decoded to physical surface velocity with a Jacobian-vector product",
        },
        "selected_best_method": selected_payload,
        "rows": {
            "prediction_summary": len(prediction),
            "prediction_methods": int(prediction[prediction.diagnosis.eq("overall")].method.nunique()),
            "velocity_per_scan": len(velocity),
            "velocity_methods": int(velocity.method.nunique()),
            "velocity_5year": len(by_five),
            "inr_reconstruction": len(inr),
        },
    }
    atomic_json(root / "analysis_manifest.json", provenance)
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
