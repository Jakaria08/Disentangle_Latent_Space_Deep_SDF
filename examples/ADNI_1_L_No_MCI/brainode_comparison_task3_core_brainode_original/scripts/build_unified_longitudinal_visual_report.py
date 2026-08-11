#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/unified_longitudinal_visual_report"
)
FAIR_MESH_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/future_mesh_forecast_comparison"
)
PCA_FLOW_DIR = (
    "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
    "pca150_direct_cocycle_flow_qc_v1"
)
OLD_BRAINODE_REPORT_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/best_model_brainode_comparison"
)
OLD_VS_QC_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/old_vs_qc_longitudinal_comparison"
)

MESH_METRICS = [
    "chamfer_l2_squared",
    "chamfer_l1",
    "assd",
    "hd95",
    "volume_relative_error",
    "surface_area_relative_error",
]
PCA_EXTRA_METRICS = [
    "endpoint_vertex_mae",
    "endpoint_vertex_rmse",
    "endpoint_pca_mse",
]
SDF_SUMMARY_METRICS = [
    "model_target_sdf_l1_mean",
    "composed_target_sdf_l1_mean",
    "no_change_target_sdf_l1_mean",
    "sdf_l1_improvement_mean",
    "composed_sdf_l1_improvement_mean",
    "model_beats_no_change_fraction",
    "composed_beats_no_change_fraction",
    "target_latent_mse_diagnostic_mean",
    "displacement_magnitude_abs_error_mean",
]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    dataset: str
    family: str
    priority: int
    experiment_dir: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "pca150_direct_cocycle_flow": ModelSpec(
        "QC PCA150 cocycle flow",
        "qc_large",
        "PCA150 flow",
        1,
        PCA_FLOW_DIR,
    ),
    "qc_brainode_pca150": ModelSpec(
        "QC BrainODE PCA150",
        "qc_large",
        "BrainODE PCA150",
        2,
    ),
    "qc_siren_drop_bad_min2": ModelSpec(
        "QC SIREN direct flow",
        "qc_large",
        "SIREN flow",
        3,
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2",
    ),
    "qc_siren_latent_ode": ModelSpec(
        "QC SIREN latent ODE",
        "qc_large",
        "SIREN latent ODE",
        4,
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1",
    ),
    "qc_siren_local_decomp_volume": ModelSpec(
        "QC local-decomp SIREN",
        "qc_large",
        "SIREN local decomposed volume",
        5,
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "siren_no_skip_local_decomposed_flow_qc_volume_v1",
    ),
    "qc_siren_seq_rollout": ModelSpec(
        "QC SIREN seq rollout",
        "qc_large",
        "SIREN sequence rollout",
        6,
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1",
    ),
    "qc_siren_structured_latents": ModelSpec(
        "QC structured-latent SIREN",
        "qc_large",
        "SIREN structured subject latents",
        7,
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1",
    ),
    "qc_siren_original_observed_cocycle": ModelSpec(
        "QC SIREN original observed cocycle",
        "qc_large",
        "SIREN flow",
        8,
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "siren_no_skip_flow_real_sdf_observed_virtual_cocycle",
    ),
    "old_brainode_pca150": ModelSpec("Old BrainODE PCA150", "old_adni", "BrainODE PCA150", 20),
    "old_pca32_siren": ModelSpec("Old PCA32 SIREN flow", "old_adni", "SIREN PCA32 flow", 21),
    "old_smallnet_deepsdf": ModelSpec("Old small-net DeepSDF", "old_adni", "DeepSDF small-net flow", 22),
    "old_siren_full256_direct": ModelSpec("Old full256 SIREN flow", "old_adni", "SIREN flow", 23),
    "old_siren_backward_timeflow": ModelSpec(
        "Old SIREN backward time-flow",
        "old_adni",
        "SIREN flow",
        24,
    ),
    "old_siren_sequence_guard": ModelSpec(
        "Old SIREN sequence cocycle guard",
        "old_adni",
        "SIREN flow",
        25,
    ),
}

SIREN_SUMMARY_FOLDERS = {
    "qc_siren_drop_bad_min2": MODEL_SPECS["qc_siren_drop_bad_min2"].experiment_dir,
    "qc_siren_latent_ode": MODEL_SPECS["qc_siren_latent_ode"].experiment_dir,
    "qc_siren_local_decomp_volume": MODEL_SPECS["qc_siren_local_decomp_volume"].experiment_dir,
    "qc_siren_seq_rollout": MODEL_SPECS["qc_siren_seq_rollout"].experiment_dir,
    "qc_siren_structured_latents": MODEL_SPECS["qc_siren_structured_latents"].experiment_dir,
}

EXTRA_VOLUME_TREND_FILES = {
    "qc_siren_original_observed_cocycle": (
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "siren_no_skip_flow_real_sdf_observed_virtual_cocycle/analysis/notebook_best/"
        "selected_observed_age_volume_trend.csv"
    ),
    "old_siren_backward_timeflow": (
        "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_backward_timeflow_future/"
        "analysis/notebook_best/batch_observed_age_volume_trend.csv"
    ),
    "old_siren_sequence_guard": (
        "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_sequence_cocycle_latent_guard/"
        "analysis/notebook_best/batch_observed_age_volume_trend.csv"
    ),
}

VOLUME_AUDIT_DIR = (
    "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
    "analysis/available_representation_volume_audit/full_large_siren"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a unified HTML visualization report from existing longitudinal "
            "BrainODE, PCA-flow, SIREN-flow, and SIREN-ODE outputs."
        )
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-mesh-dir", default=FAIR_MESH_DIR)
    parser.add_argument("--pca-flow-dir", default=PCA_FLOW_DIR)
    parser.add_argument("--include-plotlyjs", default="cdn", choices=("cdn", "inline"))
    parser.add_argument(
        "--generated-artifact-dir",
        default="",
        help="Optional output from generate_unified_longitudinal_artifacts.py.",
    )
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: str | Path) -> pd.DataFrame:
    resolved = repo_path(path)
    if not resolved.is_file():
        return pd.DataFrame()
    return pd.read_csv(resolved)


def read_json(path: str | Path) -> Any:
    resolved = repo_path(path)
    if not resolved.is_file():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def model_label(model: str) -> str:
    return MODEL_SPECS.get(str(model), ModelSpec(str(model), "", "", 999)).label


def model_priority(model: str) -> int:
    return MODEL_SPECS.get(str(model), ModelSpec(str(model), "", "", 999)).priority


def dataset_label(dataset: str) -> str:
    labels = {
        "qc_large": "QC-large strict-left",
        "old_adni": "Old ADNI subset",
    }
    return labels.get(str(dataset), str(dataset))


def transport_label(value: str) -> str:
    mapping = {
        "brainode_endpoint": "endpoint",
        "direct": "direct",
        "composed": "composed",
        "composed_observed": "composed",
        "composed_fixed_year_steps": "composed",
        "model_no_change": "no-change",
        "direct_from_base": "direct",
        "composed_from_base": "composed",
        "composed_observed_from_baseline": "composed",
        "direct_from_baseline": "direct",
        "real_observed": "GT",
        "base_reconstruction_no_change": "no-change",
    }
    return mapping.get(str(value), str(value))


def add_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    if "model" in frame.columns:
        frame["model_label"] = frame["model"].map(model_label)
        frame["model_priority"] = frame["model"].map(model_priority)
    if "dataset" in frame.columns:
        frame["dataset_label"] = frame["dataset"].map(dataset_label)
    if "transport_method" in frame.columns:
        frame["transport_label"] = frame["transport_method"].map(transport_label)
    return frame


def collect_selected_subjects(generated_artifact_dir: Path | None) -> pd.DataFrame:
    if not generated_artifact_dir:
        return pd.DataFrame()
    selected = read_csv(generated_artifact_dir / "selected_subject_manifest.csv")
    if selected.empty:
        case_manifest = read_csv(generated_artifact_dir / "surface_change_case_manifest.csv")
        if not case_manifest.empty and {"dataset", "split", "diagnosis", "subject_id"}.issubset(case_manifest.columns):
            selected = case_manifest[["dataset", "split", "diagnosis", "subject_id"]].drop_duplicates()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.copy()
    for column in ["dataset", "split", "diagnosis", "subject_id"]:
        if column in selected.columns:
            selected[column] = selected[column].astype(str)
    return add_display_columns(selected)


def filter_to_selected_subjects(frame: pd.DataFrame, selected_subjects: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or selected_subjects.empty:
        return frame
    required = {"dataset", "split", "subject_id"}
    if not required.issubset(frame.columns) or not required.issubset(selected_subjects.columns):
        return frame
    working = frame.copy()
    selected = selected_subjects[["dataset", "split", "subject_id"]].drop_duplicates().copy()
    for column in ["dataset", "split", "subject_id"]:
        working[column] = working[column].astype(str)
        selected[column] = selected[column].astype(str)
    filtered = working.merge(selected, on=["dataset", "split", "subject_id"], how="inner")
    return add_display_columns(filtered)


def collect_mesh_pairs(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fair = read_csv(repo_path(args.future_mesh_dir) / "future_mesh_per_pair.csv")
    if not fair.empty:
        fair["source_table"] = str(repo_path(args.future_mesh_dir) / "future_mesh_per_pair.csv")
        frames.append(fair)

    pca_dir = repo_path(args.pca_flow_dir) / "analysis" / "checkpoint_best_val_endpoint_vertex_mae"
    pca = read_csv(pca_dir / "pca_flow_per_pair.csv")
    if not pca.empty:
        pca["source_table"] = str(pca_dir / "pca_flow_per_pair.csv")
        frames.append(pca)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = coerce_numeric(
        combined,
        [
            *MESH_METRICS,
            *PCA_EXTRA_METRICS,
            "source_age_years",
            "target_age_years",
            "gap_years",
            "predicted_volume",
            "target_volume",
            "predicted_surface_area",
            "target_surface_area",
        ],
    )
    return add_display_columns(combined)


def summarize_numeric(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    existing_values = [column for column in value_cols if column in frame.columns]
    if frame.empty or not existing_values:
        return pd.DataFrame()
    grouped = frame.groupby(list(group_cols), dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_cols, key)}
        row["rows"] = int(len(group))
        for value_col in existing_values:
            values = pd.to_numeric(group[value_col], errors="coerce").dropna()
            row[f"{value_col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{value_col}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{value_col}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
        rows.append(row)
    return add_display_columns(pd.DataFrame(rows))


def collect_mesh_summaries(mesh_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mesh_pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_cols = [*MESH_METRICS, *PCA_EXTRA_METRICS]
    live = mesh_pairs.loc[mesh_pairs["transport_method"].astype(str) != "model_no_change"].copy()
    split = summarize_numeric(
        live,
        ["dataset", "model", "family", "split", "transport_method"],
        metric_cols,
    )
    diag = summarize_numeric(
        live,
        ["dataset", "model", "family", "split", "diagnosis", "pair_type", "transport_method"],
        metric_cols,
    )
    return split, diag


def collect_sdf_summaries() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model, folder in SIREN_SUMMARY_FOLDERS.items():
        analysis_dir = repo_path(folder) / "analysis" / "checkpoint_best"
        for split in ("train", "val", "test"):
            path = analysis_dir / f"{split}_summary.csv"
            frame = read_csv(path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["dataset"] = MODEL_SPECS[model].dataset
            frame["model"] = model
            frame["family"] = MODEL_SPECS[model].family
            frame["split"] = split
            frame["source_table"] = str(path)
            rows.append(frame)

    old = read_csv(repo_path(OLD_BRAINODE_REPORT_DIR) / "split_numeric_metrics.csv")
    if not old.empty:
        old_rows: list[dict[str, Any]] = []
        for _, row in old.iterrows():
            method = str(row.get("method", ""))
            old_rows.append(
                {
                    "dataset": row.get("dataset", "old_adni"),
                    "model": method,
                    "family": row.get("family", ""),
                    "split": row.get("split", ""),
                    "grouping": "overall",
                    "rows": row.get("rows", np.nan),
                    "model_target_sdf_l1_mean": row.get("model_error_mean", np.nan),
                    "no_change_target_sdf_l1_mean": row.get("no_change_error_mean", np.nan),
                    "sdf_l1_improvement_mean": row.get("improvement_mean", np.nan),
                    "model_beats_no_change_fraction": row.get(
                        "beats_no_change_fraction", np.nan
                    ),
                    "composed_target_sdf_l1_mean": row.get("composed_error_mean", np.nan),
                    "composed_sdf_l1_improvement_mean": row.get(
                        "composed_improvement_mean", np.nan
                    ),
                    "composed_beats_no_change_fraction": row.get(
                        "composed_beats_no_change_fraction", np.nan
                    ),
                    "source_table": row.get("source_file", ""),
                }
            )
        rows.append(pd.DataFrame(old_rows))

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = coerce_numeric(combined, [*SDF_SUMMARY_METRICS, "rows"])
    return add_display_columns(combined)


def load_torch_logs(path: Path) -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu")


def collect_training_history() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    history_specs = {
        "pca150_direct_cocycle_flow": repo_path(PCA_FLOW_DIR) / "history.json",
        "qc_brainode_pca150": repo_path(
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "brainode_pca150_qc_stable/training/core_attention_pca150_qc_stable/history.json"
        ),
        "old_brainode_pca150": repo_path(
            "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
            "training/core_attention_pca150/history.json"
        ),
    }
    for model, path in history_specs.items():
        data = read_json(path)
        if not isinstance(data, list):
            continue
        for entry in data:
            epoch = int(entry.get("epoch", len(rows) + 1))
            for key, value in entry.items():
                numeric = finite_float(value)
                if key == "epoch" or not math.isfinite(numeric):
                    continue
                split = "val" if key.startswith("val_") else "train" if key.startswith("train_") else "train"
                rows.append(
                    {
                        "dataset": MODEL_SPECS[model].dataset,
                        "model": model,
                        "family": MODEL_SPECS[model].family,
                        "epoch": epoch,
                        "split": split,
                        "metric": key,
                        "value": numeric,
                        "source_table": str(path),
                    }
                )

    for model, folder in SIREN_SUMMARY_FOLDERS.items():
        path = repo_path(folder) / "Logs.pth"
        logs = load_torch_logs(path)
        if not isinstance(logs, dict):
            continue
        for split_key, split_name in (("train", "train"), ("validation", "val")):
            entries = logs.get(split_key, [])
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries, start=1):
                if not isinstance(entry, dict):
                    continue
                epoch = int(entry.get("epoch", index))
                for key, value in entry.items():
                    numeric = finite_float(value)
                    if key == "epoch" or not math.isfinite(numeric):
                        continue
                    rows.append(
                        {
                            "dataset": MODEL_SPECS[model].dataset,
                            "model": model,
                            "family": MODEL_SPECS[model].family,
                            "epoch": epoch,
                            "split": split_name,
                            "metric": key,
                            "value": numeric,
                            "source_table": str(path),
                        }
                    )
    if not rows:
        return pd.DataFrame()
    return add_display_columns(pd.DataFrame(rows))


def derive_volume_pair_velocity(mesh_pairs: pd.DataFrame) -> pd.DataFrame:
    if mesh_pairs.empty:
        return pd.DataFrame()
    required = {
        "dataset",
        "model",
        "split",
        "subject_id",
        "source_scan_id",
        "target_scan_id",
        "transport_method",
        "predicted_volume",
        "target_volume",
        "predicted_surface_area",
        "target_surface_area",
        "gap_years",
    }
    if not required.issubset(mesh_pairs.columns):
        return pd.DataFrame()
    key_cols = ["dataset", "model", "split", "subject_id", "source_scan_id", "target_scan_id"]
    no_change = mesh_pairs.loc[
        mesh_pairs["transport_method"].astype(str) == "model_no_change",
        [*key_cols, "predicted_volume", "predicted_surface_area"],
    ].rename(
        columns={
            "predicted_volume": "source_model_volume",
            "predicted_surface_area": "source_model_surface_area",
        }
    )
    live = mesh_pairs.loc[mesh_pairs["transport_method"].astype(str) != "model_no_change"].copy()
    live = live.merge(no_change, on=key_cols, how="left")
    live["gap_years"] = pd.to_numeric(live["gap_years"], errors="coerce")
    positive_gap = live["gap_years"].abs().clip(lower=1.0e-8)
    live["real_volume_velocity_per_year"] = (
        pd.to_numeric(live["target_volume"], errors="coerce")
        - pd.to_numeric(live["source_model_volume"], errors="coerce")
    ) / positive_gap
    live["predicted_volume_velocity_per_year"] = (
        pd.to_numeric(live["predicted_volume"], errors="coerce")
        - pd.to_numeric(live["source_model_volume"], errors="coerce")
    ) / positive_gap
    live["volume_velocity_error_per_year"] = (
        live["predicted_volume_velocity_per_year"] - live["real_volume_velocity_per_year"]
    )
    live["volume_velocity_abs_error_per_year"] = live["volume_velocity_error_per_year"].abs()
    live["real_surface_area_velocity_per_year"] = (
        pd.to_numeric(live["target_surface_area"], errors="coerce")
        - pd.to_numeric(live["source_model_surface_area"], errors="coerce")
    ) / positive_gap
    live["predicted_surface_area_velocity_per_year"] = (
        pd.to_numeric(live["predicted_surface_area"], errors="coerce")
        - pd.to_numeric(live["source_model_surface_area"], errors="coerce")
    ) / positive_gap
    live["surface_area_velocity_error_per_year"] = (
        live["predicted_surface_area_velocity_per_year"]
        - live["real_surface_area_velocity_per_year"]
    )
    live["surface_area_velocity_abs_error_per_year"] = live[
        "surface_area_velocity_error_per_year"
    ].abs()
    return add_display_columns(live)


def collect_latent_velocity_from_sdf_pair_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model, folder in SIREN_SUMMARY_FOLDERS.items():
        path = repo_path(folder) / "analysis" / "checkpoint_best" / "all_pair_metrics.csv"
        frame = read_csv(path)
        if frame.empty:
            continue
        needed = {"gap_years", "gap_norm", "real_displacement_l2", "predicted_displacement_l2"}
        if not needed.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["dataset"] = MODEL_SPECS[model].dataset
        frame["model"] = model
        frame["family"] = MODEL_SPECS[model].family
        frame["source_table"] = str(path)
        for col in [
            "gap_years",
            "gap_norm",
            "real_displacement_l2",
            "predicted_displacement_l2",
            "composed_predicted_displacement_l2",
            "predicted_speed_l2_per_normalized_age",
            "composed_speed_l2_per_normalized_age",
            "source_velocity_l2_per_normalized_age",
            "target_velocity_l2_per_normalized_age",
        ]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        gap_years = frame["gap_years"].abs().clip(lower=1.0e-8)
        age_range = frame["gap_years"] / frame["gap_norm"].replace(0.0, np.nan)
        frame["real_latent_displacement_l2_per_year"] = frame["real_displacement_l2"] / gap_years
        frame["predicted_latent_displacement_l2_per_year"] = (
            frame["predicted_displacement_l2"] / gap_years
        )
        if "composed_predicted_displacement_l2" in frame.columns:
            frame["composed_latent_displacement_l2_per_year"] = (
                frame["composed_predicted_displacement_l2"] / gap_years
            )
        for source_col, target_col in [
            ("predicted_speed_l2_per_normalized_age", "average_predicted_speed_l2_per_year"),
            ("composed_speed_l2_per_normalized_age", "average_composed_speed_l2_per_year"),
            ("source_velocity_l2_per_normalized_age", "source_neighbor_speed_l2_per_year"),
            ("target_velocity_l2_per_normalized_age", "target_neighbor_speed_l2_per_year"),
        ]:
            if source_col in frame.columns:
                frame[target_col] = frame[source_col] / age_range
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return add_display_columns(pd.concat(rows, ignore_index=True, sort=False))


def collect_volume_trends() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pca_path = repo_path(PCA_FLOW_DIR) / "analysis" / "checkpoint_best_val_endpoint_vertex_mae" / "pca_flow_volume_trends.csv"
    pca = read_csv(pca_path)
    if not pca.empty:
        pca = pca.copy()
        pca["dataset"] = "qc_large"
        pca["model"] = "pca150_direct_cocycle_flow"
        pca["family"] = MODEL_SPECS["pca150_direct_cocycle_flow"].family
        pca["source_table"] = str(pca_path)
        frames.append(pca)

    for model, folder in SIREN_SUMMARY_FOLDERS.items():
        path = repo_path(folder) / "analysis" / "notebook_best" / "selected_observed_age_volume_trend.csv"
        frame = read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset"] = MODEL_SPECS[model].dataset
        frame["model"] = model
        frame["family"] = MODEL_SPECS[model].family
        frame["source_table"] = str(path)
        frames.append(frame)

    for model, path_text in EXTRA_VOLUME_TREND_FILES.items():
        path = repo_path(path_text)
        frame = read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset"] = MODEL_SPECS[model].dataset
        frame["model"] = model
        frame["family"] = MODEL_SPECS[model].family
        frame["source_table"] = str(path)
        frames.append(frame)

    old_subject_trends = {
        "old_siren_full256_direct": (
            "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized/"
            "analysis/notebook_best/batch_observed_age_volume_trend.csv"
        ),
    }
    for model, path_text in old_subject_trends.items():
        path = repo_path(path_text)
        frame = read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset"] = MODEL_SPECS[model].dataset
        frame["model"] = model
        frame["family"] = MODEL_SPECS[model].family
        frame["source_table"] = str(path)
        frames.append(frame)

    old = read_csv(repo_path(OLD_BRAINODE_REPORT_DIR) / "volume_trend_by_split.csv")
    if not old.empty:
        old = old.copy()
        old["model"] = old["method"].astype(str)
        old["family"] = old["method"].astype(str)
        old["source_table"] = str(repo_path(OLD_BRAINODE_REPORT_DIR) / "volume_trend_by_split.csv")
        frames.append(old)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return add_display_columns(combined)


def collect_volume_audit_tables() -> dict[str, pd.DataFrame]:
    audit_dir = repo_path(VOLUME_AUDIT_DIR)
    tables = {
        "scan_volumes": read_csv(audit_dir / "scan_volumes.csv"),
        "pair_volume_rates": read_csv(audit_dir / "pair_volume_rates.csv"),
        "subject_volume_summary": read_csv(audit_dir / "subject_volume_summary.csv"),
        "group_volume_summary": read_csv(audit_dir / "group_volume_summary.csv"),
    }
    for name, frame in list(tables.items()):
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_table"] = str(audit_dir / f"{name}.csv")
        frame["dataset"] = frame.get("cohort", "volume_audit").astype(str)
        frame["model"] = "volume_audit_" + frame.get("method", "unknown").astype(str)
        frame["family"] = "whole-dataset volume audit"
        if "subject_diagnosis" in frame.columns and "diagnosis" not in frame.columns:
            frame["diagnosis"] = frame["subject_diagnosis"]
        tables[name] = add_display_columns(frame)
    return tables


def collect_ood_rows(generated_artifact_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    pca_dir = repo_path(PCA_FLOW_DIR) / "analysis" / "checkpoint_best_val_endpoint_vertex_mae"
    pca = read_csv(pca_dir / "pca_flow_ood_conditional_forecasts.csv")
    if not pca.empty:
        pca = pca.copy()
        pca["dataset"] = "qc_large"
        pca["model"] = "pca150_direct_cocycle_flow"
        pca["source_table"] = str(pca_dir / "pca_flow_ood_conditional_forecasts.csv")
        rows.append(pca)
    pca_summary = read_csv(pca_dir / "pca_flow_ood_summary.csv")
    if not pca_summary.empty:
        pca_summary = pca_summary.copy()
        pca_summary["dataset"] = "qc_large"
        pca_summary["model"] = "pca150_direct_cocycle_flow"
        pca_summary["source_table"] = str(pca_dir / "pca_flow_ood_summary.csv")
        summaries.append(pca_summary)

    if generated_artifact_dir:
        for stem, bucket in [
            ("pca_ood_forecasts.csv", rows),
            ("pca_ood_summary.csv", summaries),
        ]:
            frame = read_csv(generated_artifact_dir / stem)
            if not frame.empty:
                frame["source_table"] = str(generated_artifact_dir / stem)
                bucket.append(frame)

    for model, folder in SIREN_SUMMARY_FOLDERS.items():
        path = repo_path(folder) / "analysis" / "notebook_best" / "counterfactual_condition_summary.csv"
        frame = read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset"] = MODEL_SPECS[model].dataset
        frame["model"] = model
        frame["source_table"] = str(path)
        rows.append(frame)

    return (
        add_display_columns(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame(),
        add_display_columns(pd.concat(summaries, ignore_index=True, sort=False))
        if summaries
        else pd.DataFrame(),
    )


def collect_coverage(
    *,
    args: argparse.Namespace,
    mesh_pairs: pd.DataFrame,
    sdf_summary: pd.DataFrame,
    training: pd.DataFrame,
    volume_trends: pd.DataFrame,
    volume_velocity: pd.DataFrame,
    volume_audit_tables: dict[str, pd.DataFrame],
    latent_velocity: pd.DataFrame,
    ood_rows: pd.DataFrame,
    generated_artifact_dir: Path | None,
) -> pd.DataFrame:
    checks = [
        ("fair_mesh_per_pair", repo_path(args.future_mesh_dir) / "future_mesh_per_pair.csv", len(mesh_pairs)),
        (
            "pca_flow_per_pair",
            repo_path(args.pca_flow_dir)
            / "analysis"
            / "checkpoint_best_val_endpoint_vertex_mae"
            / "pca_flow_per_pair.csv",
            len(mesh_pairs.loc[mesh_pairs.get("model", pd.Series(dtype=str)).astype(str) == "pca150_direct_cocycle_flow"])
            if not mesh_pairs.empty
            else 0,
        ),
        ("sdf_summary", Path("derived"), len(sdf_summary)),
        ("training_history", Path("derived"), len(training)),
        ("volume_trends", Path("derived"), len(volume_trends)),
        ("volume_velocity", Path("derived"), len(volume_velocity)),
        (
            "whole_dataset_volume_audit_group_summary",
            repo_path(VOLUME_AUDIT_DIR) / "group_volume_summary.csv",
            len(volume_audit_tables.get("group_volume_summary", pd.DataFrame())),
        ),
        (
            "whole_dataset_volume_audit_scan_volumes",
            repo_path(VOLUME_AUDIT_DIR) / "scan_volumes.csv",
            len(volume_audit_tables.get("scan_volumes", pd.DataFrame())),
        ),
        ("latent_velocity", Path("derived"), len(latent_velocity)),
        ("ood_rows", Path("derived"), len(ood_rows)),
    ]
    if generated_artifact_dir:
        checks.append(
            (
                "generated_instantaneous_velocity",
                generated_artifact_dir / "instantaneous_velocity.csv",
                len(read_csv(generated_artifact_dir / "instantaneous_velocity.csv")),
            )
        )
    rows = []
    for name, path, row_count in checks:
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": bool(path.exists()) if path != Path("derived") else True,
                "rows": int(row_count),
                "status": "ok" if int(row_count) > 0 else "missing_or_empty",
            }
        )
    return pd.DataFrame(rows)


def table_html(frame: pd.DataFrame, columns: Sequence[str], *, max_rows: int = 40) -> str:
    if frame.empty:
        return "<p class=\"note\">No rows available.</p>"
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "<p class=\"note\">No requested columns are available.</p>"
    view = frame.loc[:, cols].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: f"{value:.6g}" if pd.notna(value) and math.isfinite(float(value)) else ""
            )
    return view.to_html(index=False, escape=True, classes="data-table")


def figure_html(fig: go.Figure, include_plotlyjs: str = "cdn") -> str:
    include = "cdn" if include_plotlyjs == "cdn" else True
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(size=13),
    )
    return pio.to_html(fig, include_plotlyjs=include, full_html=False)


def write_page(output_dir: Path, relative_path: str, title: str, body: str) -> None:
    path = output_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "../" if "/" in relative_path else ""
    css = """
body { font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f7f8fb; }
header { padding: 22px 28px; background: white; border-bottom: 1px solid #d9dee7; }
main { max-width: 1440px; margin: 0 auto; padding: 22px 28px 40px; }
nav a { display: inline-block; margin: 0 16px 8px 0; color: #0f5f8f; text-decoration: none; }
section { background: white; border: 1px solid #d9dee7; border-radius: 6px; margin: 0 0 22px; padding: 18px; overflow: hidden; }
h1 { margin: 0 0 8px; font-size: 24px; }
h2 { margin: 0 0 12px; font-size: 19px; }
h3 { margin: 16px 0 10px; font-size: 16px; }
.note { color: #52606d; line-height: 1.45; max-width: 1080px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; align-items: start; }
.table-wrap { overflow-x: auto; max-width: 100%; }
table.data-table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: auto; }
.data-table th, .data-table td { border-bottom: 1px solid #e5e9f0; padding: 6px 8px; text-align: left; white-space: nowrap; }
.data-table th { background: #eef2f7; }
code { background: #eef2f7; padding: 2px 5px; border-radius: 4px; }
"""
    nav = f"""
<nav>
<a href="{prefix}index.html">Overview</a>
<a href="{prefix}pages/metrics_by_split.html">Metrics</a>
<a href="{prefix}pages/loss_curves.html">Loss</a>
<a href="{prefix}pages/all_metrics.html">All Metrics</a>
<a href="{prefix}pages/volume_trends.html">Volume</a>
<a href="{prefix}pages/velocity.html">Velocity</a>
<a href="{prefix}pages/ood_future_volume.html">OOD</a>
<a href="{prefix}pages/surface_change_cases.html">Surface Change</a>
<a href="{prefix}pages/coverage_and_missing.html">Coverage</a>
</nav>
"""
    text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<header>
<h1>{html.escape(title)}</h1>
{nav}
</header>
<main>
{body}
</main>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def metric_bar_figure(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    column = f"{metric}_mean"
    if summary.empty or column not in summary.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {metric}", showarrow=False)
        return fig
    view = summary.dropna(subset=[column]).copy()
    view = view.sort_values(["dataset", "split", "model_priority", "transport_label"])
    fig = px.bar(
        view,
        x="model_label",
        y=column,
        color="transport_label",
        facet_col="split",
        facet_row="dataset_label",
        barmode="group",
        title=title,
        labels={column: metric, "model_label": "model", "transport_label": "transport"},
    )
    fig.update_xaxes(tickangle=35)
    return fig


def training_figure(training: pd.DataFrame, model: str, metrics: Sequence[str]) -> go.Figure:
    view = training.loc[
        (training["model"].astype(str) == model)
        & (training["metric"].astype(str).isin(metrics))
    ].copy()
    if view.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No training history for {model_label(model)}", showarrow=False)
        return fig
    view = view.sort_values(["metric", "epoch"])
    fig = px.line(
        view,
        x="epoch",
        y="value",
        color="metric",
        line_dash="split",
        title=model_label(model),
        labels={"value": "loss/metric value"},
    )
    return fig


def write_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    mesh_pairs: pd.DataFrame,
    mesh_split_summary: pd.DataFrame,
    mesh_diag_summary: pd.DataFrame,
    sdf_summary: pd.DataFrame,
    training: pd.DataFrame,
    volume_trends: pd.DataFrame,
    volume_velocity: pd.DataFrame,
    selected_subjects: pd.DataFrame,
    selected_volume_trends: pd.DataFrame,
    selected_volume_velocity: pd.DataFrame,
    volume_audit_tables: dict[str, pd.DataFrame],
    latent_velocity: pd.DataFrame,
    ood_rows: pd.DataFrame,
    ood_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    include_plotlyjs = args.include_plotlyjs
    tables_dir = output_dir / "tables"
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    ranking_cols = [
        "dataset_label",
        "split",
        "model_label",
        "transport_label",
        "chamfer_l2_squared_mean",
        "chamfer_l1_mean",
        "assd_mean",
        "hd95_mean",
        "volume_relative_error_mean",
        "surface_area_relative_error_mean",
    ]
    test_ranking = (
        mesh_split_summary.loc[mesh_split_summary["split"].astype(str) == "test"]
        .sort_values("chamfer_l2_squared_mean")
        if not mesh_split_summary.empty and "chamfer_l2_squared_mean" in mesh_split_summary.columns
        else pd.DataFrame()
    )
    overview_body = f"""
<section>
<h2>Summary</h2>
<p class="note">This report consolidates existing QC-large and old ADNI longitudinal results. Mesh metrics are real-mesh future forecast metrics. SDF metrics are decoder-sampled SDF metrics and are kept separate because their scale is not directly comparable to Chamfer or ASSD.</p>
</section>
<section>
<h2>Test Mesh Ranking</h2>
<div class="table-wrap">{table_html(test_ranking, ranking_cols, max_rows=30)}</div>
</section>
<section>
<h2>Generated Tables</h2>
<p class="note">CSV tables are in <code>tables/</code>. Use <code>coverage_and_missing.html</code> to see which inputs were available and which optional heavy artifacts are missing.</p>
</section>
"""
    write_page(output_dir, "index.html", "Unified Longitudinal Visual Report", overview_body)

    metric_sections = []
    for metric in MESH_METRICS:
        fig = metric_bar_figure(mesh_split_summary, metric, f"{metric} by split")
        metric_sections.append(
            f"<section><h2>{html.escape(metric)}</h2>{figure_html(fig, include_plotlyjs)}</section>"
        )
    write_page(
        output_dir,
        "pages/metrics_by_split.html",
        "Metrics by Split",
        "\n".join(metric_sections),
    )

    loss_metrics = {
        "pca150_direct_cocycle_flow": [
            "pair_total_loss",
            "pair_target_loss",
            "val_endpoint_vertex_mae",
            "val_endpoint_pca_mse",
        ],
        "qc_brainode_pca150": [
            "train_total_loss",
            "train_endpoint_pca_mse",
            "val_forward_endpoint_vertex_mae",
            "val_forward_endpoint_pca_mse",
        ],
        "qc_siren_drop_bad_min2": ["total", "real_prediction", "observed_consistency", "virtual_consistency"],
        "qc_siren_latent_ode": ["total", "real_prediction", "observed_consistency", "virtual_consistency"],
        "qc_siren_local_decomp_volume": ["total", "real_prediction", "relative_volume", "disease_volume_ordering"],
    }
    loss_sections = []
    for model, metrics in loss_metrics.items():
        fig = training_figure(training, model, metrics)
        loss_sections.append(
            f"<section><h2>{html.escape(model_label(model))}</h2>{figure_html(fig, include_plotlyjs)}</section>"
        )
    write_page(output_dir, "pages/loss_curves.html", "Training and Validation Loss", "\n".join(loss_sections))

    all_metric_body = f"""
<section>
<h2>Mesh Summary</h2>
<div class="table-wrap">{table_html(mesh_split_summary.sort_values(['dataset', 'split', 'model_priority']) if not mesh_split_summary.empty else mesh_split_summary, ranking_cols, max_rows=120)}</div>
</section>
<section>
<h2>SDF Summary</h2>
<div class="table-wrap">{table_html(sdf_summary.loc[sdf_summary.get('grouping', '').astype(str) == 'overall'] if not sdf_summary.empty and 'grouping' in sdf_summary.columns else sdf_summary, ['dataset_label', 'model_label', 'split', 'rows', *SDF_SUMMARY_METRICS], max_rows=160)}</div>
</section>
"""
    write_page(output_dir, "pages/all_metrics.html", "All Available Metrics", all_metric_body)

    volume_sections = []
    trend_subject_note = (
        "Trend figures are filtered to the selected subject manifest "
        "(up to 25 AD and 25 CN subjects per dataset/split). Whole-dataset "
        "numeric metric tables remain available in tables/."
        if not selected_subjects.empty
        else "No selected subject manifest was provided; trend figures use all available trend rows."
    )
    volume_sections.append(f"<section><p class=\"note\">{html.escape(trend_subject_note)}</p></section>")
    trend_velocity = selected_volume_velocity if not selected_volume_velocity.empty else volume_velocity
    trend_rows = selected_volume_trends if not selected_volume_trends.empty else volume_trends
    if not trend_velocity.empty:
        v_summary = summarize_numeric(
            trend_velocity,
            ["dataset", "model", "family", "split", "diagnosis", "transport_method"],
            [
                "real_volume_velocity_per_year",
                "predicted_volume_velocity_per_year",
                "volume_velocity_abs_error_per_year",
                "real_surface_area_velocity_per_year",
                "predicted_surface_area_velocity_per_year",
                "surface_area_velocity_abs_error_per_year",
            ],
        )
        melted = v_summary.melt(
            id_vars=["dataset_label", "model_label", "split", "diagnosis", "transport_label"],
            value_vars=[
                col
                for col in [
                    "real_volume_velocity_per_year_mean",
                    "predicted_volume_velocity_per_year_mean",
                ]
                if col in v_summary.columns
            ],
            var_name="velocity_type",
            value_name="velocity",
        )
        fig = px.bar(
            melted.dropna(subset=["velocity"]),
            x="model_label",
            y="velocity",
            color="velocity_type",
            facet_col="split",
            facet_row="diagnosis",
            barmode="group",
            title="Mean volume velocity from pair endpoints",
        )
        fig.update_xaxes(tickangle=35)
        volume_sections.append(f"<section><h2>Endpoint Volume Velocity</h2>{figure_html(fig, include_plotlyjs)}</section>")
        volume_sections.append(
            f"<section><h2>Volume Velocity Table</h2><div class=\"table-wrap\">{table_html(v_summary, ['dataset_label', 'model_label', 'split', 'diagnosis', 'transport_label', 'rows', 'real_volume_velocity_per_year_mean', 'predicted_volume_velocity_per_year_mean', 'volume_velocity_abs_error_per_year_mean'], max_rows=180)}</div></section>"
        )
    if not trend_rows.empty and {"age_years", "volume", "model_label"}.issubset(trend_rows.columns):
        view = trend_rows.dropna(subset=["age_years", "volume"]).copy()
        if len(view) > 8000:
            view = view.groupby(["dataset", "model", "split", "diagnosis", "transport_method"], group_keys=False).head(250)
        fig = px.line(
            view,
            x="age_years",
            y="volume",
            color="transport_label",
            facet_col="split",
            facet_row="diagnosis",
            line_group="subject_id",
            hover_data=["model_label", "subject_id"],
            title="Selected subject volume trends",
        )
        volume_sections.append(f"<section><h2>Selected Subject Volume Trends</h2>{figure_html(fig, include_plotlyjs)}</section>")
    audit_group = volume_audit_tables.get("group_volume_summary", pd.DataFrame())
    if not audit_group.empty and {"cohort", "method", "split", "subject_diagnosis", "mean_annual_percent_slope"}.issubset(audit_group.columns):
        audit_view = audit_group.loc[
            audit_group["cohort"].astype(str).isin(["large_qc", "large_all", "old_small"])
        ].copy()
        audit_view = audit_view.loc[audit_view["subject_diagnosis"].astype(str).isin(["AD", "CN"])].copy()
        fig = px.bar(
            audit_view.dropna(subset=["mean_annual_percent_slope"]),
            x="method",
            y="mean_annual_percent_slope",
            color="method",
            facet_col="split",
            facet_row="subject_diagnosis",
            barmode="group",
            hover_data=["cohort", "subjects", "scans", "median_annual_percent_slope"],
            title="Whole-dataset GT/reconstruction volume slope from completed mesh-volume audit",
        )
        volume_sections.append(
            f"<section><h2>Whole-Dataset GT/Reconstruction Volume Audit</h2>"
            f"<p class=\"note\">This uses completed mesh-volume audit CSVs and does not require mesh regeneration. It compares ground-truth mesh volume slopes with available reconstructed mesh volume slopes over the whole dataset cohorts.</p>"
            f"{figure_html(fig, include_plotlyjs)}</section>"
        )
        volume_sections.append(
            f"<section><h2>Whole-Dataset Volume Audit Table</h2><div class=\"table-wrap\">"
            f"{table_html(audit_view, ['cohort', 'method', 'split', 'subject_diagnosis', 'subjects', 'scans', 'mean_annual_percent_slope', 'median_annual_percent_slope', 'subjects_decreasing_fraction', 'mean_pair_percent_rate'], max_rows=220)}"
            f"</div></section>"
        )
    if not volume_sections:
        volume_sections.append("<section><p class=\"note\">No volume trend data available.</p></section>")
    write_page(output_dir, "pages/volume_trends.html", "Volume Trends", "\n".join(volume_sections))

    velocity_sections = []
    if not latent_velocity.empty:
        latent_summary = summarize_numeric(
            latent_velocity,
            ["dataset", "model", "family", "split", "diagnosis", "pair_type"],
            [
                "real_latent_displacement_l2_per_year",
                "predicted_latent_displacement_l2_per_year",
                "composed_latent_displacement_l2_per_year",
                "average_predicted_speed_l2_per_year",
                "source_neighbor_speed_l2_per_year",
                "target_neighbor_speed_l2_per_year",
            ],
        )
        melted = latent_summary.melt(
            id_vars=["dataset_label", "model_label", "split", "diagnosis", "pair_type"],
            value_vars=[
                col
                for col in [
                    "real_latent_displacement_l2_per_year_mean",
                    "predicted_latent_displacement_l2_per_year_mean",
                    "average_predicted_speed_l2_per_year_mean",
                ]
                if col in latent_summary.columns
            ],
            var_name="velocity_type",
            value_name="velocity",
        )
        fig = px.bar(
            melted.dropna(subset=["velocity"]),
            x="model_label",
            y="velocity",
            color="velocity_type",
            facet_col="split",
            facet_row="diagnosis",
            barmode="group",
            title="SIREN latent average velocity versus real latent displacement",
        )
        fig.update_xaxes(tickangle=35)
        velocity_sections.append(f"<section><h2>Latent Velocity</h2>{figure_html(fig, include_plotlyjs)}</section>")
        velocity_sections.append(
            f"<section><h2>Latent Velocity Table</h2><div class=\"table-wrap\">{table_html(latent_summary, ['dataset_label', 'model_label', 'split', 'diagnosis', 'pair_type', 'rows', 'real_latent_displacement_l2_per_year_mean', 'predicted_latent_displacement_l2_per_year_mean', 'average_predicted_speed_l2_per_year_mean'], max_rows=220)}</div></section>"
        )
    generated_dir = Path(args.generated_artifact_dir) if args.generated_artifact_dir else None
    inst = read_csv(generated_dir / "instantaneous_velocity.csv") if generated_dir else pd.DataFrame()
    if not inst.empty:
        inst = add_display_columns(inst)
        inst_summary = summarize_numeric(
            inst,
            ["dataset", "model", "family", "split", "diagnosis", "condition_name"],
            [
                "instantaneous_diag_l2_per_year",
                "finite_difference_diag_l2_per_year",
                "real_latent_velocity_l2_per_year",
                "instantaneous_diag_cosine_with_real",
            ],
        )
        melted = inst_summary.melt(
            id_vars=["dataset_label", "model_label", "split", "diagnosis", "condition_name"],
            value_vars=[
                col
                for col in [
                    "instantaneous_diag_l2_per_year_mean",
                    "finite_difference_diag_l2_per_year_mean",
                    "real_latent_velocity_l2_per_year_mean",
                ]
                if col in inst_summary.columns
            ],
            var_name="velocity_type",
            value_name="velocity",
        )
        fig = px.bar(
            melted.dropna(subset=["velocity"]),
            x="model_label",
            y="velocity",
            color="velocity_type",
            facet_col="split",
            facet_row="diagnosis",
            barmode="group",
            title="Instantaneous diagonal latent velocity",
        )
        fig.update_xaxes(tickangle=35)
        velocity_sections.append(f"<section><h2>Instantaneous Diagonal Velocity</h2>{figure_html(fig, include_plotlyjs)}</section>")
        velocity_sections.append(
            f"<section><h2>Instantaneous Velocity Table</h2><div class=\"table-wrap\">{table_html(inst_summary, ['dataset_label', 'model_label', 'split', 'diagnosis', 'condition_name', 'rows', 'instantaneous_diag_l2_per_year_mean', 'finite_difference_diag_l2_per_year_mean', 'real_latent_velocity_l2_per_year_mean', 'instantaneous_diag_cosine_with_real_mean'], max_rows=220)}</div></section>"
        )
    if not velocity_sections:
        velocity_sections.append("<section><p class=\"note\">No velocity data available.</p></section>")
    write_page(output_dir, "pages/velocity.html", "Velocity", "\n".join(velocity_sections))

    ood_sections = []
    if not ood_summary.empty and {"horizon_years", "predicted_volume_relative_delta_mean"}.issubset(ood_summary.columns):
        ood_summary = add_display_columns(ood_summary)
        fig = px.line(
            ood_summary.dropna(subset=["horizon_years", "predicted_volume_relative_delta_mean"]),
            x="horizon_years",
            y="predicted_volume_relative_delta_mean",
            color="rollout_condition",
            line_dash="transport_method",
            facet_col="split",
            facet_row="source_diagnosis",
            markers=True,
            title="OOD conditional future volume delta",
        )
        ood_sections.append(f"<section><h2>OOD Volume Delta</h2>{figure_html(fig, include_plotlyjs)}</section>")
        ood_sections.append(
            f"<section><h2>OOD Summary Table</h2><div class=\"table-wrap\">{table_html(ood_summary, ['dataset_label', 'model_label', 'split', 'source_diagnosis', 'rollout_condition', 'transport_method', 'horizon_years', 'rows', 'ood_fraction', 'predicted_volume_relative_delta_mean', 'predicted_volume_delta_mean'], max_rows=240)}</div></section>"
        )
    if not ood_rows.empty:
        ood_sections.append(
            f"<section><h2>OOD Row Samples</h2><div class=\"table-wrap\">{table_html(add_display_columns(ood_rows), ['dataset_label', 'model_label', 'split', 'subject_id', 'source_diagnosis', 'rollout_condition', 'transport_method', 'source_age_years', 'future_age_years', 'horizon_years', 'source_volume', 'predicted_volume'], max_rows=120)}</div></section>"
        )
    if not ood_sections:
        ood_sections.append("<section><p class=\"note\">No OOD forecast rows available. Run the artifact generator with <code>--pca-ood</code> to add 10/20-year PCA-flow forecasts.</p></section>")
    write_page(output_dir, "pages/ood_future_volume.html", "OOD Future Volume", "\n".join(ood_sections))

    rich_dir = repo_path(
        "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
        "analysis/future_mesh_rich_visualization"
    )
    generated_dir = Path(args.generated_artifact_dir) if args.generated_artifact_dir else None
    surface_manifest = read_csv(generated_dir / "surface_change_case_manifest.csv") if generated_dir else pd.DataFrame()
    selected_coverage = read_csv(generated_dir / "selected_subject_coverage.csv") if generated_dir else pd.DataFrame()
    surface_body = f"""
<section>
<h2>Selected Subject Coverage</h2>
<p class="note">These are the exact subjects used for trend visualizations and intended selected-mesh comparison. The selector keeps up to 25 AD and 25 CN subjects per dataset/split when available.</p>
<div class="table-wrap">{table_html(selected_coverage, ['dataset', 'split', 'diagnosis', 'selected_subjects', 'requested_subjects', 'selected_pairs', 'pair_source_mode'], max_rows=160)}</div>
</section>
<section>
<h2>Selected Subjects</h2>
<div class="table-wrap">{table_html(selected_subjects, ['dataset_label', 'split', 'diagnosis', 'subject_id', 'selected_rank', 'candidate_pairs', 'candidate_targets', 'max_gap_years', 'pair_source_mode'], max_rows=220)}</div>
</section>
<section>
<h2>Selected Future Pairs</h2>
<div class="table-wrap">{table_html(surface_manifest, ['dataset', 'split', 'diagnosis', 'subject_id', 'source_scan_id', 'target_scan_id', 'source_age_years', 'target_age_years', 'gap_years', 'pair_type'], max_rows=220)}</div>
</section>
<section>
<h2>Existing Rich Mesh Pages</h2>
<p class="note">Existing case-level pages are linked below. The unified surface-change batch for 25 AD/CN subjects per split is a heavy artifact and should be generated separately.</p>
<div class="table-wrap">{table_html(pd.DataFrame([{"page": str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else str(path)} for path in sorted(rich_dir.glob('*.html'))]), ['page'], max_rows=80)}</div>
</section>
"""
    write_page(output_dir, "pages/surface_change_cases.html", "Surface Change Cases", surface_body)

    coverage_body = f"""
<section>
<h2>Coverage</h2>
<div class="table-wrap">{table_html(coverage, ['artifact', 'status', 'rows', 'exists', 'path'], max_rows=120)}</div>
</section>
<section>
<h2>Notes</h2>
<p class="note">Missing heavy artifacts are expected until <code>generate_unified_longitudinal_artifacts.py</code> is run. SDF metrics and mesh metrics are intentionally separated because they use different units and surfaces.</p>
</section>
"""
    write_page(output_dir, "pages/coverage_and_missing.html", "Coverage and Missing Data", coverage_body)


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    generated_artifact_dir = repo_path(args.generated_artifact_dir) if args.generated_artifact_dir else None

    mesh_pairs = collect_mesh_pairs(args)
    mesh_split_summary, mesh_diag_summary = collect_mesh_summaries(mesh_pairs)
    sdf_summary = collect_sdf_summaries()
    training = collect_training_history()
    volume_trends = collect_volume_trends()
    volume_audit_tables = collect_volume_audit_tables()
    volume_velocity = derive_volume_pair_velocity(mesh_pairs)
    latent_velocity = collect_latent_velocity_from_sdf_pair_metrics()
    ood_rows, ood_summary = collect_ood_rows(generated_artifact_dir)
    selected_subjects = collect_selected_subjects(generated_artifact_dir)
    selected_volume_trends = filter_to_selected_subjects(volume_trends, selected_subjects)
    selected_volume_velocity = filter_to_selected_subjects(volume_velocity, selected_subjects)
    coverage = collect_coverage(
        args=args,
        mesh_pairs=mesh_pairs,
        sdf_summary=sdf_summary,
        training=training,
        volume_trends=volume_trends,
        volume_velocity=volume_velocity,
        volume_audit_tables=volume_audit_tables,
        latent_velocity=latent_velocity,
        ood_rows=ood_rows,
        generated_artifact_dir=generated_artifact_dir,
    )

    write_csv(tables_dir / "mesh_pair_metrics.csv", mesh_pairs)
    write_csv(tables_dir / "mesh_metric_split_summary.csv", mesh_split_summary)
    write_csv(tables_dir / "mesh_metric_diagnosis_pair_summary.csv", mesh_diag_summary)
    write_csv(tables_dir / "sdf_metric_summary.csv", sdf_summary)
    write_csv(tables_dir / "training_history_long.csv", training)
    write_csv(tables_dir / "volume_trends.csv", volume_trends)
    write_csv(tables_dir / "volume_pair_velocity.csv", volume_velocity)
    for name, frame in volume_audit_tables.items():
        write_csv(tables_dir / f"whole_dataset_volume_audit_{name}.csv", frame)
    write_csv(tables_dir / "selected_subject_manifest.csv", selected_subjects)
    write_csv(tables_dir / "selected_volume_trends.csv", selected_volume_trends)
    write_csv(tables_dir / "selected_volume_pair_velocity.csv", selected_volume_velocity)
    write_csv(tables_dir / "latent_average_velocity.csv", latent_velocity)
    write_csv(tables_dir / "ood_rows.csv", ood_rows)
    write_csv(tables_dir / "ood_summary.csv", ood_summary)
    write_csv(tables_dir / "coverage.csv", coverage)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "future_mesh_dir": str(repo_path(args.future_mesh_dir)),
                "pca_flow_dir": str(repo_path(args.pca_flow_dir)),
                "generated_artifact_dir": str(generated_artifact_dir) if generated_artifact_dir else "",
                "mesh_rows": int(len(mesh_pairs)),
                "sdf_summary_rows": int(len(sdf_summary)),
                "training_rows": int(len(training)),
                "volume_trend_rows": int(len(volume_trends)),
                "whole_dataset_volume_audit_scan_rows": int(
                    len(volume_audit_tables.get("scan_volumes", pd.DataFrame()))
                ),
                "whole_dataset_volume_audit_group_rows": int(
                    len(volume_audit_tables.get("group_volume_summary", pd.DataFrame()))
                ),
                "volume_velocity_rows": int(len(volume_velocity)),
                "selected_subject_rows": int(len(selected_subjects)),
                "selected_volume_trend_rows": int(len(selected_volume_trends)),
                "selected_volume_velocity_rows": int(len(selected_volume_velocity)),
                "latent_velocity_rows": int(len(latent_velocity)),
                "ood_rows": int(len(ood_rows)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_report(
        args=args,
        output_dir=output_dir,
        mesh_pairs=mesh_pairs,
        mesh_split_summary=mesh_split_summary,
        mesh_diag_summary=mesh_diag_summary,
        sdf_summary=sdf_summary,
        training=training,
        volume_trends=volume_trends,
        volume_velocity=volume_velocity,
        selected_subjects=selected_subjects,
        selected_volume_trends=selected_volume_trends,
        selected_volume_velocity=selected_volume_velocity,
        volume_audit_tables=volume_audit_tables,
        latent_velocity=latent_velocity,
        ood_rows=ood_rows,
        ood_summary=ood_summary,
        coverage=coverage,
    )
    print(f"Wrote unified report: {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
