from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
WITH_MCI_HELPER_DIR = REPO_ROOT / "examples" / "ADNI_1_L_With_MCI"
EXPERIMENT_DIR = Path(__file__).resolve().parent
ORIGINAL_MANIFEST_ROOT = (
    REPO_ROOT / "examples" / "ADNI_1_L_No_MCI" / "brainode_comparison_task1_manifest_original"
)
VELOCITY_DIR = EXPERIMENT_DIR / "analysis" / "velocity"
FORECAST_ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"

for extra_path in (REPO_ROOT, WITH_MCI_HELPER_DIR, EXPERIMENT_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import adni_original_speed_helpers as original_helpers
import adni_no_mci_longitudinal_model_helpers as model_helpers


def _display_name(title: str) -> Dict[str, str]:
    return {"title": title}


def optional_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.is_file() else None


def optional_json(path: Path) -> Optional[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _missing(paths: Sequence[Path]) -> List[str]:
    return [str(path) for path in paths if not path.exists()]


def require_paths(paths: Sequence[Path], message_prefix: str) -> None:
    missing = _missing(paths)
    if missing:
        raise FileNotFoundError(f"{message_prefix} Missing: {missing}")


@lru_cache(maxsize=1)
def load_original_manifest():
    return original_helpers.load_manifest(
        ORIGINAL_MANIFEST_ROOT,
        original_helpers.NO_MCI_PREFIX,
        name="ADNI no-MCI original",
    )


@lru_cache(maxsize=1)
def load_records_df() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENT_DIR / "metadata" / "adni_no_mci_longitudinal_records.csv")


def original_volume_speed_figures() -> List[go.Figure]:
    return original_helpers.step1_figures(load_original_manifest())


def original_longitudinal_figures() -> List[go.Figure]:
    return original_helpers.common_longitudinal_figures(
        load_original_manifest(), ["CN", "AD"], "ADNI no-MCI original"
    )


def original_local_shape_figures() -> List[go.Figure]:
    return original_helpers.local_shape_figures(
        load_original_manifest().clean_df, "ADNI no-MCI original", ["CN", "AD"]
    )


def original_hotspot_analysis() -> Dict[str, object]:
    bundle = load_original_manifest()
    hotspot_df = original_helpers.shape_hotspot_summary_table(
        bundle.clean_df, ["CN", "AD"], top_fraction=0.10, focus="inward"
    )
    similarity_df = original_helpers.pairwise_shape_similarity_table(
        bundle.clean_df, ["CN", "AD"], top_fraction=0.10, focus="inward"
    )
    group_maps = original_helpers.build_group_shape_maps(bundle.clean_df, ["CN", "AD"])
    cn_map = group_maps["CN"]
    ad_map = group_maps["AD"]
    display_vertices = 0.5 * (
        np.asarray(cn_map["display_vertices"], dtype=float)
        + np.asarray(ad_map["display_vertices"], dtype=float)
    )
    faces = np.asarray(cn_map["faces"], dtype=int)
    signed_diff = np.asarray(cn_map["signed_speed"], dtype=float) - np.asarray(
        ad_map["signed_speed"], dtype=float
    )
    abs_diff = np.asarray(cn_map["abs_speed"], dtype=float) - np.asarray(
        ad_map["abs_speed"], dtype=float
    )
    signed_fig = vertex_value_figure(
        display_vertices,
        faces,
        signed_diff,
        title="Original CN minus AD signed local shape speed",
        symmetric=True,
        colorbar_title="CN - AD normal speed / year",
    )
    abs_fig = vertex_value_figure(
        display_vertices,
        faces,
        abs_diff,
        title="Original CN minus AD absolute local shape speed",
        symmetric=True,
        colorbar_title="CN - AD |normal speed| / year",
    )
    return {
        "hotspot_summary": hotspot_df,
        "pairwise_similarity": similarity_df,
        "signed_difference_figure": signed_fig,
        "absolute_difference_figure": abs_fig,
    }


def velocity_required_paths() -> List[Path]:
    return [
        VELOCITY_DIR / "velocity_per_subject.csv",
        VELOCITY_DIR / "velocity_age_bins.csv",
        VELOCITY_DIR / "velocity_model_vs_observed.csv",
        VELOCITY_DIR / "cocycle_diagnostics.csv",
        VELOCITY_DIR / "velocity_summary.json",
        VELOCITY_DIR / "maps",
    ]


def require_velocity_outputs() -> None:
    require_paths(
        velocity_required_paths(),
        "Run evaluate_adni_no_mci_cocycle_velocity.py before opening the model-speed sections.",
    )


def forecast_required_paths() -> List[Path]:
    return [
        FORECAST_ANALYSIS_DIR / "validation" / "selected_checkpoint.json",
        FORECAST_ANALYSIS_DIR / "test_forecast_per_scan.csv",
        FORECAST_ANALYSIS_DIR / "test_forecast_per_subject.csv",
        FORECAST_ANALYSIS_DIR / "test_forecast_summary.csv",
        FORECAST_ANALYSIS_DIR / "test_forecast_summary.json",
        FORECAST_ANALYSIS_DIR / "test_forecast_meshes",
    ]


def require_forecast_outputs() -> None:
    require_paths(
        forecast_required_paths(),
        "Run validation checkpoint selection and locked test forecasting before opening the forecast notebook.",
    )


@lru_cache(maxsize=1)
def load_velocity_outputs() -> Dict[str, object]:
    require_velocity_outputs()
    return {
        "subject_df": pd.read_csv(VELOCITY_DIR / "velocity_per_subject.csv"),
        "age_bins_df": pd.read_csv(VELOCITY_DIR / "velocity_age_bins.csv"),
        "pair_df": pd.read_csv(VELOCITY_DIR / "velocity_model_vs_observed.csv"),
        "diag_df": pd.read_csv(VELOCITY_DIR / "cocycle_diagnostics.csv"),
        "summary_json": json.loads((VELOCITY_DIR / "velocity_summary.json").read_text(encoding="utf-8")),
        "records_df": load_records_df(),
        "maps_dir": VELOCITY_DIR / "maps",
    }


@lru_cache(maxsize=1)
def load_forecast_outputs() -> Dict[str, object]:
    require_forecast_outputs()
    return {
        "selected_checkpoint": json.loads(
            (FORECAST_ANALYSIS_DIR / "validation" / "selected_checkpoint.json").read_text(encoding="utf-8")
        ),
        "per_scan_df": pd.read_csv(FORECAST_ANALYSIS_DIR / "test_forecast_per_scan.csv"),
        "per_subject_df": pd.read_csv(FORECAST_ANALYSIS_DIR / "test_forecast_per_subject.csv"),
        "summary_df": pd.read_csv(FORECAST_ANALYSIS_DIR / "test_forecast_summary.csv"),
        "summary_json": json.loads(
            (FORECAST_ANALYSIS_DIR / "test_forecast_summary.json").read_text(encoding="utf-8")
        ),
        "records_df": load_records_df(),
        "mesh_dir": FORECAST_ANALYSIS_DIR / "test_forecast_meshes",
    }


def _parse_speed_map_name(path: Path) -> Tuple[str, str]:
    parts = path.stem.split("__")
    if len(parts) < 3 or parts[-1] != "speed_maps":
        raise ValueError(f"Unexpected speed-map filename: {path.name}")
    return parts[0], "__".join(parts[1:-1])


def _parse_pair_map_name(path: Path) -> Tuple[str, str, str]:
    parts = path.stem.split("__")
    if len(parts) < 5 or parts[-1] != "pair_speed_maps":
        raise ValueError(f"Unexpected pair-map filename: {path.name}")
    subject_id = parts[0]
    to_index = parts.index("to")
    source_scan_id = "__".join(parts[1:to_index])
    target_scan_id = "__".join(parts[to_index + 1 : -1])
    return subject_id, source_scan_id, target_scan_id


@lru_cache(maxsize=1)
def load_scan_map_index() -> pd.DataFrame:
    outputs = load_velocity_outputs()
    maps_dir = Path(outputs["maps_dir"])
    records_df = outputs["records_df"].copy()
    by_scan = records_df.set_index("scan_id")
    rows: List[Dict[str, object]] = []
    for path in sorted(maps_dir.glob("*__speed_maps.npz")):
        subject_id, scan_id = _parse_speed_map_name(path)
        meta = by_scan.loc[scan_id]
        with np.load(path) as data:
            rows.append(
                {
                    "subject_id": subject_id,
                    "scan_id": scan_id,
                    "diagnosis": str(meta["diagnosis"]),
                    "split": str(meta["split"]),
                    "visit_order": int(meta["visit_order"]),
                    "continuous_age_years": float(meta["continuous_age_years"]),
                    "map_path": str(path.resolve()),
                    "num_vertices": int(data["vertices"].shape[0]),
                    "num_faces": int(data["faces"].shape[0]),
                }
            )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_pair_map_index() -> pd.DataFrame:
    outputs = load_velocity_outputs()
    maps_dir = Path(outputs["maps_dir"])
    records_df = outputs["records_df"].copy()
    by_scan = records_df.set_index("scan_id")
    rows: List[Dict[str, object]] = []
    for path in sorted(maps_dir.glob("*__pair_speed_maps.npz")):
        subject_id, source_scan_id, target_scan_id = _parse_pair_map_name(path)
        source_meta = by_scan.loc[source_scan_id]
        target_meta = by_scan.loc[target_scan_id]
        rows.append(
            {
                "subject_id": subject_id,
                "source_scan_id": source_scan_id,
                "target_scan_id": target_scan_id,
                "diagnosis": str(source_meta["diagnosis"]),
                "source_age_years": float(source_meta["continuous_age_years"]),
                "target_age_years": float(target_meta["continuous_age_years"]),
                "map_path": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def vertex_value_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    title: str,
    symmetric: bool = True,
    colorbar_title: str = "value",
) -> go.Figure:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    values = np.asarray(values, dtype=float).reshape(-1)
    if symmetric:
        vmax = float(np.max(np.abs(values)))
        cmin, cmax = -vmax, vmax
        colorscale = "RdBu"
    else:
        cmin = float(np.min(values))
        cmax = float(np.max(values))
        if cmin >= 0.0:
            cmin = 0.0
        colorscale = "Viridis" if cmin >= 0.0 else "RdBu"
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=values,
                intensitymode="vertex",
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title=colorbar_title),
                lighting=dict(ambient=0.45, diffuse=0.7, specular=0.2, roughness=0.7),
                flatshading=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=760,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def _stack_scan_map_field(
    field: str,
    diagnosis: Optional[str] = None,
    scan_ids: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    index_df = load_scan_map_index()
    subset = index_df.copy()
    if diagnosis is not None:
        subset = subset.loc[subset["diagnosis"] == diagnosis]
    if scan_ids is not None:
        subset = subset.loc[subset["scan_id"].isin(list(scan_ids))]
    if subset.empty:
        raise ValueError(f"No scan maps found for field={field}, diagnosis={diagnosis}")
    arrays = []
    vertices = []
    faces = None
    for _, row in subset.iterrows():
        with np.load(row["map_path"]) as data:
            arrays.append(np.asarray(data[field], dtype=float))
            vertices.append(np.asarray(data["vertices"], dtype=float))
            if faces is None:
                faces = np.asarray(data["faces"], dtype=int)
    return {
        "values": np.stack(arrays, axis=0).mean(axis=0),
        "vertices": np.stack(vertices, axis=0).mean(axis=0),
        "faces": faces,
        "num_scans": int(len(subset)),
        "num_subjects": int(subset["subject_id"].nunique()),
    }


def model_factual_velocity_figures() -> List[go.Figure]:
    require_velocity_outputs()
    figures: List[go.Figure] = []
    for diagnosis in ("CN", "AD"):
        packed = _stack_scan_map_field("yearly_speed_factual", diagnosis=diagnosis)
        title = (
            f"Model-predicted yearly normal speed: {diagnosis} "
            f"(subjects={packed['num_subjects']}, scans={packed['num_scans']})"
        )
        figures.append(
            vertex_value_figure(
                packed["vertices"],
                packed["faces"],
                packed["values"],
                title=title,
                symmetric=True,
                colorbar_title="normal speed / year",
            )
        )
    return figures


def counterfactual_velocity_figures() -> List[go.Figure]:
    require_velocity_outputs()
    figures: List[go.Figure] = []
    overall = _stack_scan_map_field("yearly_speed_ad_minus_cn", diagnosis=None)
    figures.append(
        vertex_value_figure(
            overall["vertices"],
            overall["faces"],
            overall["values"],
            title=(
                "Counterfactual AD condition minus CN condition at matched anchors "
                f"(all scans, subjects={overall['num_subjects']})"
            ),
            symmetric=True,
            colorbar_title="counterfactual AD - CN speed / year",
        )
    )
    for diagnosis in ("CN", "AD"):
        packed = _stack_scan_map_field("yearly_speed_ad_minus_cn", diagnosis=diagnosis)
        figures.append(
            vertex_value_figure(
                packed["vertices"],
                packed["faces"],
                packed["values"],
                title=(
                    "Counterfactual AD condition minus CN condition at matched anchors "
                    f"for actual {diagnosis} scans (subjects={packed['num_subjects']})"
                ),
                symmetric=True,
                colorbar_title="counterfactual AD - CN speed / year",
            )
        )
    return figures


def model_vs_observed_volume_speed_figures() -> List[go.Figure]:
    require_velocity_outputs()
    pair_df = load_velocity_outputs()["pair_df"].copy()
    figures: List[go.Figure] = []
    for y_obs, y_model, label in (
        (
            "observed_area_weighted_rms_speed_per_year",
            "model_area_weighted_rms_speed_per_year",
            "Area-weighted RMS speed",
        ),
        (
            "observed_net_volume_rate_per_year",
            "model_net_volume_rate_per_year",
            "Net volume rate",
        ),
    ):
        fig = go.Figure()
        for diagnosis, group in pair_df.groupby("diagnosis", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=group[y_obs],
                    y=group[y_model],
                    mode="markers",
                    name=diagnosis,
                    text=group["subject_id"],
                )
            )
        low = float(min(pair_df[y_obs].min(), pair_df[y_model].min()))
        high = float(max(pair_df[y_obs].max(), pair_df[y_model].max()))
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="y=x",
                line=dict(color="black", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"Model versus observed {label}",
            xaxis_title=f"Observed {label}",
            yaxis_title=f"Model {label}",
            template="plotly_white",
        )
        figures.append(fig)
    return figures


def model_vs_observed_local_speed_figures(per_diagnosis_examples: int = 1) -> List[go.Figure]:
    require_velocity_outputs()
    pair_index = load_pair_map_index()
    figures: List[go.Figure] = []
    for diagnosis in ("CN", "AD"):
        subset = (
            pair_index.loc[pair_index["diagnosis"] == diagnosis]
            .sort_values(["source_age_years", "subject_id"])
            .head(per_diagnosis_examples)
        )
        for _, row in subset.iterrows():
            with np.load(row["map_path"]) as data:
                vertices = np.asarray(data["vertices"], dtype=float)
                faces = np.asarray(data["faces"], dtype=int)
                observed = np.asarray(data["observed_local_speed"], dtype=float)
                model = np.asarray(data["model_local_speed"], dtype=float)
                diff = np.asarray(data["local_speed_difference"], dtype=float)
            prefix = (
                f"{diagnosis} subject {row['subject_id']} "
                f"{row['source_scan_id']} -> {row['target_scan_id']}"
            )
            figures.extend(
                [
                    vertex_value_figure(
                        vertices,
                        faces,
                        observed,
                        title=f"Observed local normal speed: {prefix}",
                        symmetric=True,
                        colorbar_title="observed speed / year",
                    ),
                    vertex_value_figure(
                        vertices,
                        faces,
                        model,
                        title=f"Model local normal speed: {prefix}",
                        symmetric=True,
                        colorbar_title="model speed / year",
                    ),
                    vertex_value_figure(
                        vertices,
                        faces,
                        diff,
                        title=f"Model minus observed local normal speed: {prefix}",
                        symmetric=True,
                        colorbar_title="model - observed / year",
                    ),
                ]
            )
    return figures


def velocity_age_bin_figures() -> List[go.Figure]:
    require_velocity_outputs()
    age_bins_df = load_velocity_outputs()["age_bins_df"].copy()
    age_bins_df = age_bins_df.sort_values(["diagnosis", "age_bin_center"])
    figures: List[go.Figure] = []
    for metric, title in (
        ("area_weighted_rms_speed_per_year_mean", "Age-binned area-weighted RMS speed"),
        ("net_volume_rate_per_year_mean", "Age-binned net volume rate"),
        (
            "counterfactual_ad_minus_cn_rms_speed_per_year_mean",
            "Age-binned counterfactual AD minus CN speed",
        ),
    ):
        fig = go.Figure()
        for diagnosis, group in age_bins_df.groupby("diagnosis", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=group["age_bin_center"],
                    y=group[metric],
                    mode="lines+markers",
                    name=diagnosis,
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="Age bin center (years)",
            yaxis_title=metric.replace("_mean", ""),
            template="plotly_white",
        )
        figures.append(fig)
    return figures


def cocycle_diagnostic_figures() -> List[go.Figure]:
    require_velocity_outputs()
    diag_df = load_velocity_outputs()["diag_df"].copy()
    figures: List[go.Figure] = []
    for diagnostic_type, group in diag_df.groupby("diagnostic_type", sort=True):
        fig = go.Figure()
        for diagnosis, diag_group in group.groupby("diagnosis", sort=True):
            fig.add_trace(
                go.Box(
                    y=diag_group["latent_l2_error"],
                    name=diagnosis,
                    boxmean=True,
                )
            )
        fig.update_layout(
            title=f"{diagnostic_type.replace('_', ' ').title()} latent L2 error",
            yaxis_title="latent L2 error",
            template="plotly_white",
        )
        figures.append(fig)
    return figures


def _load_mesh(mesh_like: str | Path | trimesh.Trimesh) -> trimesh.Trimesh:
    return model_helpers.load_mesh(mesh_like)


def overlay_mesh_figure(
    mesh_entries: Sequence[Tuple[str, str | Path | trimesh.Trimesh, str, float]],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for name, mesh_like, color, opacity in mesh_entries:
        mesh = _load_mesh(mesh_like)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        width=950,
        height=760,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def load_selected_subject_manifests() -> List[Dict[str, object]]:
    outputs = load_forecast_outputs()
    mesh_dir = Path(outputs["mesh_dir"])
    manifests = []
    for path in sorted(mesh_dir.glob("*/manifest.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return manifests


def choose_forecast_example_subjects() -> Dict[str, str]:
    manifests = load_selected_subject_manifests()
    chosen: Dict[str, str] = {}
    for payload in manifests:
        diagnosis = str(payload["diagnosis"])
        if diagnosis not in chosen:
            chosen[diagnosis] = str(payload["subject_id"])
    return chosen


def baseline_example_figures() -> List[go.Figure]:
    require_forecast_outputs()
    records_df = load_records_df()
    chosen = choose_forecast_example_subjects()
    figures: List[go.Figure] = []
    for diagnosis in ("CN", "AD"):
        subject_id = chosen.get(diagnosis)
        if subject_id is None:
            continue
        baseline_row = (
            records_df.loc[records_df["subject_id"] == subject_id]
            .sort_values("visit_order")
            .iloc[0]
        )
        figures.append(
            overlay_mesh_figure(
                [
                    (
                        f"{diagnosis} baseline",
                        baseline_row["mesh_path"],
                        "#1f77b4" if diagnosis == "CN" else "#d62728",
                        1.0,
                    )
                ],
                title=(
                    f"{diagnosis} baseline observation: subject {subject_id}, "
                    f"scan {baseline_row['scan_id']}"
                ),
            )
        )
    return figures


def _forecast_rows_for_subject(
    subject_id: str,
    observation_mode: Optional[str] = None,
    method: Optional[str] = None,
) -> pd.DataFrame:
    outputs = load_forecast_outputs()
    frame = outputs["per_scan_df"].copy()
    frame = frame.loc[frame["subject_id"] == subject_id]
    if observation_mode is not None:
        frame = frame.loc[frame["observation_mode"] == observation_mode]
    if method is not None:
        frame = frame.loc[frame["method"] == method]
    return frame.sort_values(["target_visit_order", "method"]).reset_index(drop=True)


def forecast_example_figures(
    observation_mode: str,
    method: str = "direct",
) -> List[go.Figure]:
    require_forecast_outputs()
    chosen = choose_forecast_example_subjects()
    figures: List[go.Figure] = []
    for diagnosis in ("CN", "AD"):
        subject_id = chosen.get(diagnosis)
        if subject_id is None:
            continue
        frame = _forecast_rows_for_subject(subject_id, observation_mode=observation_mode, method=method)
        frame = frame.loc[frame["prediction_mesh_path"].astype(str) != ""]
        for _, row in frame.iterrows():
            predicted = row["prediction_mesh_path"]
            latest_observed = row["latest_observed_scan_id"]
            records_df = load_records_df()
            latest_row = records_df.loc[records_df["scan_id"] == latest_observed].iloc[0]
            figures.append(
                overlay_mesh_figure(
                    [
                        ("latest observed", latest_row["mesh_path"], "#7f7f7f", 0.25),
                        ("ground truth", row["target_mesh_path"], "#2ca02c", 0.50),
                        ("prediction", predicted, "#d62728", 0.40),
                    ],
                    title=(
                        f"{diagnosis} subject {subject_id}: {observation_mode} -> "
                        f"{row['target_label']} ({method})"
                    ),
                )
            )
    return figures


def direct_vs_composed_figure() -> go.Figure:
    require_forecast_outputs()
    frame = load_forecast_outputs()["per_scan_df"].copy()
    frame = frame.loc[frame["method"].isin(["direct", "composed"])]
    pivot = frame.pivot_table(
        index=["subject_id", "diagnosis", "observation_mode", "target_label", "target_scan_id"],
        columns="method",
        values="chamfer_aligned",
        aggfunc="first",
    ).reset_index()
    fig = go.Figure()
    for diagnosis, group in pivot.groupby("diagnosis", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["direct"],
                y=group["composed"],
                mode="markers",
                name=diagnosis,
                text=group["subject_id"],
            )
        )
    low = float(min(pivot["direct"].min(), pivot["composed"].min()))
    high = float(max(pivot["direct"].max(), pivot["composed"].max()))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="y=x",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(
        title="Direct versus composed forecast error",
        xaxis_title="Direct aligned Chamfer",
        yaxis_title="Composed aligned Chamfer",
        template="plotly_white",
    )
    return fig


def subject_error_trajectory_figure() -> go.Figure:
    require_forecast_outputs()
    frame = load_forecast_outputs()["per_subject_df"].copy()
    frame = frame.loc[frame["method"].isin(["direct", "composed", "no_change"])]
    fig = go.Figure()
    for method, method_group in frame.groupby("method", sort=True):
        for diagnosis, diag_group in method_group.groupby("diagnosis", sort=True):
            fig.add_trace(
                go.Box(
                    y=diag_group["mean_chamfer_aligned"],
                    name=f"{diagnosis}:{method}",
                    boxmean=True,
                )
            )
    fig.update_layout(
        title="Per-subject forecast error trajectories",
        yaxis_title="Mean aligned Chamfer per subject",
        template="plotly_white",
    )
    return fig


def cohort_summary_figure() -> go.Figure:
    require_forecast_outputs()
    summary_df = load_forecast_outputs()["summary_df"].copy()
    subset = summary_df.loc[summary_df["method"].isin(["direct", "composed", "no_change"])]
    fig = go.Figure()
    for method, group in subset.groupby("method", sort=True):
        fig.add_trace(
            go.Bar(
                x=group["cohort"] + " | " + group["observation_mode"] + " | " + group["target_label"],
                y=group["chamfer_aligned_mean"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=group["chamfer_aligned_ci_high"] - group["chamfer_aligned_mean"],
                    arrayminus=group["chamfer_aligned_mean"] - group["chamfer_aligned_ci_low"],
                ),
                name=method,
            )
        )
    fig.update_layout(
        title="Forecast cohort summary with subject-level confidence intervals",
        xaxis_title="cohort | observation mode | target",
        yaxis_title="Aligned Chamfer mean with subject bootstrap CI",
        barmode="group",
        template="plotly_white",
    )
    return fig


def no_change_comparison_figure() -> go.Figure:
    require_forecast_outputs()
    frame = load_forecast_outputs()["per_scan_df"].copy()
    frame = frame.loc[frame["method"].isin(["direct", "no_change"])]
    pivot = frame.pivot_table(
        index=["subject_id", "diagnosis", "observation_mode", "target_label", "target_scan_id"],
        columns="method",
        values="chamfer_aligned",
        aggfunc="first",
    ).reset_index()
    fig = go.Figure()
    for diagnosis, group in pivot.groupby("diagnosis", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["no_change"],
                y=group["direct"],
                mode="markers",
                name=diagnosis,
                text=group["subject_id"],
            )
        )
    low = float(min(pivot["direct"].min(), pivot["no_change"].min()))
    high = float(max(pivot["direct"].max(), pivot["no_change"].max()))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="y=x",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(
        title="Direct forecast versus no-change baseline",
        xaxis_title="No-change aligned Chamfer",
        yaxis_title="Direct aligned Chamfer",
        template="plotly_white",
    )
    return fig


def volume_comparison_figure() -> go.Figure:
    require_forecast_outputs()
    frame = load_forecast_outputs()["per_scan_df"].copy()
    frame = frame.loc[frame["method"].isin(["direct", "composed", "no_change", "task2_upper_bound"])]
    fig = go.Figure()
    for method, group in frame.groupby("method", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["target_volume"],
                y=group["pred_volume"],
                mode="markers",
                name=method,
                text=group["subject_id"],
            )
        )
    low = float(min(frame["target_volume"].min(), frame["pred_volume"].min()))
    high = float(max(frame["target_volume"].max(), frame["pred_volume"].max()))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="y=x",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(
        title="Forecast volume comparison",
        xaxis_title="Target volume",
        yaxis_title="Predicted volume",
        template="plotly_white",
    )
    return fig


@lru_cache(maxsize=1)
def _cached_forecast_bundle(device: str = "cpu"):
    checkpoint = load_forecast_outputs()["selected_checkpoint"]["selected_checkpoint"]
    return model_helpers.load_model_bundle(EXPERIMENT_DIR, checkpoint, device=device)


def local_shape_change_figures(
    method: str = "direct",
    observation_mode: str = "baseline_only",
    target_label: str = "m12",
    device: str = "cpu",
    anchor_fit_steps: int = 50,
    anchor_fit_samples: int = 512,
) -> List[go.Figure]:
    require_forecast_outputs()
    outputs = load_forecast_outputs()
    frame = outputs["per_scan_df"].copy()
    chosen = choose_forecast_example_subjects()
    figures: List[go.Figure] = []
    bundle = _cached_forecast_bundle(device)
    records_df = load_records_df()
    subject_groups = model_helpers.grouped_subject_rows(records_df)
    for diagnosis in ("CN", "AD"):
        subject_id = chosen.get(diagnosis)
        if subject_id is None:
            continue
        row = frame.loc[
            (frame["subject_id"] == subject_id)
            & (frame["method"] == method)
            & (frame["observation_mode"] == observation_mode)
            & (frame["target_label"] == target_label)
        ]
        if row.empty:
            continue
        row = row.iloc[0]
        observation_scan_ids = str(row["observation_scan_ids"]).split("|")
        anchor, _, observations = model_helpers.fit_subject_anchor(
            bundle,
            observation_scan_ids,
            seed=0,
            num_iterations=anchor_fit_steps,
            num_samples=anchor_fit_samples,
            lr=1e-2,
            init_std=1e-2,
        )
        baseline_time = float(model_helpers.anchor_baseline_time(observations))
        subject_rows = subject_groups[subject_id]
        target_meta = subject_rows.loc[subject_rows["scan_id"] == row["target_scan_id"]].iloc[0]
        current_meta = subject_rows.loc[subject_rows["scan_id"] == row["latest_observed_scan_id"]].iloc[0]
        latent_current = model_helpers.transport_direct(
            bundle,
            anchor,
            baseline_time=baseline_time,
            target_time=float(current_meta["continuous_age_norm"]),
            target_label_ad=int(current_meta["label_ad"]),
        )
        if method == "direct":
            latent_future = model_helpers.transport_direct(
                bundle,
                anchor,
                baseline_time=baseline_time,
                target_time=float(target_meta["continuous_age_norm"]),
                target_label_ad=int(target_meta["label_ad"]),
            )
        else:
            intermediate_times = [
                float(v)
                for v in subject_rows.loc[
                    (subject_rows["visit_order"] > int(current_meta["visit_order"]))
                    & (subject_rows["visit_order"] < int(target_meta["visit_order"]))
                ]["continuous_age_norm"].tolist()
            ]
            latent_future = model_helpers.transport_composed(
                bundle,
                anchor,
                baseline_time=baseline_time,
                intermediate_times=intermediate_times,
                target_time=float(target_meta["continuous_age_norm"]),
                target_label_ad=int(target_meta["label_ad"]),
            )
        current_mesh = model_helpers.load_mesh(current_meta["mesh_path"])
        vertices = np.asarray(current_mesh.vertices, dtype=np.float32)
        faces = np.asarray(current_mesh.faces, dtype=int)
        delta_years = float(row["horizon_months_from_latest_observed"]) / 12.0
        observed = model_helpers.observed_correspondence_speed(
            current_meta["mesh_path"],
            target_meta["mesh_path"],
            delta_years=delta_years,
        )
        model_speed = model_helpers.finite_step_surface_change(
            bundle,
            latent_current=latent_current,
            latent_future=latent_future,
            query_points=vertices,
        ) / max(delta_years, 1e-8)
        figures.extend(
            [
                vertex_value_figure(
                    vertices,
                    faces,
                    observed["speed"],
                    title=f"Observed local shape change: {diagnosis} subject {subject_id}",
                    symmetric=True,
                    colorbar_title="observed speed / year",
                ),
                vertex_value_figure(
                    vertices,
                    faces,
                    model_speed,
                    title=f"Predicted local shape change: {diagnosis} subject {subject_id}",
                    symmetric=True,
                    colorbar_title="predicted speed / year",
                ),
                vertex_value_figure(
                    vertices,
                    faces,
                    model_speed - observed["speed"],
                    title=f"Predicted minus observed local shape change: {diagnosis} subject {subject_id}",
                    symmetric=True,
                    colorbar_title="predicted - observed / year",
                ),
            ]
        )
    return figures


def prospective_trajectory_bundle(
    observation_mode: str = "baseline_only",
    device: str = "auto",
    num_future_steps: int = 10,
    step_years: float = 0.5,
    mesh_resolution: int = 128,
    mesh_max_batch: int = 2 ** 17,
    anchor_fit_steps: int = 50,
    anchor_fit_samples: int = 512,
) -> Dict[str, object]:
    require_forecast_outputs()
    if int(num_future_steps) < 1:
        raise ValueError("num_future_steps must be at least 1")
    if float(step_years) <= 0.0:
        raise ValueError("step_years must be positive")

    bundle = _cached_forecast_bundle(device)
    if bundle.device.type != "cuda":
        raise RuntimeError(
            "Prospective trajectory mesh decoding requires CUDA. "
            "Use device='auto' on a CUDA machine or pass device='cuda:0'."
        )

    records_df = load_records_df()
    subject_groups = model_helpers.grouped_subject_rows(records_df)
    chosen = choose_forecast_example_subjects()
    step_time_norm = float(step_years) / float(model_helpers.TRAINING_AGE_RANGE_YEARS)
    horizon_years = float(num_future_steps) * float(step_years)

    prediction_rows: List[Dict[str, object]] = []
    ground_truth_rows: List[Dict[str, object]] = []
    map_payloads: List[Dict[str, object]] = []

    for diagnosis in ("CN", "AD"):
        subject_id = chosen.get(diagnosis)
        if subject_id is None:
            continue
        subject_forecasts = _forecast_rows_for_subject(
            subject_id, observation_mode=observation_mode, method="direct"
        )
        if subject_forecasts.empty:
            continue

        reference_row = subject_forecasts.iloc[0]
        observation_scan_ids = str(reference_row["observation_scan_ids"]).split("|")
        anchor, loss_hist, observations = model_helpers.fit_subject_anchor(
            bundle,
            observation_scan_ids,
            seed=0,
            num_iterations=anchor_fit_steps,
            num_samples=anchor_fit_samples,
            lr=1e-2,
            init_std=1e-2,
        )
        baseline_time = float(model_helpers.anchor_baseline_time(observations))
        subject_rows = subject_groups[subject_id].sort_values("visit_order").reset_index(drop=True)
        current_meta = subject_rows.loc[
            subject_rows["scan_id"] == str(reference_row["latest_observed_scan_id"])
        ].iloc[0]
        current_mesh = model_helpers.load_mesh(current_meta["mesh_path"])
        current_vertices = np.asarray(current_mesh.vertices, dtype=np.float32)
        current_faces = np.asarray(current_mesh.faces, dtype=int)
        current_time = float(current_meta["continuous_age_norm"])
        current_age = float(current_meta["continuous_age_years"])
        label_ad = int(current_meta["label_ad"])

        latent_start = model_helpers.transport_direct(
            bundle,
            anchor,
            baseline_time=baseline_time,
            target_time=current_time,
            target_label_ad=label_ad,
        )
        start_speed = model_helpers.implicit_surface_normal_velocity(
            bundle,
            latent_start,
            current_time,
            label_ad,
            current_vertices,
            yearly=True,
        )
        start_summary = model_helpers.summarize_speed_map(current_mesh, start_speed)
        start_volume = model_helpers.mesh_volume(current_mesh)
        prediction_rows.append(
            {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "observation_mode": observation_mode,
                "step_index": 0,
                "years_from_start": 0.0,
                "age_years": current_age,
                "volume": start_volume,
                "volume_change_from_start": 0.0,
                "source": "observed_start",
                "anchor_fit_loss_start": float(loss_hist[0]) if loss_hist else float("nan"),
                "anchor_fit_loss_end": float(loss_hist[-1]) if loss_hist else float("nan"),
                **start_summary,
            }
        )

        final_latent = latent_start
        final_mesh = current_mesh
        final_speed = start_speed
        for step_index in range(1, int(num_future_steps) + 1):
            future_time = current_time + step_index * step_time_norm
            future_age = current_age + step_index * float(step_years)
            latent_future = model_helpers.transport_direct(
                bundle,
                anchor,
                baseline_time=baseline_time,
                target_time=future_time,
                target_label_ad=label_ad,
            )
            pred_mesh = model_helpers.decode_mesh(
                bundle,
                latent_future,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            pred_vertices = np.asarray(pred_mesh.vertices, dtype=np.float32)
            pred_speed = model_helpers.implicit_surface_normal_velocity(
                bundle,
                latent_future,
                future_time,
                label_ad,
                pred_vertices,
                yearly=True,
            )
            pred_summary = model_helpers.summarize_speed_map(pred_mesh, pred_speed)
            pred_volume = model_helpers.mesh_volume(pred_mesh)
            prediction_rows.append(
                {
                    "subject_id": subject_id,
                    "diagnosis": diagnosis,
                    "observation_mode": observation_mode,
                    "step_index": step_index,
                    "years_from_start": step_index * float(step_years),
                    "age_years": future_age,
                    "volume": pred_volume,
                    "volume_change_from_start": pred_volume - start_volume,
                    "source": "predicted_future",
                    "anchor_fit_loss_start": float(loss_hist[0]) if loss_hist else float("nan"),
                    "anchor_fit_loss_end": float(loss_hist[-1]) if loss_hist else float("nan"),
                    **pred_summary,
                }
            )
            final_latent = latent_future
            final_mesh = pred_mesh
            final_speed = pred_speed

        cumulative_change = model_helpers.finite_step_surface_change(
            bundle,
            latent_current=latent_start,
            latent_future=final_latent,
            query_points=current_vertices,
        )
        map_payloads.append(
            {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "current_age_years": current_age,
                "final_age_years": current_age + horizon_years,
                "horizon_years": horizon_years,
                "start_vertices": current_vertices,
                "start_faces": current_faces,
                "start_speed": np.asarray(start_speed, dtype=np.float32),
                "cumulative_change": np.asarray(cumulative_change, dtype=np.float32),
                "final_vertices": np.asarray(final_mesh.vertices, dtype=np.float32),
                "final_faces": np.asarray(final_mesh.faces, dtype=int),
                "final_speed": np.asarray(final_speed, dtype=np.float32),
            }
        )

        future_ground_truth = subject_rows.loc[
            subject_rows["visit_order"] >= int(current_meta["visit_order"])
        ].copy()
        for _, gt_row in future_ground_truth.iterrows():
            gt_age = float(gt_row["continuous_age_years"])
            ground_truth_rows.append(
                {
                    "subject_id": subject_id,
                    "diagnosis": diagnosis,
                    "observation_mode": observation_mode,
                    "scan_id": str(gt_row["scan_id"]),
                    "visit_order": int(gt_row["visit_order"]),
                    "years_from_start": gt_age - current_age,
                    "age_years": gt_age,
                    "volume": model_helpers.mesh_volume(gt_row["mesh_path"]),
                }
            )

    prediction_df = pd.DataFrame(prediction_rows).sort_values(
        ["diagnosis", "subject_id", "step_index"]
    ).reset_index(drop=True)
    ground_truth_df = pd.DataFrame(ground_truth_rows).sort_values(
        ["diagnosis", "subject_id", "visit_order"]
    ).reset_index(drop=True)
    summary_df = (
        prediction_df.groupby(["diagnosis", "subject_id"], sort=True)
        .agg(
            start_age_years=("age_years", "min"),
            final_age_years=("age_years", "max"),
            start_volume=("volume", "first"),
            final_volume=("volume", "last"),
            total_volume_change=("volume_change_from_start", "last"),
            start_rms_speed=("area_weighted_rms_speed_per_year", "first"),
            final_rms_speed=("area_weighted_rms_speed_per_year", "last"),
            mean_rms_speed=("area_weighted_rms_speed_per_year", "mean"),
            start_net_volume_rate=("net_volume_rate_per_year", "first"),
            final_net_volume_rate=("net_volume_rate_per_year", "last"),
            mean_net_volume_rate=("net_volume_rate_per_year", "mean"),
        )
        .reset_index()
    )

    return {
        "prediction_df": prediction_df,
        "ground_truth_df": ground_truth_df,
        "summary_df": summary_df,
        "map_payloads": map_payloads,
        "observation_mode": observation_mode,
        "num_future_steps": int(num_future_steps),
        "step_years": float(step_years),
        "horizon_years": horizon_years,
        "mesh_resolution": int(mesh_resolution),
        "device": str(bundle.device),
    }


def prospective_trajectory_figures(bundle_payload: Dict[str, object]) -> List[go.Figure]:
    prediction_df = bundle_payload["prediction_df"].copy()
    ground_truth_df = bundle_payload["ground_truth_df"].copy()
    figures: List[go.Figure] = []

    volume_fig = go.Figure()
    for diagnosis in ("CN", "AD"):
        pred_group = prediction_df.loc[prediction_df["diagnosis"] == diagnosis]
        if pred_group.empty:
            continue
        subject_id = str(pred_group.iloc[0]["subject_id"])
        volume_fig.add_trace(
            go.Scatter(
                x=pred_group["years_from_start"],
                y=pred_group["volume_change_from_start"],
                mode="lines+markers",
                name=f"{diagnosis} predicted ({subject_id})",
            )
        )
        gt_group = ground_truth_df.loc[ground_truth_df["diagnosis"] == diagnosis].copy()
        if not gt_group.empty:
            gt_start_volume = float(gt_group.iloc[0]["volume"])
            volume_fig.add_trace(
                go.Scatter(
                    x=gt_group["years_from_start"],
                    y=gt_group["volume"] - gt_start_volume,
                    mode="lines+markers",
                    name=f"{diagnosis} ground truth",
                    line=dict(dash="dot"),
                )
            )
    volume_fig.update_layout(
        title=(
            "Predicted future volume change from the given shape "
            f"({bundle_payload['num_future_steps']} steps, {bundle_payload['step_years']:.2f} years/step)"
        ),
        xaxis_title="Years from given shape",
        yaxis_title="Volume change from start",
        template="plotly_white",
    )
    figures.append(volume_fig)

    for y_col, title in (
        ("area_weighted_rms_speed_per_year", "Instantaneous yearly RMS normal speed"),
        ("net_volume_rate_per_year", "Instantaneous yearly net volume rate"),
    ):
        fig = go.Figure()
        for diagnosis in ("CN", "AD"):
            group = prediction_df.loc[prediction_df["diagnosis"] == diagnosis]
            if group.empty:
                continue
            subject_id = str(group.iloc[0]["subject_id"])
            fig.add_trace(
                go.Scatter(
                    x=group["age_years"],
                    y=group[y_col],
                    mode="lines+markers",
                    name=f"{diagnosis} ({subject_id})",
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="Age (years)",
            yaxis_title=y_col.replace("_", " "),
            template="plotly_white",
        )
        figures.append(fig)

    return figures


def prospective_trajectory_map_figures(bundle_payload: Dict[str, object]) -> List[go.Figure]:
    figures: List[go.Figure] = []
    for payload in bundle_payload["map_payloads"]:
        diagnosis = str(payload["diagnosis"])
        subject_id = str(payload["subject_id"])
        horizon_years = float(payload["horizon_years"])
        figures.append(
            vertex_value_figure(
                payload["start_vertices"],
                payload["start_faces"],
                payload["start_speed"],
                title=(
                    f"Instantaneous normal speed at the given shape: {diagnosis} subject {subject_id} "
                    f"(age {payload['current_age_years']:.1f})"
                ),
                symmetric=True,
                colorbar_title="normal speed / year",
            )
        )
        figures.append(
            vertex_value_figure(
                payload["start_vertices"],
                payload["start_faces"],
                payload["cumulative_change"],
                title=(
                    f"Predicted cumulative local change over {horizon_years:.1f} years: "
                    f"{diagnosis} subject {subject_id}"
                ),
                symmetric=True,
                colorbar_title="cumulative local change",
            )
        )
    return figures


def _age_years_to_norm(age_years: float) -> float:
    return (
        float(age_years) - float(model_helpers.TRAINING_AGE_MIN)
    ) / float(model_helpers.TRAINING_AGE_RANGE_YEARS)


def _inclusive_age_grid(
    start_age_years: float,
    final_age_years: float,
    step_years: float,
) -> List[float]:
    start_age = float(start_age_years)
    final_age = float(final_age_years)
    step = float(step_years)
    if step <= 0.0:
        raise ValueError("step_years must be positive")
    if final_age <= start_age:
        return [start_age]
    ages = [start_age]
    next_age = start_age + step
    while next_age < final_age - 1e-8:
        ages.append(next_age)
        next_age += step
    if final_age > ages[-1] + 1e-8:
        ages.append(final_age)
    return ages


def _example_subject_contexts(
    bundle,
    observation_mode: str = "baseline_only",
    anchor_fit_steps: int = 50,
    anchor_fit_samples: int = 512,
) -> Dict[str, Dict[str, object]]:
    require_forecast_outputs()
    records_df = load_records_df()
    subject_groups = model_helpers.grouped_subject_rows(records_df)
    chosen = choose_forecast_example_subjects()
    contexts: Dict[str, Dict[str, object]] = {}
    for diagnosis in ("CN", "AD"):
        subject_id = chosen.get(diagnosis)
        if subject_id is None:
            continue
        subject_forecasts = _forecast_rows_for_subject(
            subject_id,
            observation_mode=observation_mode,
            method="direct",
        )
        if subject_forecasts.empty:
            continue
        reference_row = subject_forecasts.iloc[0]
        observation_scan_ids = str(reference_row["observation_scan_ids"]).split("|")
        anchor, loss_hist, observations = model_helpers.fit_subject_anchor(
            bundle,
            observation_scan_ids,
            seed=0,
            num_iterations=anchor_fit_steps,
            num_samples=anchor_fit_samples,
            lr=1e-2,
            init_std=1e-2,
        )
        baseline_time = float(model_helpers.anchor_baseline_time(observations))
        subject_rows = subject_groups[subject_id].sort_values("visit_order").reset_index(drop=True)
        latest_meta = subject_rows.loc[
            subject_rows["scan_id"] == str(reference_row["latest_observed_scan_id"])
        ].iloc[0]
        latest_time = float(latest_meta["continuous_age_norm"])
        latest_age = float(latest_meta["continuous_age_years"])
        factual_label_ad = int(latest_meta["label_ad"])
        latest_latent = model_helpers.transport_direct(
            bundle,
            anchor,
            baseline_time=baseline_time,
            target_time=latest_time,
            target_label_ad=factual_label_ad,
        )
        contexts[diagnosis] = {
            "source_diagnosis": diagnosis,
            "subject_id": str(subject_id),
            "reference_row": reference_row,
            "observation_scan_ids": observation_scan_ids,
            "anchor": anchor,
            "baseline_time": baseline_time,
            "subject_rows": subject_rows,
            "latest_meta": latest_meta,
            "latest_time_norm": latest_time,
            "latest_age_years": latest_age,
            "factual_label_ad": factual_label_ad,
            "factual_diagnosis": "AD" if factual_label_ad == 1 else "CN",
            "latest_latent": latest_latent,
            "anchor_fit_loss_start": float(loss_hist[0]) if loss_hist else float("nan"),
            "anchor_fit_loss_end": float(loss_hist[-1]) if loss_hist else float("nan"),
        }
    return contexts


def _transport_latent_sequence(
    bundle,
    latent_start,
    start_age_years: float,
    target_ages_years: Sequence[float],
    label_ad: int,
    transport_method: str,
) -> List[torch.Tensor]:
    method_name = str(transport_method).strip().lower()
    if method_name not in {"direct", "composed"}:
        raise ValueError("transport_method must be 'direct' or 'composed'")
    if not target_ages_years:
        return []

    start_time = _age_years_to_norm(start_age_years)
    age_targets = [float(age) for age in target_ages_years]
    latents: List[torch.Tensor] = []
    current_latent = latent_start
    current_time = start_time
    for age_target in age_targets:
        target_time = _age_years_to_norm(age_target)
        if abs(target_time - start_time) <= 1e-12:
            latents.append(latent_start)
            continue
        if method_name == "direct":
            latent_target = model_helpers.transport_direct(
                bundle,
                latent_start,
                baseline_time=start_time,
                target_time=target_time,
                target_label_ad=int(label_ad),
            )
        else:
            latent_target = model_helpers.transport_direct(
                bundle,
                current_latent,
                baseline_time=current_time,
                target_time=target_time,
                target_label_ad=int(label_ad),
            )
            current_latent = latent_target
            current_time = target_time
        latents.append(latent_target)
    return latents


def _condition_name(label_ad: int) -> str:
    return "AD" if int(label_ad) == 1 else "CN"


def ood_counterfactual_bundle(
    observation_mode: str = "baseline_only",
    device: str = "auto",
    final_age_years: float = 110.0,
    trend_start_ages_years: Sequence[float] = (60.0, 70.0),
    trend_horizon_years: float = 5.0,
    trend_step_years: float = 0.5,
    composition_step_years: float = 0.5,
    mesh_resolution: int = 96,
    mesh_max_batch: int = 2 ** 17,
    anchor_fit_steps: int = 50,
    anchor_fit_samples: int = 512,
) -> Dict[str, object]:
    require_forecast_outputs()
    if float(final_age_years) <= 0.0:
        raise ValueError("final_age_years must be positive")
    if float(trend_horizon_years) <= 0.0:
        raise ValueError("trend_horizon_years must be positive")
    if float(trend_step_years) <= 0.0:
        raise ValueError("trend_step_years must be positive")
    if float(composition_step_years) <= 0.0:
        raise ValueError("composition_step_years must be positive")

    bundle = _cached_forecast_bundle(device)
    if bundle.device.type != "cuda":
        raise RuntimeError(
            "OOD mesh decoding requires CUDA. "
            "Use device='auto' on a CUDA machine or pass device='cuda:0'."
        )

    contexts = _example_subject_contexts(
        bundle,
        observation_mode=observation_mode,
        anchor_fit_steps=anchor_fit_steps,
        anchor_fit_samples=anchor_fit_samples,
    )

    velocity_rows: List[Dict[str, object]] = []
    trend_rows: List[Dict[str, object]] = []
    horizon_rows: List[Dict[str, object]] = []
    shape_payloads: List[Dict[str, object]] = []

    for source_diagnosis in ("CN", "AD"):
        context = contexts.get(source_diagnosis)
        if context is None:
            continue

        final_mesh_entries: Dict[str, List[Tuple[str, trimesh.Trimesh, str, float]]] = {
            "direct": [],
            "composed": [],
        }

        latest_latent = context["latest_latent"]
        latest_age = float(context["latest_age_years"])
        latest_time = float(context["latest_time_norm"])
        latest_meta = context["latest_meta"]
        latest_mesh_path = latest_meta["mesh_path"]
        start_mesh_for_horizon = model_helpers.load_mesh(latest_mesh_path)
        start_volume_for_horizon = model_helpers.mesh_volume(start_mesh_for_horizon)

        for label_ad in (0, 1):
            condition_name = _condition_name(label_ad)
            direct_final_latent = model_helpers.transport_direct(
                bundle,
                latest_latent,
                baseline_time=latest_time,
                target_time=_age_years_to_norm(final_age_years),
                target_label_ad=label_ad,
            )
            direct_final_mesh = model_helpers.decode_mesh(
                bundle,
                direct_final_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            final_mesh_entries["direct"].append(
                (f"{condition_name} at age {final_age_years:.0f}", direct_final_mesh, "#1f77b4" if label_ad == 0 else "#d62728", 0.68)
            )
            horizon_rows.append(
                {
                    "source_subject_diagnosis": source_diagnosis,
                    "source_subject_id": context["subject_id"],
                    "source_scan_id": str(latest_meta["scan_id"]),
                    "start_age_years": latest_age,
                    "start_volume": start_volume_for_horizon,
                    "final_age_years": float(final_age_years),
                    "condition_label_ad": int(label_ad),
                    "condition_diagnosis": condition_name,
                    "transport_method": "direct",
                    "final_volume": model_helpers.mesh_volume(direct_final_mesh),
                    "final_volume_change_from_start": model_helpers.mesh_volume(direct_final_mesh) - start_volume_for_horizon,
                    "anchor_fit_loss_end": context["anchor_fit_loss_end"],
                }
            )

            composed_age_grid = _inclusive_age_grid(
                latest_age,
                final_age_years,
                composition_step_years,
            )
            composed_latents = _transport_latent_sequence(
                bundle,
                latest_latent,
                latest_age,
                composed_age_grid,
                label_ad=label_ad,
                transport_method="composed",
            )
            composed_final_mesh = model_helpers.decode_mesh(
                bundle,
                composed_latents[-1],
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            final_mesh_entries["composed"].append(
                (f"{condition_name} at age {final_age_years:.0f}", composed_final_mesh, "#1f77b4" if label_ad == 0 else "#d62728", 0.68)
            )
            horizon_rows.append(
                {
                    "source_subject_diagnosis": source_diagnosis,
                    "source_subject_id": context["subject_id"],
                    "source_scan_id": str(latest_meta["scan_id"]),
                    "start_age_years": latest_age,
                    "start_volume": start_volume_for_horizon,
                    "final_age_years": float(final_age_years),
                    "condition_label_ad": int(label_ad),
                    "condition_diagnosis": condition_name,
                    "transport_method": "composed",
                    "final_volume": model_helpers.mesh_volume(composed_final_mesh),
                    "final_volume_change_from_start": model_helpers.mesh_volume(composed_final_mesh) - start_volume_for_horizon,
                    "anchor_fit_loss_end": context["anchor_fit_loss_end"],
                }
            )

        for transport_method, mesh_entries in final_mesh_entries.items():
            shape_payloads.append(
                {
                    "source_subject_diagnosis": source_diagnosis,
                    "source_subject_id": context["subject_id"],
                    "transport_method": transport_method,
                    "figure_title": (
                        f"{source_diagnosis} example subject {context['subject_id']}: "
                        f"age {final_age_years:.0f} OOD forecast, {transport_method}"
                    ),
                    "mesh_entries": mesh_entries,
                }
            )

        for start_age_years in [float(age) for age in trend_start_ages_years]:
            start_time_norm = _age_years_to_norm(start_age_years)
            start_latent = model_helpers.transport_direct(
                bundle,
                context["anchor"],
                baseline_time=float(context["baseline_time"]),
                target_time=start_time_norm,
                target_label_ad=int(context["factual_label_ad"]),
            )
            start_mesh = model_helpers.decode_mesh(
                bundle,
                start_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            start_vertices = np.asarray(start_mesh.vertices, dtype=np.float32)
            start_volume = model_helpers.mesh_volume(start_mesh)

            for label_ad in (0, 1):
                condition_name = _condition_name(label_ad)
                for velocity_method in ("finite_difference", "direct_diagonal"):
                    speed = model_helpers.implicit_surface_normal_velocity(
                        bundle,
                        start_latent,
                        start_time_norm,
                        label_ad,
                        start_vertices,
                        yearly=True,
                        method=velocity_method,
                    )
                    velocity_rows.append(
                        {
                            "source_subject_diagnosis": source_diagnosis,
                            "source_subject_id": context["subject_id"],
                            "start_age_years": start_age_years,
                            "start_volume": start_volume,
                            "start_shape_condition": context["factual_diagnosis"],
                            "condition_label_ad": int(label_ad),
                            "condition_diagnosis": condition_name,
                            "velocity_method": velocity_method,
                            **model_helpers.summarize_speed_map(start_mesh, speed),
                        }
                    )

                trend_ages = _inclusive_age_grid(
                    start_age_years,
                    start_age_years + float(trend_horizon_years),
                    trend_step_years,
                )
                for transport_method in ("direct", "composed"):
                    trend_latents = _transport_latent_sequence(
                        bundle,
                        start_latent,
                        start_age_years,
                        trend_ages,
                        label_ad=label_ad,
                        transport_method=transport_method,
                    )
                    for step_index, (age_years, latent_state) in enumerate(
                        zip(trend_ages, trend_latents)
                    ):
                        if step_index == 0:
                            mesh_obj = start_mesh
                            volume = start_volume
                        else:
                            mesh_obj = model_helpers.decode_mesh(
                                bundle,
                                latent_state,
                                resolution=mesh_resolution,
                                max_batch=mesh_max_batch,
                            )
                            volume = model_helpers.mesh_volume(mesh_obj)
                        trend_rows.append(
                            {
                                "source_subject_diagnosis": source_diagnosis,
                                "source_subject_id": context["subject_id"],
                                "start_age_years": start_age_years,
                                "start_shape_condition": context["factual_diagnosis"],
                                "condition_label_ad": int(label_ad),
                                "condition_diagnosis": condition_name,
                                "transport_method": transport_method,
                                "step_index": int(step_index),
                                "age_years": float(age_years),
                                "years_from_start": float(age_years - start_age_years),
                                "volume": float(volume),
                                "volume_change_from_start": float(volume - start_volume),
                            }
                        )

    velocity_df = pd.DataFrame(velocity_rows).sort_values(
        [
            "source_subject_diagnosis",
            "source_subject_id",
            "start_age_years",
            "condition_label_ad",
            "velocity_method",
        ]
    ).reset_index(drop=True)
    trend_df = pd.DataFrame(trend_rows).sort_values(
        [
            "source_subject_diagnosis",
            "source_subject_id",
            "start_age_years",
            "condition_label_ad",
            "transport_method",
            "step_index",
        ]
    ).reset_index(drop=True)
    horizon_df = pd.DataFrame(horizon_rows).sort_values(
        [
            "source_subject_diagnosis",
            "source_subject_id",
            "transport_method",
            "condition_label_ad",
        ]
    ).reset_index(drop=True)
    trend_summary_df = (
        trend_df.groupby(
            [
                "source_subject_diagnosis",
                "source_subject_id",
                "start_age_years",
                "condition_label_ad",
                "condition_diagnosis",
                "transport_method",
            ],
            sort=True,
        )
        .agg(
            start_volume=("volume", "first"),
            final_volume=("volume", "last"),
            total_volume_change=("volume_change_from_start", "last"),
        )
        .reset_index()
    )

    return {
        "velocity_df": velocity_df,
        "trend_df": trend_df,
        "trend_summary_df": trend_summary_df,
        "horizon_df": horizon_df,
        "shape_payloads": shape_payloads,
        "observation_mode": observation_mode,
        "final_age_years": float(final_age_years),
        "trend_start_ages_years": [float(age) for age in trend_start_ages_years],
        "trend_horizon_years": float(trend_horizon_years),
        "trend_step_years": float(trend_step_years),
        "composition_step_years": float(composition_step_years),
        "mesh_resolution": int(mesh_resolution),
        "device": str(bundle.device),
    }


def ood_counterfactual_shape_figures(bundle_payload: Dict[str, object]) -> List[go.Figure]:
    figures: List[go.Figure] = []
    for payload in bundle_payload["shape_payloads"]:
        figures.append(
            overlay_mesh_figure(
                payload["mesh_entries"],
                title=str(payload["figure_title"]),
            )
        )
    return figures


def ood_counterfactual_volume_figures(bundle_payload: Dict[str, object]) -> List[go.Figure]:
    trend_df = bundle_payload["trend_df"].copy()
    figures: List[go.Figure] = []
    for (source_diagnosis, subject_id, start_age_years), group in trend_df.groupby(
        ["source_subject_diagnosis", "source_subject_id", "start_age_years"],
        sort=True,
    ):
        fig = go.Figure()
        for (condition_diagnosis, transport_method), line_group in group.groupby(
            ["condition_diagnosis", "transport_method"],
            sort=True,
        ):
            fig.add_trace(
                go.Scatter(
                    x=line_group["age_years"],
                    y=line_group["volume"],
                    mode="lines+markers",
                    name=f"{condition_diagnosis} | {transport_method}",
                    line=dict(dash="solid" if transport_method == "direct" else "dot"),
                )
            )
        fig.update_layout(
            title=(
                f"{source_diagnosis} example subject {subject_id}: "
                f"volume trend from age {start_age_years:.0f} for {bundle_payload['trend_horizon_years']:.1f} years"
            ),
            xaxis_title="Age (years)",
            yaxis_title="Volume",
            template="plotly_white",
        )
        figures.append(fig)
    return figures
