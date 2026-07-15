from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots


NO_MCI_PREFIX = "adni_no_mci_left_original"
WITH_MCI_PREFIX = "adni_with_mci_left_original"
COLORS = {"CN": "#1f77b4", "MCI": "#ff7f0e", "AD": "#d62728"}


@dataclass
class ManifestBundle:
    name: str
    root: Path
    prefix: str
    clean_df: pd.DataFrame
    summary: Dict[str, object]
    trajectories: Dict[str, object]


_MESH_CACHE: Dict[str, trimesh.Trimesh] = {}
_VOLUME_CACHE: Dict[str, float] = {}


def _read_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    path = str(mesh_path)
    mesh = _MESH_CACHE.get(path)
    if mesh is None:
        loaded = trimesh.load(path, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected a mesh at {path}, got {type(loaded)!r}")
        mesh = loaded
        _MESH_CACHE[path] = mesh
    return mesh


def mesh_volume(mesh_path: str | Path) -> float:
    path = str(mesh_path)
    cached = _VOLUME_CACHE.get(path)
    if cached is None:
        cached = float(abs(_read_mesh(path).volume))
        _VOLUME_CACHE[path] = cached
    return cached


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return f"rgba(0, 0, 0, {alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def manifest_file(root: Path, prefix: str, kind: str) -> Path:
    return root / "metadata" / f"{prefix}_{kind}.csv"


def assert_manifest_ready(root: Path, prefix: str) -> None:
    required = [
        root / "metadata" / f"{prefix}_clean.csv",
        root / "metadata" / f"{prefix}_subject_trajectories.json",
        root / "metadata" / "split_summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing manifest artifacts under {root}. Missing: {missing}"
        )


def load_manifest(root: str | Path, prefix: str, name: Optional[str] = None) -> ManifestBundle:
    root = Path(root).resolve()
    assert_manifest_ready(root, prefix)
    clean_df = pd.read_csv(root / "metadata" / f"{prefix}_clean.csv")
    summary = json.loads((root / "metadata" / "split_summary.json").read_text())
    trajectories = json.loads(
        (root / "metadata" / f"{prefix}_subject_trajectories.json").read_text()
    )
    clean_df["visit_order"] = clean_df["visit_order"].astype(int)
    clean_df["visit_month_from_label"] = clean_df["visit_month_from_label"].astype(int)
    clean_df["age_years"] = clean_df["age_years"].astype(float)
    clean_df["age_norm"] = clean_df["age_norm"].astype(float)
    clean_df["months_from_baseline"] = clean_df["months_from_baseline"].astype(float)
    if "label_dx" in clean_df.columns:
        clean_df["label_dx"] = clean_df["label_dx"].astype(int)
    else:
        clean_df["label_dx"] = clean_df["diagnosis"].map({"CN": 0, "MCI": 1, "AD": 2}).astype(int)
    return ManifestBundle(
        name=name or prefix,
        root=root,
        prefix=prefix,
        clean_df=clean_df.sort_values(["subject_id", "visit_order"]).reset_index(drop=True),
        summary=summary,
        trajectories=trajectories,
    )


def build_adjacent_pair_dataframe(clean_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for subject_id, group in clean_df.groupby("subject_id", sort=True):
        group = group.sort_values("visit_order")
        rows = list(group.to_dict("records"))
        for source, target in zip(rows[:-1], rows[1:]):
            delta_months = float(target["months_from_baseline"]) - float(source["months_from_baseline"])
            if delta_months <= 0:
                continue
            delta_years = delta_months / 12.0
            source_volume = mesh_volume(source["mesh_path"])
            target_volume = mesh_volume(target["mesh_path"])
            delta_volume = target_volume - source_volume
            records.append(
                {
                    "dataset_name": "",
                    "subject_id": subject_id,
                    "split": source["split"],
                    "diagnosis": source["diagnosis"],
                    "label_dx": int(source["label_dx"]),
                    "source_scan_id": source["scan_id"],
                    "target_scan_id": target["scan_id"],
                    "source_mesh_path": source["mesh_path"],
                    "target_mesh_path": target["mesh_path"],
                    "source_visit_order": int(source["visit_order"]),
                    "target_visit_order": int(target["visit_order"]),
                    "source_months_from_baseline": float(source["months_from_baseline"]),
                    "target_months_from_baseline": float(target["months_from_baseline"]),
                    "midpoint_months_from_baseline": 0.5
                    * (
                        float(source["months_from_baseline"])
                        + float(target["months_from_baseline"])
                    ),
                    "source_age_years": float(source["age_years"]),
                    "target_age_years": float(target["age_years"]),
                    "midpoint_age_years": 0.5
                    * (float(source["age_years"]) + float(target["age_years"])),
                    "delta_months": delta_months,
                    "delta_years": delta_years,
                    "source_volume": source_volume,
                    "target_volume": target_volume,
                    "delta_volume": delta_volume,
                    "signed_volume_rate": delta_volume / delta_years,
                    "atrophy_volume_rate": (source_volume - target_volume) / delta_years,
                    "atrophy_pct_per_year": 100.0
                    * np.log(source_volume / target_volume)
                    / delta_years,
                }
            )
    return pd.DataFrame.from_records(records)


def build_visit_dataframe(clean_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for subject_id, group in clean_df.groupby("subject_id", sort=True):
        group = group.sort_values("visit_order")
        rows = list(group.to_dict("records"))
        baseline_volume = mesh_volume(rows[0]["mesh_path"])
        for row in rows:
            volume = mesh_volume(row["mesh_path"])
            records.append(
                {
                    "subject_id": subject_id,
                    "split": row["split"],
                    "diagnosis": row["diagnosis"],
                    "label_dx": int(row["label_dx"]),
                    "scan_id": row["scan_id"],
                    "visit_order": int(row["visit_order"]),
                    "visit_month_from_label": int(row["visit_month_from_label"]),
                    "months_from_baseline": float(row["months_from_baseline"]),
                    "age_years": float(row["age_years"]),
                    "volume": volume,
                    "baseline_volume": baseline_volume,
                    "relative_volume_to_baseline": volume / baseline_volume,
                    "pct_change_from_baseline": 100.0 * (volume - baseline_volume) / baseline_volume,
                    "atrophy_pct_from_baseline": 100.0 * (baseline_volume - volume) / baseline_volume,
                }
            )
    return pd.DataFrame.from_records(records)


def subject_mean_speed_dataframe(pair_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "signed_volume_rate",
        "atrophy_volume_rate",
        "atrophy_pct_per_year",
        "delta_years",
        "midpoint_age_years",
    ]
    agg = pair_df.groupby("subject_id", sort=True)[metrics].mean().reset_index()
    meta = (
        pair_df.groupby("subject_id", sort=True)[["diagnosis", "split", "label_dx"]]
        .first()
        .reset_index()
    )
    return meta.merge(agg, on="subject_id", how="inner")


def longitudinal_time_summary(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (diagnosis, x_value), group in frame.groupby(["diagnosis", x_col], sort=True):
        values = group[y_col].to_numpy(dtype=float)
        count = len(values)
        mean = float(values.mean())
        if count > 1:
            sem = float(values.std(ddof=1) / np.sqrt(count))
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem
        else:
            sem = 0.0
            ci_low = mean
            ci_high = mean
        rows.append(
            {
                "diagnosis": diagnosis,
                x_col: float(x_value),
                "mean": mean,
                "sem": sem,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "count": count,
            }
        )
    return pd.DataFrame(rows).sort_values(["diagnosis", x_col])


def age_bin_summary(
    pair_df: pd.DataFrame,
    metric: str = "atrophy_pct_per_year",
    bin_width_years: float = 2.0,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=["diagnosis", "age_bin_center", metric, "count"])
    ages = pair_df["midpoint_age_years"].to_numpy(dtype=float)
    start = np.floor(ages.min() / bin_width_years) * bin_width_years
    stop = np.ceil(ages.max() / bin_width_years) * bin_width_years + bin_width_years
    bins = np.arange(start, stop + 1e-8, bin_width_years)
    frame = pair_df.copy()
    frame["age_bin"] = pd.cut(
        frame["midpoint_age_years"], bins=bins, include_lowest=True, right=False
    )
    rows: List[Dict[str, object]] = []
    for (diagnosis, age_bin), group in frame.groupby(["diagnosis", "age_bin"], observed=True):
        if len(group) == 0:
            continue
        interval = age_bin
        center = 0.5 * (float(interval.left) + float(interval.right))
        rows.append(
            {
                "diagnosis": diagnosis,
                "age_bin_center": center,
                metric: float(group[metric].mean()),
                "count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["diagnosis", "age_bin_center"])


def make_speed_distribution_figure(
    subject_df: pd.DataFrame,
    title: str,
    diagnoses: Sequence[str],
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Subject mean atrophy percent per year",
            "Subject mean signed volume rate",
        ),
    )
    for diagnosis in diagnoses:
        group = subject_df[subject_df["diagnosis"] == diagnosis]
        if group.empty:
            continue
        color = COLORS.get(diagnosis, "#7f7f7f")
        fig.add_trace(
            go.Box(
                y=group["atrophy_pct_per_year"],
                name=diagnosis,
                marker_color=color,
                boxmean=True,
                jitter=0.25,
                pointpos=0,
                boxpoints="all",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(
                y=group["signed_volume_rate"],
                name=diagnosis,
                marker_color=color,
                boxmean=True,
                jitter=0.25,
                pointpos=0,
                boxpoints="all",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="%/year", row=1, col=1)
    fig.update_yaxes(title_text="scaled volume/year", row=1, col=2)
    fig.update_layout(height=520, width=1100, title=title, template="plotly_white")
    return fig


def make_speed_age_figure(
    pair_df: pd.DataFrame,
    title: str,
    diagnoses: Sequence[str],
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Adjacent-pair atrophy percent vs midpoint age",
            "Age-binned mean atrophy percent",
        ),
    )
    summary = age_bin_summary(pair_df, metric="atrophy_pct_per_year", bin_width_years=2.0)
    for diagnosis in diagnoses:
        group = pair_df[pair_df["diagnosis"] == diagnosis].sort_values("midpoint_age_years")
        if group.empty:
            continue
        color = COLORS.get(diagnosis, "#7f7f7f")
        fig.add_trace(
            go.Scatter(
                x=group["midpoint_age_years"],
                y=group["atrophy_pct_per_year"],
                mode="markers",
                name=diagnosis,
                marker=dict(color=color, size=6, opacity=0.55),
                legendgroup=diagnosis,
            ),
            row=1,
            col=1,
        )
        binned = summary[summary["diagnosis"] == diagnosis]
        fig.add_trace(
            go.Scatter(
                x=binned["age_bin_center"],
                y=binned["atrophy_pct_per_year"],
                mode="lines+markers",
                name=f"{diagnosis} mean",
                marker=dict(color=color, size=8),
                line=dict(color=color, width=3),
                legendgroup=diagnosis,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="midpoint age (years)", row=1, col=1)
    fig.update_xaxes(title_text="age bin center (years)", row=1, col=2)
    fig.update_yaxes(title_text="%/year", row=1, col=1)
    fig.update_yaxes(title_text="%/year", row=1, col=2)
    fig.update_layout(height=520, width=1200, title=title, template="plotly_white")
    return fig


def make_subject_trajectory_figure(
    visit_df: pd.DataFrame,
    diagnoses: Sequence[str],
    y_col: str,
    x_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=len(diagnoses),
        subplot_titles=tuple(f"{diagnosis}" for diagnosis in diagnoses),
        shared_yaxes=True,
    )
    summary = longitudinal_time_summary(visit_df, x_col=x_col, y_col=y_col)
    for idx, diagnosis in enumerate(diagnoses, start=1):
        group = visit_df[visit_df["diagnosis"] == diagnosis].sort_values(
            ["subject_id", x_col]
        )
        if group.empty:
            continue
        color = COLORS.get(diagnosis, "#7f7f7f")
        for _, subject_group in group.groupby("subject_id", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=subject_group[x_col],
                    y=subject_group[y_col],
                    mode="lines+markers",
                    line=dict(color=color, width=1),
                    marker=dict(size=4, color=color),
                    opacity=0.18,
                    hovertemplate=(
                        "subject=%{customdata[0]}<br>"
                        + f"{x_col}=%{{x}}<br>{y_col}=%{{y:.4f}}<extra></extra>"
                    ),
                    customdata=subject_group[["subject_id"]].to_numpy(),
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
        diag_summary = summary[summary["diagnosis"] == diagnosis]
        fig.add_trace(
            go.Scatter(
                x=diag_summary[x_col],
                y=diag_summary["mean"],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=8, color=color),
                name=f"{diagnosis} mean",
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text=x_col.replace("_", " "), row=1, col=idx)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(height=520, width=420 * max(1, len(diagnoses)), title=title, template="plotly_white")
    return fig


def make_mean_trajectory_band_figure(
    visit_df: pd.DataFrame,
    diagnoses: Sequence[str],
    y_col: str,
    x_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    summary = longitudinal_time_summary(visit_df, x_col=x_col, y_col=y_col)
    for diagnosis in diagnoses:
        diag_summary = summary[summary["diagnosis"] == diagnosis].sort_values(x_col)
        if diag_summary.empty:
            continue
        color = COLORS.get(diagnosis, "#7f7f7f")
        x_vals = diag_summary[x_col].to_numpy(dtype=float)
        low = diag_summary["ci_low"].to_numpy(dtype=float)
        high = diag_summary["ci_high"].to_numpy(dtype=float)
        mean = diag_summary["mean"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_vals, x_vals[::-1]]),
                y=np.concatenate([high, low[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.16),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=mean,
                mode="lines+markers",
                name=diagnosis,
                line=dict(color=color, width=3),
                marker=dict(color=color, size=8),
            )
        )
    fig.update_xaxes(title_text=x_col.replace("_", " "))
    fig.update_yaxes(title_text=y_title)
    fig.update_layout(height=520, width=1000, title=title, template="plotly_white")
    return fig


def make_subject_speed_trajectory_figure(
    pair_df: pd.DataFrame,
    diagnoses: Sequence[str],
    x_col: str,
    y_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=len(diagnoses),
        subplot_titles=tuple(f"{diagnosis}" for diagnosis in diagnoses),
        shared_yaxes=True,
    )
    summary = longitudinal_time_summary(pair_df, x_col=x_col, y_col=y_col)
    for idx, diagnosis in enumerate(diagnoses, start=1):
        group = pair_df[pair_df["diagnosis"] == diagnosis].sort_values(["subject_id", x_col])
        if group.empty:
            continue
        color = COLORS.get(diagnosis, "#7f7f7f")
        for _, subject_group in group.groupby("subject_id", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=subject_group[x_col],
                    y=subject_group[y_col],
                    mode="lines+markers",
                    line=dict(color=color, width=1),
                    marker=dict(size=5, color=color),
                    opacity=0.2,
                    hovertemplate=(
                        "subject=%{customdata[0]}<br>"
                        + f"{x_col}=%{{x}}<br>{y_col}=%{{y:.4f}}<extra></extra>"
                    ),
                    customdata=subject_group[["subject_id"]].to_numpy(),
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
        diag_summary = summary[summary["diagnosis"] == diagnosis]
        fig.add_trace(
            go.Scatter(
                x=diag_summary[x_col],
                y=diag_summary["mean"],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=8, color=color),
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text=x_col.replace("_", " "), row=1, col=idx)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(height=520, width=420 * max(1, len(diagnoses)), title=title, template="plotly_white")
    return fig


def make_baseline_age_vs_speed_figure(
    visit_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    diagnoses: Sequence[str],
    title: str,
) -> go.Figure:
    baseline = (
        visit_df.sort_values(["subject_id", "visit_order"])
        .groupby("subject_id", sort=True)[["age_years", "diagnosis"]]
        .first()
        .reset_index()
        .rename(columns={"age_years": "baseline_age_years"})
    )
    merged = baseline.merge(subject_df, on=["subject_id", "diagnosis"], how="inner")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Baseline age vs mean atrophy %/year", "Baseline age vs mean signed volume rate"),
    )
    for diagnosis in diagnoses:
        group = merged[merged["diagnosis"] == diagnosis]
        if group.empty:
            continue
        color = COLORS.get(diagnosis, "#7f7f7f")
        fig.add_trace(
            go.Scatter(
                x=group["baseline_age_years"],
                y=group["atrophy_pct_per_year"],
                mode="markers",
                name=diagnosis,
                marker=dict(color=color, size=7, opacity=0.75),
                legendgroup=diagnosis,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["baseline_age_years"],
                y=group["signed_volume_rate"],
                mode="markers",
                name=diagnosis,
                marker=dict(color=color, size=7, opacity=0.75),
                legendgroup=diagnosis,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="baseline age (years)", row=1, col=1)
    fig.update_xaxes(title_text="baseline age (years)", row=1, col=2)
    fig.update_yaxes(title_text="%/year", row=1, col=1)
    fig.update_yaxes(title_text="scaled volume/year", row=1, col=2)
    fig.update_layout(height=520, width=1100, title=title, template="plotly_white")
    return fig


def common_longitudinal_figures(
    bundle: ManifestBundle,
    diagnoses: Sequence[str],
    label: str,
) -> List[go.Figure]:
    visit_df = build_visit_dataframe(bundle.clean_df)
    pair_df = build_adjacent_pair_dataframe(bundle.clean_df)
    subject_df = subject_mean_speed_dataframe(pair_df)
    visit_df = visit_df[visit_df["diagnosis"].isin(diagnoses)].copy()
    pair_df = pair_df[pair_df["diagnosis"].isin(diagnoses)].copy()
    subject_df = subject_df[subject_df["diagnosis"].isin(diagnoses)].copy()
    return [
        make_subject_trajectory_figure(
            visit_df,
            diagnoses=diagnoses,
            y_col="volume",
            x_col="months_from_baseline",
            title=f"{bundle.name}: individual and mean raw-volume trajectories ({label})",
            y_title="scaled volume",
        ),
        make_subject_trajectory_figure(
            visit_df,
            diagnoses=diagnoses,
            y_col="atrophy_pct_from_baseline",
            x_col="months_from_baseline",
            title=f"{bundle.name}: individual and mean atrophy-from-baseline trajectories ({label})",
            y_title="atrophy from baseline (%)",
        ),
        make_mean_trajectory_band_figure(
            visit_df,
            diagnoses=diagnoses,
            y_col="volume",
            x_col="months_from_baseline",
            title=f"{bundle.name}: cohort mean raw-volume trajectories with 95% CI ({label})",
            y_title="scaled volume",
        ),
        make_mean_trajectory_band_figure(
            visit_df,
            diagnoses=diagnoses,
            y_col="atrophy_pct_from_baseline",
            x_col="months_from_baseline",
            title=f"{bundle.name}: cohort mean atrophy-from-baseline trajectories with 95% CI ({label})",
            y_title="atrophy from baseline (%)",
        ),
        make_subject_speed_trajectory_figure(
            pair_df,
            diagnoses=diagnoses,
            x_col="midpoint_months_from_baseline",
            y_col="atrophy_pct_per_year",
            title=f"{bundle.name}: individual and mean annualized speed trajectories by interval time ({label})",
            y_title="atrophy speed (%/year)",
        ),
        make_subject_speed_trajectory_figure(
            pair_df,
            diagnoses=diagnoses,
            x_col="midpoint_age_years",
            y_col="atrophy_pct_per_year",
            title=f"{bundle.name}: individual and mean annualized speed trajectories by age ({label})",
            y_title="atrophy speed (%/year)",
        ),
        make_baseline_age_vs_speed_figure(
            visit_df,
            subject_df,
            diagnoses=diagnoses,
            title=f"{bundle.name}: baseline age vs subject mean speed ({label})",
        ),
    ]


def kabsch_align_points(target_vertices: np.ndarray, source_vertices: np.ndarray) -> np.ndarray:
    source_center = source_vertices.mean(axis=0)
    target_center = target_vertices.mean(axis=0)
    source0 = source_vertices - source_center
    target0 = target_vertices - target_center
    covariance = target0.T @ source0
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return target0 @ rotation + source_center


def vertex_area_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    weights = np.zeros(len(vertices), dtype=float)
    share = face_areas / 3.0
    np.add.at(weights, faces[:, 0], share)
    np.add.at(weights, faces[:, 1], share)
    np.add.at(weights, faces[:, 2], share)
    return weights


def _subject_local_speed_records(group: pd.DataFrame) -> Optional[Dict[str, object]]:
    group = group.sort_values("visit_order")
    rows = list(group.to_dict("records"))
    signed_maps: List[np.ndarray] = []
    abs_maps: List[np.ndarray] = []
    display_vertices: List[np.ndarray] = []
    area_weights: List[np.ndarray] = []
    faces_reference: Optional[np.ndarray] = None
    pair_count = 0
    for source, target in zip(rows[:-1], rows[1:]):
        delta_months = float(target["months_from_baseline"]) - float(source["months_from_baseline"])
        if delta_months <= 0:
            continue
        delta_years = delta_months / 12.0
        source_mesh = _read_mesh(source["mesh_path"])
        target_mesh = _read_mesh(target["mesh_path"])
        source_vertices = np.asarray(source_mesh.vertices, dtype=float)
        target_vertices = np.asarray(target_mesh.vertices, dtype=float)
        if source_vertices.shape != target_vertices.shape:
            raise ValueError(
                f"Topology mismatch for subject {source['subject_id']}: "
                f"{source['scan_id']} vs {target['scan_id']}"
            )
        if faces_reference is None:
            faces_reference = np.asarray(source_mesh.faces, dtype=int)
        elif not np.array_equal(faces_reference, np.asarray(source_mesh.faces, dtype=int)):
            raise ValueError(f"Face topology changed within subject {source['subject_id']}")
        aligned_target = kabsch_align_points(target_vertices, source_vertices)
        displacement = aligned_target - source_vertices
        normals = np.asarray(source_mesh.vertex_normals, dtype=float)
        signed = np.einsum("ij,ij->i", displacement, normals) / delta_years
        abs_signed = np.abs(signed)
        signed_maps.append(signed)
        abs_maps.append(abs_signed)
        display_vertices.append(source_vertices)
        area_weights.append(vertex_area_weights(source_vertices, faces_reference))
        pair_count += 1
    if pair_count == 0 or faces_reference is None:
        return None
    return {
        "subject_id": rows[0]["subject_id"],
        "diagnosis": rows[0]["diagnosis"],
        "split": rows[0]["split"],
        "signed_speed": np.mean(np.stack(signed_maps, axis=0), axis=0),
        "abs_speed": np.mean(np.stack(abs_maps, axis=0), axis=0),
        "display_vertices": np.mean(np.stack(display_vertices, axis=0), axis=0),
        "area_weights": np.mean(np.stack(area_weights, axis=0), axis=0),
        "faces": faces_reference,
        "pair_count": pair_count,
    }


def build_group_shape_maps(
    clean_df: pd.DataFrame,
    diagnoses: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    subject_records: Dict[str, List[Dict[str, object]]] = {diag: [] for diag in diagnoses}
    for _, group in clean_df.groupby("subject_id", sort=True):
        record = _subject_local_speed_records(group)
        if record is None:
            continue
        diagnosis = str(record["diagnosis"])
        if diagnosis in subject_records:
            subject_records[diagnosis].append(record)

    result: Dict[str, Dict[str, object]] = {}
    for diagnosis, records in subject_records.items():
        if not records:
            continue
        signed = np.stack([r["signed_speed"] for r in records], axis=0)
        abs_speed = np.stack([r["abs_speed"] for r in records], axis=0)
        vertices = np.stack([r["display_vertices"] for r in records], axis=0)
        weights = np.stack([r["area_weights"] for r in records], axis=0)
        faces = records[0]["faces"]
        result[diagnosis] = {
            "diagnosis": diagnosis,
            "signed_speed": signed.mean(axis=0),
            "abs_speed": abs_speed.mean(axis=0),
            "display_vertices": vertices.mean(axis=0),
            "area_weights": weights.mean(axis=0),
            "faces": faces,
            "subject_count": len(records),
            "pair_count": int(sum(int(r["pair_count"]) for r in records)),
        }
    return result


def make_shape_speed_figure(
    group_map: Dict[str, object],
    metric: str,
    title: str,
) -> go.Figure:
    vertices = np.asarray(group_map["display_vertices"], dtype=float)
    faces = np.asarray(group_map["faces"], dtype=int)
    intensity = np.asarray(group_map[metric], dtype=float)
    if metric == "signed_speed":
        vmax = float(np.max(np.abs(intensity)))
        cmin, cmax = -vmax, vmax
        colorscale = "RdBu"
        colorbar_title = "normal speed / year"
    else:
        if float(np.min(intensity)) < 0.0:
            vmax = float(np.max(np.abs(intensity)))
            cmin, cmax = -vmax, vmax
            colorscale = "RdBu"
            colorbar_title = "difference in |normal speed| / year"
        else:
            vmax = float(np.max(intensity))
            cmin, cmax = 0.0, vmax
            colorscale = "Viridis"
            colorbar_title = "|normal speed| / year"
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=intensity,
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


def local_shape_figures(
    clean_df: pd.DataFrame,
    dataset_label: str,
    diagnoses: Sequence[str],
) -> List[go.Figure]:
    figures: List[go.Figure] = []
    group_maps = build_group_shape_maps(clean_df, diagnoses=diagnoses)
    for diagnosis in diagnoses:
        group_map = group_maps.get(diagnosis)
        if group_map is None:
            continue
        count_text = (
            f"{dataset_label}: {diagnosis} signed local shape speed "
            f"(subjects={group_map['subject_count']}, adjacent pairs={group_map['pair_count']})"
        )
        figures.append(make_shape_speed_figure(group_map, "signed_speed", count_text))
        hot_text = (
            f"{dataset_label}: {diagnosis} absolute local shape speed "
            f"(subjects={group_map['subject_count']}, adjacent pairs={group_map['pair_count']})"
        )
        figures.append(make_shape_speed_figure(group_map, "abs_speed", hot_text))
    return figures


def _weighted_top_fraction_mask(
    values: np.ndarray,
    weights: np.ndarray,
    top_fraction: float,
) -> np.ndarray:
    order = np.argsort(values)[::-1]
    sorted_weights = weights[order]
    total_weight = float(sorted_weights.sum())
    if total_weight <= 0:
        return np.zeros_like(values, dtype=bool)
    target_weight = max(1e-12, float(top_fraction) * total_weight)
    cumulative = np.cumsum(sorted_weights)
    keep_count = int(np.searchsorted(cumulative, target_weight, side="left")) + 1
    keep_indices = order[:keep_count]
    mask = np.zeros_like(values, dtype=bool)
    mask[keep_indices] = True
    return mask


def _focus_values_from_group_map(
    group_map: Dict[str, object],
    focus: str,
) -> np.ndarray:
    signed_speed = np.asarray(group_map["signed_speed"], dtype=float)
    abs_speed = np.asarray(group_map["abs_speed"], dtype=float)
    if focus == "abs":
        return abs_speed
    if focus == "inward":
        return np.clip(-signed_speed, 0.0, None)
    if focus == "outward":
        return np.clip(signed_speed, 0.0, None)
    raise ValueError(f"Unsupported focus: {focus}")


def shape_hotspot_summary_table(
    clean_df: pd.DataFrame,
    diagnoses: Sequence[str],
    top_fraction: float = 0.10,
    focus: str = "inward",
) -> pd.DataFrame:
    group_maps = build_group_shape_maps(clean_df, diagnoses=diagnoses)
    rows: List[Dict[str, object]] = []
    for diagnosis in diagnoses:
        group_map = group_maps.get(diagnosis)
        if group_map is None:
            continue
        signed_speed = np.asarray(group_map["signed_speed"], dtype=float)
        focus_values = _focus_values_from_group_map(group_map, focus)
        vertices = np.asarray(group_map["display_vertices"], dtype=float)
        weights = np.asarray(group_map["area_weights"], dtype=float)
        hotspot_mask = _weighted_top_fraction_mask(focus_values, weights, top_fraction)
        hotspot_weights = weights[hotspot_mask]
        hotspot_vertices = vertices[hotspot_mask]
        centroid = np.average(hotspot_vertices, axis=0, weights=hotspot_weights)
        weighted_mean_focus = float(np.average(focus_values, weights=weights))
        weighted_signed = float(np.average(signed_speed, weights=weights))
        rows.append(
            {
                "diagnosis": diagnosis,
                "analysis_focus": focus,
                "subject_count": int(group_map["subject_count"]),
                "adjacent_pair_count": int(group_map["pair_count"]),
                "weighted_mean_focus_speed": weighted_mean_focus,
                "weighted_mean_signed_speed": weighted_signed,
                "peak_focus_speed": float(focus_values.max()),
                "hotspot_top_fraction": float(top_fraction),
                "hotspot_vertex_count": int(hotspot_mask.sum()),
                "hotspot_centroid_x": float(centroid[0]),
                "hotspot_centroid_y": float(centroid[1]),
                "hotspot_centroid_z": float(centroid[2]),
            }
        )
    return pd.DataFrame(rows)


def pairwise_shape_similarity_table(
    clean_df: pd.DataFrame,
    diagnoses: Sequence[str],
    top_fraction: float = 0.10,
    focus: str = "inward",
) -> pd.DataFrame:
    group_maps = build_group_shape_maps(clean_df, diagnoses=diagnoses)
    rows: List[Dict[str, object]] = []
    for idx, diag_a in enumerate(diagnoses):
        map_a = group_maps.get(diag_a)
        if map_a is None:
            continue
        for diag_b in diagnoses[idx + 1 :]:
            map_b = group_maps.get(diag_b)
            if map_b is None:
                continue
            signed_a = np.asarray(map_a["signed_speed"], dtype=float)
            signed_b = np.asarray(map_b["signed_speed"], dtype=float)
            focus_a = _focus_values_from_group_map(map_a, focus)
            focus_b = _focus_values_from_group_map(map_b, focus)
            weights = np.asarray(map_a["area_weights"], dtype=float)
            mask_a = _weighted_top_fraction_mask(focus_a, weights, top_fraction)
            mask_b = _weighted_top_fraction_mask(focus_b, weights, top_fraction)
            inter_w = float(weights[mask_a & mask_b].sum())
            union_w = float(weights[mask_a | mask_b].sum())
            top_a_w = float(weights[mask_a].sum())
            top_b_w = float(weights[mask_b].sum())
            shared_mask = mask_a & mask_b
            shared_weight = weights[shared_mask]
            shared_focus_a = float(np.average(focus_a[shared_mask], weights=shared_weight)) if shared_mask.any() else np.nan
            shared_focus_b = float(np.average(focus_b[shared_mask], weights=shared_weight)) if shared_mask.any() else np.nan
            rows.append(
                {
                    "diagnosis_a": diag_a,
                    "diagnosis_b": diag_b,
                    "analysis_focus": focus,
                    "top_fraction": float(top_fraction),
                    "focus_speed_corr": float(np.corrcoef(focus_a, focus_b)[0, 1]),
                    "signed_speed_corr": float(np.corrcoef(signed_a, signed_b)[0, 1]),
                    "hotspot_weighted_jaccard": inter_w / union_w if union_w > 0 else np.nan,
                    "hotspot_weighted_dice": 2.0 * inter_w / (top_a_w + top_b_w) if (top_a_w + top_b_w) > 0 else np.nan,
                    "hotspot_overlap_of_a": inter_w / top_a_w if top_a_w > 0 else np.nan,
                    "hotspot_overlap_of_b": inter_w / top_b_w if top_b_w > 0 else np.nan,
                    "shared_hotspot_mean_focus_speed_a": shared_focus_a,
                    "shared_hotspot_mean_focus_speed_b": shared_focus_b,
                    "shared_hotspot_speed_ratio_b_over_a": shared_focus_b / shared_focus_a if shared_mask.any() and shared_focus_a != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _difference_group_map(
    group_map_a: Dict[str, object],
    group_map_b: Dict[str, object],
    metric: str,
) -> Dict[str, object]:
    return {
        "display_vertices": np.asarray(group_map_b["display_vertices"], dtype=float),
        "faces": np.asarray(group_map_b["faces"], dtype=int),
        metric: np.asarray(group_map_b[metric], dtype=float) - np.asarray(group_map_a[metric], dtype=float),
    }


def shape_difference_figures(
    clean_df: pd.DataFrame,
    dataset_label: str,
    diagnoses: Sequence[str],
    comparisons: Sequence[Tuple[str, str]],
) -> List[go.Figure]:
    figures: List[go.Figure] = []
    group_maps = build_group_shape_maps(clean_df, diagnoses=diagnoses)
    for diag_a, diag_b in comparisons:
        map_a = group_maps.get(diag_a)
        map_b = group_maps.get(diag_b)
        if map_a is None or map_b is None:
            continue
        signed_diff = _difference_group_map(map_a, map_b, "signed_speed")
        abs_diff = _difference_group_map(map_a, map_b, "abs_speed")
        figures.append(
            make_shape_speed_figure(
                signed_diff,
                "signed_speed",
                f"{dataset_label}: signed local speed difference ({diag_b} minus {diag_a})",
            )
        )
        figures.append(
            make_shape_speed_figure(
                abs_diff,
                "abs_speed",
                f"{dataset_label}: absolute local speed difference ({diag_b} minus {diag_a})",
            )
        )
    return figures


def step1_figures(bundle: ManifestBundle) -> List[go.Figure]:
    pair_df = build_adjacent_pair_dataframe(bundle.clean_df)
    subject_df = subject_mean_speed_dataframe(pair_df)
    pair_df["dataset_name"] = bundle.name
    diagnoses = ["CN", "AD"]
    return [
        make_speed_distribution_figure(
            subject_df[subject_df["diagnosis"].isin(diagnoses)],
            title=f"{bundle.name}: no-MCI subject mean volume-change speeds",
            diagnoses=diagnoses,
        ),
        make_speed_age_figure(
            pair_df[pair_df["diagnosis"].isin(diagnoses)],
            title=f"{bundle.name}: no-MCI adjacent-pair age-speed curves",
            diagnoses=diagnoses,
        ),
    ]


def step2_figures(bundle: ManifestBundle) -> List[go.Figure]:
    pair_df = build_adjacent_pair_dataframe(bundle.clean_df)
    subject_df = subject_mean_speed_dataframe(pair_df)
    pair_df["dataset_name"] = bundle.name
    diagnoses = ["CN", "MCI", "AD"]
    return [
        make_speed_distribution_figure(
            subject_df[subject_df["diagnosis"].isin(diagnoses)],
            title=f"{bundle.name}: with-MCI subject mean volume-change speeds",
            diagnoses=diagnoses,
        ),
        make_speed_age_figure(
            pair_df[pair_df["diagnosis"].isin(diagnoses)],
            title=f"{bundle.name}: with-MCI adjacent-pair age-speed curves",
            diagnoses=diagnoses,
        ),
    ]
