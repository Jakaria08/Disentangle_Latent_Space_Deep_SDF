from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI"
    / "brainode_comparison_task3_core_brainode_original"
    / "analysis"
)
NOTEBOOK_DIR = ANALYSIS_DIR / "longitudinal_visual_notebooks"
TABLE_DIR = ANALYSIS_DIR / "unified_longitudinal_visual_report" / "tables"
AUDIT_SCAN_VOLUME_CSV = TABLE_DIR / "whole_dataset_volume_audit_scan_volumes.csv"

DATASET_ORDER = ("large_all", "large_qc", "old_small")
DATASET_LABELS = {
    "large_all": "Large ADNI no MCI",
    "large_qc": "QC-filtered large ADNI",
    "old_small": "Old ADNI subset",
}
DATASET_DESCRIPTIONS = {
    "large_all": "All selected strict-left no-MCI ADNI large scans before the later QC longitudinal filter.",
    "large_qc": "QC-filtered large ADNI subset used by the current BrainODE/PCA/SIREN comparison.",
    "old_small": "Older ADNI no-MCI subset used by the previous BrainODE/PCA experiments.",
}
DATASET_COLORS = {
    "large_all": "#2563eb",
    "large_qc": "#0f766e",
    "old_small": "#7c3aed",
}
DIAG_COLORS = {"CN": "#2563eb", "AD": "#dc2626"}
MM3_PER_CM3 = 1000.0


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(120,120,120,{float(alpha)})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{float(alpha)})"


def ordered_dataset_labels(values: Iterable[str]) -> list[str]:
    order = {name: index for index, name in enumerate(DATASET_ORDER)}
    return sorted(set(str(value) for value in values), key=lambda value: order.get(value, 99))


def display_dataset(value: str) -> str:
    return DATASET_LABELS.get(str(value), str(value))


def data_sources_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "display_name": DATASET_LABELS[dataset],
                "description": DATASET_DESCRIPTIONS[dataset],
                "primary_source": str(AUDIT_SCAN_VOLUME_CSV),
            }
            for dataset in DATASET_ORDER
        ]
    )


def _volume_unscale_factor(audit: pd.DataFrame) -> float:
    both = audit.loc[
        audit["metadata_mesh_volume_mm3"].notna()
        & audit["volume"].notna()
        & (pd.to_numeric(audit["volume"], errors="coerce").abs() > 1.0e-12)
    ].copy()
    if both.empty:
        return 1.0
    ratio = pd.to_numeric(both["metadata_mesh_volume_mm3"], errors="coerce") / pd.to_numeric(
        both["volume"],
        errors="coerce",
    )
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.median()) if not ratio.empty else 1.0


def load_cohort_frame() -> pd.DataFrame:
    audit = pd.read_csv(AUDIT_SCAN_VOLUME_CSV, low_memory=False)
    audit = audit.loc[
        audit["dataset"].astype(str).isin(DATASET_ORDER)
        & audit["method"].astype(str).eq("ground_truth")
    ].copy()
    scale = _volume_unscale_factor(audit)
    numeric_cols = [
        "age_years",
        "visit_order",
        "months_from_baseline",
        "metadata_mesh_volume_mm3",
        "mask_volume_mm3",
        "volume",
    ]
    for column in numeric_cols:
        if column in audit.columns:
            audit[column] = pd.to_numeric(audit[column], errors="coerce")
    audit["subject_id"] = audit["subject_id"].astype(str)
    audit["scan_id"] = audit["scan_id"].astype(str)
    audit["split"] = audit["split"].astype(str)
    audit["diagnosis"] = audit["diagnosis"].astype(str).str.upper()
    audit = audit.loc[audit["diagnosis"].isin(["CN", "AD"])].copy()
    audit["dataset"] = pd.Categorical(audit["dataset"].astype(str), categories=DATASET_ORDER, ordered=True)
    audit["dataset_label"] = audit["dataset"].astype(str).map(DATASET_LABELS)
    audit["volume_mm3_source"] = "metadata_mesh_volume_mm3"
    volume_mm3 = audit["metadata_mesh_volume_mm3"].copy()
    mask = volume_mm3.isna() & audit["mask_volume_mm3"].notna()
    volume_mm3.loc[mask] = audit.loc[mask, "mask_volume_mm3"]
    audit.loc[mask, "volume_mm3_source"] = "mask_volume_mm3"
    scaled = volume_mm3.isna() & audit["volume"].notna()
    volume_mm3.loc[scaled] = audit.loc[scaled, "volume"] * scale
    audit.loc[scaled, "volume_mm3_source"] = f"scaled_audit_volume_x_{scale:.6g}"
    audit["volume_mm3"] = volume_mm3
    audit["volume_cm3"] = audit["volume_mm3"] / MM3_PER_CM3
    audit["years_from_baseline"] = audit["months_from_baseline"] / 12.0
    missing_time = audit["years_from_baseline"].isna()
    if missing_time.any():
        baseline_age = audit.groupby(["dataset", "subject_id"], observed=True)["age_years"].transform("min")
        audit.loc[missing_time, "years_from_baseline"] = audit.loc[missing_time, "age_years"] - baseline_age.loc[missing_time]
    audit["time_axis_years"] = (
        audit.groupby(["dataset", "subject_id"], observed=True)["age_years"].transform("min")
        + audit["years_from_baseline"]
    )
    audit = audit.sort_values(["dataset", "split", "diagnosis", "subject_id", "years_from_baseline", "visit_order", "scan_id"])
    return audit.reset_index(drop=True)


def baseline_frame(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_cohort_frame() if frame is None else frame.copy()
    if data.empty:
        return data
    return (
        data.sort_values(["dataset", "subject_id", "years_from_baseline", "visit_order", "scan_id"])
        .groupby(["dataset", "subject_id"], observed=True, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def subject_summary(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_cohort_frame() if frame is None else frame.copy()
    rows: list[dict[str, Any]] = []
    for (dataset, subject_id), group in data.groupby(["dataset", "subject_id"], observed=True, sort=False):
        group = group.sort_values(["years_from_baseline", "visit_order", "scan_id"])
        first = group.iloc[0]
        last = group.iloc[-1]
        followup = finite_float(last["years_from_baseline"]) - finite_float(first["years_from_baseline"])
        baseline_volume = finite_float(first["volume_cm3"])
        final_volume = finite_float(last["volume_cm3"])
        if followup > 1.0e-8 and baseline_volume > 1.0e-8:
            annual_pct = 100.0 * (final_volume - baseline_volume) / baseline_volume / followup
            annual_abs = (final_volume - baseline_volume) / followup
        else:
            annual_pct = float("nan")
            annual_abs = float("nan")
        rows.append(
            {
                "dataset": str(dataset),
                "dataset_label": display_dataset(str(dataset)),
                "subject_id": str(subject_id),
                "split": str(first["split"]),
                "diagnosis": str(first["diagnosis"]),
                "scan_count": int(len(group)),
                "baseline_age_years": finite_float(first["age_years"]),
                "followup_years": followup,
                "baseline_volume_cm3": baseline_volume,
                "final_volume_cm3": final_volume,
                "absolute_change_cm3": final_volume - baseline_volume,
                "annual_absolute_change_cm3_per_year": annual_abs,
                "annual_percent_change": annual_pct,
            }
        )
    return pd.DataFrame(rows)


def cohort_overview(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_cohort_frame() if frame is None else frame.copy()
    subjects = subject_summary(data)
    scan_summary = (
        data.groupby(["dataset", "dataset_label", "diagnosis"], observed=True, sort=False)
        .agg(
            scans=("scan_id", "nunique"),
            subjects=("subject_id", "nunique"),
            mean_age_years=("age_years", "mean"),
            median_volume_cm3=("volume_cm3", "median"),
        )
        .reset_index()
    )
    subj_summary = (
        subjects.groupby(["dataset", "diagnosis"], observed=True, sort=False)
        .agg(
            avg_scans_per_subject=("scan_count", "mean"),
            median_scans_per_subject=("scan_count", "median"),
            median_followup_years=("followup_years", "median"),
            mean_annual_percent_change=("annual_percent_change", "mean"),
            median_annual_percent_change=("annual_percent_change", "median"),
        )
        .reset_index()
    )
    merged = scan_summary.merge(subj_summary, on=["dataset", "diagnosis"], how="left")
    merged["dataset_sort"] = merged["dataset"].astype(str).map({name: index for index, name in enumerate(DATASET_ORDER)})
    return merged.sort_values(["dataset_sort", "diagnosis"]).drop(columns=["dataset_sort"]).reset_index(drop=True)


def plot_subject_and_scan_counts(frame: pd.DataFrame | None = None) -> go.Figure:
    overview = cohort_overview(frame)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Subjects", "Scans"])
    for diagnosis in ("CN", "AD"):
        group = overview.loc[overview["diagnosis"].eq(diagnosis)].copy()
        fig.add_trace(
            go.Bar(
                x=group["dataset_label"],
                y=group["subjects"],
                name=f"{diagnosis} subjects",
                marker_color=DIAG_COLORS[diagnosis],
                legendgroup=diagnosis,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=group["dataset_label"],
                y=group["scans"],
                name=f"{diagnosis} scans",
                marker_color=DIAG_COLORS[diagnosis],
                legendgroup=diagnosis,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(
        title="AD/CN cohort size by dataset",
        template="plotly_white",
        barmode="group",
        width=1200,
        height=500,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 90},
    )
    return fig


def plot_split_counts(frame: pd.DataFrame | None = None) -> go.Figure:
    data = load_cohort_frame() if frame is None else frame.copy()
    grouped = (
        data.groupby(["dataset", "dataset_label", "split", "diagnosis"], observed=True)
        .agg(scans=("scan_id", "nunique"), subjects=("subject_id", "nunique"))
        .reset_index()
    )
    fig = make_subplots(rows=1, cols=3, subplot_titles=[DATASET_LABELS[name] for name in DATASET_ORDER])
    for col, dataset in enumerate(DATASET_ORDER, start=1):
        subset = grouped.loc[grouped["dataset"].astype(str).eq(dataset)].copy()
        for diagnosis in ("CN", "AD"):
            group = subset.loc[subset["diagnosis"].eq(diagnosis)].copy()
            fig.add_trace(
                go.Bar(
                    x=group["split"],
                    y=group["subjects"],
                    name=diagnosis,
                    marker_color=DIAG_COLORS[diagnosis],
                    legendgroup=diagnosis,
                    showlegend=(col == 1),
                    customdata=np.stack([group["scans"].to_numpy()], axis=1) if not group.empty else None,
                    hovertemplate="split=%{x}<br>subjects=%{y}<br>scans=%{customdata[0]}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Split", row=1, col=col)
        fig.update_yaxes(title_text="Subjects", row=1, col=col)
    fig.update_layout(
        title="Train/val/test subject distribution",
        template="plotly_white",
        barmode="group",
        width=1350,
        height=500,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 90},
    )
    return fig


def plot_scan_count_and_followup(frame: pd.DataFrame | None = None) -> go.Figure:
    subjects = subject_summary(load_cohort_frame() if frame is None else frame.copy())
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Scans per subject", "Follow-up length"])
    for diagnosis in ("CN", "AD"):
        group = subjects.loc[subjects["diagnosis"].eq(diagnosis)].copy()
        fig.add_trace(
            go.Box(
                x=group["dataset_label"],
                y=group["scan_count"],
                name=diagnosis,
                marker_color=DIAG_COLORS[diagnosis],
                boxmean=True,
                legendgroup=diagnosis,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=group["dataset_label"],
                y=group["followup_years"],
                name=diagnosis,
                marker_color=DIAG_COLORS[diagnosis],
                boxmean=True,
                legendgroup=diagnosis,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Scans", row=1, col=1)
    fig.update_yaxes(title_text="Years", row=1, col=2)
    fig.update_layout(
        title="Longitudinal density by dataset and diagnosis",
        template="plotly_white",
        boxmode="group",
        width=1250,
        height=520,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 100},
    )
    return fig


def plot_age_distribution(frame: pd.DataFrame | None = None) -> go.Figure:
    data = baseline_frame(load_cohort_frame() if frame is None else frame.copy())
    fig = make_subplots(rows=1, cols=3, subplot_titles=[DATASET_LABELS[name] for name in DATASET_ORDER])
    for col, dataset in enumerate(DATASET_ORDER, start=1):
        subset = data.loc[data["dataset"].astype(str).eq(dataset)].copy()
        for diagnosis in ("CN", "AD"):
            group = subset.loc[subset["diagnosis"].eq(diagnosis)].copy()
            fig.add_trace(
                go.Histogram(
                    x=group["age_years"],
                    name=diagnosis,
                    marker_color=DIAG_COLORS[diagnosis],
                    opacity=0.72,
                    legendgroup=diagnosis,
                    showlegend=(col == 1),
                    nbinsx=18,
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Baseline age (years)", row=1, col=col)
        fig.update_yaxes(title_text="Subjects", row=1, col=col)
    fig.update_layout(
        title="Baseline age distribution",
        template="plotly_white",
        barmode="overlay",
        width=1350,
        height=500,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 90},
    )
    return fig


def plot_baseline_volume_and_atrophy(frame: pd.DataFrame | None = None) -> go.Figure:
    data = load_cohort_frame() if frame is None else frame.copy()
    base = baseline_frame(data)
    subjects = subject_summary(data)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Baseline volume", "Start-to-end atrophy rate"])
    for diagnosis in ("CN", "AD"):
        base_group = base.loc[base["diagnosis"].eq(diagnosis)].copy()
        subj_group = subjects.loc[subjects["diagnosis"].eq(diagnosis)].copy()
        fig.add_trace(
            go.Box(
                x=base_group["dataset_label"],
                y=base_group["volume_cm3"],
                name=diagnosis,
                marker_color=DIAG_COLORS[diagnosis],
                boxmean=True,
                legendgroup=diagnosis,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=subj_group["dataset_label"],
                y=subj_group["annual_percent_change"],
                name=diagnosis,
                marker_color=DIAG_COLORS[diagnosis],
                boxmean=True,
                legendgroup=diagnosis,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#777777", row=1, col=2)
    fig.update_yaxes(title_text="Volume (cm^3)", row=1, col=1)
    fig.update_yaxes(title_text="Annualized volume change (%/year)", row=1, col=2)
    fig.update_layout(
        title="Baseline volume and longitudinal atrophy rate",
        template="plotly_white",
        boxmode="group",
        width=1250,
        height=520,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 100},
    )
    return fig


def atrophy_summary_table(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    subjects = subject_summary(load_cohort_frame() if frame is None else frame.copy())
    summary = (
        subjects.groupby(["dataset", "dataset_label", "diagnosis"], observed=True, sort=False)
        .agg(
            subjects=("subject_id", "nunique"),
            subjects_with_followup=("annual_percent_change", "count"),
            mean_followup_years=("followup_years", "mean"),
            median_followup_years=("followup_years", "median"),
            mean_annual_percent_change=("annual_percent_change", "mean"),
            median_annual_percent_change=("annual_percent_change", "median"),
            mean_annual_absolute_change_cm3=("annual_absolute_change_cm3_per_year", "mean"),
        )
        .reset_index()
    )
    summary["dataset_sort"] = summary["dataset"].astype(str).map({name: idx for idx, name in enumerate(DATASET_ORDER)})
    return summary.sort_values(["dataset_sort", "diagnosis"]).drop(columns=["dataset_sort"]).reset_index(drop=True)


def _subject_relative_curves(frame: pd.DataFrame, value_column: str = "volume_cm3") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, subject_id), group in frame.groupby(["dataset", "subject_id"], observed=True, sort=False):
        group = group.sort_values(["years_from_baseline", "visit_order", "scan_id"])
        baseline = finite_float(group.iloc[0][value_column])
        if baseline <= 1.0e-8:
            continue
        for row in group.itertuples(index=False):
            current = finite_float(getattr(row, value_column))
            rows.append(
                {
                    "dataset": str(dataset),
                    "dataset_label": display_dataset(str(dataset)),
                    "subject_id": str(subject_id),
                    "diagnosis": str(getattr(row, "diagnosis")),
                    "years_from_baseline": finite_float(getattr(row, "years_from_baseline")),
                    "relative_change_pct": 100.0 * (current - baseline) / baseline,
                    "volume_cm3": current,
                }
            )
    return pd.DataFrame(rows)


def plot_volume_trends(frame: pd.DataFrame | None = None) -> go.Figure:
    data = load_cohort_frame() if frame is None else frame.copy()
    relative = _subject_relative_curves(data)
    fig = make_subplots(rows=1, cols=3, subplot_titles=[DATASET_LABELS[name] for name in DATASET_ORDER])
    for col, dataset in enumerate(DATASET_ORDER, start=1):
        subset = relative.loc[relative["dataset"].eq(dataset)].copy()
        max_followup = finite_float(subset["years_from_baseline"].max())
        if not math.isfinite(max_followup) or max_followup <= 0:
            grid = np.array([0.0])
        else:
            grid = np.arange(0.0, min(12.0, math.ceil(max_followup)) + 0.01, 0.5)
        for diagnosis in ("CN", "AD"):
            diag = subset.loc[subset["diagnosis"].eq(diagnosis)].copy()
            curves = []
            for _, group in diag.groupby("subject_id", sort=False):
                group = group.sort_values("years_from_baseline")
                x = group["years_from_baseline"].to_numpy(dtype=float)
                y = group["relative_change_pct"].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                if len(x) < 2:
                    continue
                interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
                interp[(grid < x.min()) | (grid > x.max())] = np.nan
                curves.append(interp)
            if not curves:
                continue
            matrix = np.vstack(curves)
            finite = np.isfinite(matrix).any(axis=0)
            mean = np.full(grid.shape, np.nan)
            q25 = np.full(grid.shape, np.nan)
            q75 = np.full(grid.shape, np.nan)
            mean[finite] = np.nanmean(matrix[:, finite], axis=0)
            q25[finite] = np.nanquantile(matrix[:, finite], 0.25, axis=0)
            q75[finite] = np.nanquantile(matrix[:, finite], 0.75, axis=0)
            color = DIAG_COLORS[diagnosis]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([grid, grid[::-1]]),
                    y=np.concatenate([q75, q25[::-1]]),
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.14),
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=mean,
                    mode="lines+markers",
                    name=diagnosis,
                    line={"color": color, "width": 3},
                    marker={"size": 6},
                    legendgroup=diagnosis,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Years from baseline", row=1, col=col)
        fig.update_yaxes(title_text="Relative volume change (%)", row=1, col=col)
    fig.update_layout(
        title="Observed hippocampus volume trend by dataset",
        template="plotly_white",
        width=1450,
        height=540,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 90},
    )
    return fig


def plot_volume_vs_age(frame: pd.DataFrame | None = None) -> go.Figure:
    data = load_cohort_frame() if frame is None else frame.copy()
    base = baseline_frame(data)
    fig = make_subplots(rows=1, cols=3, subplot_titles=[DATASET_LABELS[name] for name in DATASET_ORDER])
    for col, dataset in enumerate(DATASET_ORDER, start=1):
        subset = base.loc[base["dataset"].astype(str).eq(dataset)].copy()
        for diagnosis in ("CN", "AD"):
            group = subset.loc[subset["diagnosis"].eq(diagnosis)].copy()
            fig.add_trace(
                go.Scatter(
                    x=group["age_years"],
                    y=group["volume_cm3"],
                    mode="markers",
                    name=diagnosis,
                    marker={"color": DIAG_COLORS[diagnosis], "size": 6, "opacity": 0.65},
                    legendgroup=diagnosis,
                    showlegend=(col == 1),
                    hovertemplate="age=%{x:.1f}<br>volume=%{y:.3f} cm^3<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Baseline age (years)", row=1, col=col)
        fig.update_yaxes(title_text="Baseline volume (cm^3)", row=1, col=col)
    fig.update_layout(
        title="Baseline volume vs age",
        template="plotly_white",
        width=1450,
        height=520,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 90},
    )
    return fig


def qc_filter_impact_table(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_cohort_frame() if frame is None else frame.copy()
    large = data.loc[data["dataset"].astype(str).eq("large_all")].copy()
    qc = data.loc[data["dataset"].astype(str).eq("large_qc")].copy()
    qc_scan_ids = set(qc["scan_id"].astype(str))
    large = large.copy()
    large["qc_status"] = np.where(large["scan_id"].astype(str).isin(qc_scan_ids), "kept_in_qc", "removed_by_qc")
    rows = []
    for (diagnosis, status), group in large.groupby(["diagnosis", "qc_status"], sort=False):
        rows.append(
            {
                "diagnosis": diagnosis,
                "qc_status": status,
                "scans": int(group["scan_id"].nunique()),
                "subjects": int(group["subject_id"].nunique()),
                "median_volume_cm3": float(group["volume_cm3"].median()),
                "median_age_years": float(group["age_years"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_qc_filter_impact(frame: pd.DataFrame | None = None) -> go.Figure:
    impact = qc_filter_impact_table(frame)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Scans", "Subjects"])
    for status, color in (("kept_in_qc", "#0f766e"), ("removed_by_qc", "#f97316")):
        group = impact.loc[impact["qc_status"].eq(status)].copy()
        fig.add_trace(
            go.Bar(
                x=group["diagnosis"],
                y=group["scans"],
                name=status,
                marker_color=color,
                legendgroup=status,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=group["diagnosis"],
                y=group["subjects"],
                name=status,
                marker_color=color,
                legendgroup=status,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Scans", row=1, col=1)
    fig.update_yaxes(title_text="Subjects", row=1, col=2)
    fig.update_layout(
        title="Large no-MCI scans kept vs removed by QC filter",
        template="plotly_white",
        barmode="group",
        width=1100,
        height=500,
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 50, "r": 20, "t": 80, "b": 90},
    )
    return fig


def _closest_to_value(group: pd.DataFrame, value: float) -> pd.Series | None:
    clean = group.loc[group["volume_cm3"].notna()].copy()
    if clean.empty:
        return None
    index = (clean["volume_cm3"] - float(value)).abs().idxmin()
    return clean.loc[index]


def shape_examples_table(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    data = baseline_frame(load_cohort_frame() if frame is None else frame.copy())
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        subset = data.loc[data["dataset"].astype(str).eq(dataset)].copy()
        cn = subset.loc[subset["diagnosis"].eq("CN")].copy()
        ad = subset.loc[subset["diagnosis"].eq("AD")].copy()
        selections: list[tuple[str, pd.Series | None]] = []
        if not cn.empty:
            selections.append(("CN median baseline volume", _closest_to_value(cn, float(cn["volume_cm3"].median()))))
            selections.append(("CN large baseline volume", cn.sort_values("volume_cm3", ascending=False).iloc[0]))
            selections.append(("CN small baseline volume", cn.sort_values("volume_cm3", ascending=True).iloc[0]))
        if not ad.empty:
            selections.append(("AD median baseline volume", _closest_to_value(ad, float(ad["volume_cm3"].median()))))
        for label, row in selections:
            if row is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "example": label,
                    "diagnosis": str(row["diagnosis"]),
                    "subject_id": str(row["subject_id"]),
                    "scan_id": str(row["scan_id"]),
                    "split": str(row["split"]),
                    "age_years": finite_float(row["age_years"]),
                    "volume_cm3": finite_float(row["volume_cm3"]),
                    "mesh_path": str(row["ground_truth_mesh_path"]),
                }
            )
    return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def load_mesh_cached(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Empty mesh scene at {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh at {path}, got {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Empty mesh at {path}")
    mesh = loaded.copy()
    try:
        mesh.process(validate=True)
        mesh.fill_holes()
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _mesh_trace(mesh: trimesh.Trimesh, name: str, color: str) -> go.Mesh3d:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=1.0,
        flatshading=True,
        lighting={"ambient": 0.62, "diffuse": 0.78, "specular": 0.18, "roughness": 0.72, "fresnel": 0.02},
        lightposition={"x": 180, "y": 140, "z": 120},
        showlegend=False,
        hovertemplate=f"{name}<extra></extra>",
    )


def plot_shape_examples(frame: pd.DataFrame | None = None) -> go.Figure:
    examples = shape_examples_table(load_cohort_frame() if frame is None else frame.copy())
    example_order = [
        "CN median baseline volume",
        "AD median baseline volume",
        "CN large baseline volume",
        "CN small baseline volume",
    ]
    rows = []
    for dataset in DATASET_ORDER:
        subset = examples.loc[examples["dataset"].eq(dataset)].copy()
        by_example = {str(row.example): row for row in subset.itertuples(index=False)}
        rows.append([by_example.get(example) for example in example_order])
    fig = make_subplots(
        rows=len(DATASET_ORDER),
        cols=len(example_order),
        specs=[[{"type": "scene"} for _ in example_order] for _ in DATASET_ORDER],
        subplot_titles=[
            f"{DATASET_LABELS[dataset]}<br>{example.replace(' baseline volume', '')}"
            for dataset in DATASET_ORDER
            for example in example_order
        ],
    )
    failures: list[str] = []
    for row_idx, dataset in enumerate(DATASET_ORDER, start=1):
        for col_idx, entry in enumerate(rows[row_idx - 1], start=1):
            if entry is None:
                continue
            try:
                mesh = load_mesh_cached(str(entry.mesh_path))
            except Exception as exc:
                failures.append(f"{DATASET_LABELS[dataset]} {entry.scan_id}: {exc}")
                continue
            color = DIAG_COLORS.get(str(entry.diagnosis), "#737373")
            title = f"{entry.diagnosis} {entry.subject_id}<br>{entry.volume_cm3:.3f} cm^3, age {entry.age_years:.1f}"
            fig.add_trace(_mesh_trace(mesh, title, color), row=row_idx, col=col_idx)
            fig.update_scenes(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                aspectmode="data",
                camera={"eye": {"x": 1.55, "y": 1.45, "z": 0.8}},
                row=row_idx,
                col=col_idx,
            )
    title = "Representative solid left-hippocampus meshes"
    if failures:
        title += f" | skipped {len(failures)} failed mesh loads"
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1500,
        height=1080,
        margin={"l": 10, "r": 10, "t": 120, "b": 10},
    )
    return fig
