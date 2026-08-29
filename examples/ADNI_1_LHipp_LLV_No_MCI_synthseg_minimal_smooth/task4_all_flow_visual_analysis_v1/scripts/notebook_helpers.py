#!/usr/bin/env python3
"""Load-only plotting helpers for the all-flow results notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {"PCA": "#4C78A8", "Spiral": "#F58518", "Adaptive": "#54A24B", "INR": "#E45756"}
LEGACY_COLORS = {"PCA Cocycle": "#4C78A8", "INR Cocycle": "#E45756", "Latent ODE": "#B279A2", "BrainODE": "#FF9DA6"}


def load_cache(root: str | Path) -> dict[str, object]:
    root = Path(root).expanduser().resolve()
    tables = root / "tables"
    velocity_root = Path(__file__).resolve().parents[1] / "derived_cache" / "instantaneous_surface_velocity_v1"
    velocity_tables = velocity_root / "tables"
    velocity_arrays = velocity_root / "arrays"
    return {
        "root": root,
        "endpoint": pd.read_csv(tables / "current_endpoint_metrics.csv"),
        "metric_summary": pd.read_csv(tables / "current_metric_summary.csv"),
        "bootstrap": pd.read_csv(tables / "current_metric_bootstrap.csv"),
        "velocity": pd.read_csv(tables / "current_velocity_per_scan.csv"),
        "velocity_summary": pd.read_csv(tables / "current_velocity_summary.csv"),
        "paired_tests": pd.read_csv(tables / "current_paired_tests.csv"),
        "consistency": pd.read_csv(tables / "current_consistency.csv"),
        "inventory": pd.read_csv(tables / "checkpoint_inventory.csv"),
        "legacy_endpoint": pd.read_csv(tables / "legacy_endpoint_metrics.csv", low_memory=False),
        "legacy_velocity": pd.read_csv(tables / "legacy_velocity_per_scan.csv", low_memory=False),
        "current_maps": np.load(root / "arrays" / "current_surface_maps.npz", allow_pickle=False),
        "legacy_maps": np.load(root / "arrays" / "legacy_surface_maps.npz", allow_pickle=False),
        "current_surface_velocity": pd.read_csv(velocity_tables / "current_instantaneous_surface_velocity.csv"),
        "legacy_surface_velocity": pd.read_csv(velocity_tables / "legacy_instantaneous_surface_velocity.csv"),
        "current_instantaneous_maps": np.load(velocity_arrays / "current_instantaneous_surface_maps.npz", allow_pickle=False),
        "legacy_instantaneous_maps": np.load(velocity_arrays / "legacy_instantaneous_surface_maps.npz", allow_pickle=False),
    }


def metric_bars(data: dict[str, object], metric: str, diagnosis: str = "overall", title: str | None = None) -> go.Figure:
    summary = data["metric_summary"]
    current = summary[summary.diagnosis.eq(diagnosis)].copy()
    long = []
    for row in current.itertuples(index=False):
        long.extend((
            {"Method": row.method_label, "Series": "Cocycle", "Value": getattr(row, metric)},
            {"Method": row.method_label, "Series": "No change", "Value": getattr(row, f"nochange_{metric}")},
            {"Method": row.method_label, "Series": "Representation floor", "Value": getattr(row, f"representation_floor_{metric}")},
        ))
    frame = pd.DataFrame(long)
    figure = px.bar(frame, x="Method", y="Value", color="Series", barmode="group", title=title or metric.replace("_", " ").title())
    figure.update_layout(template="plotly_white", legend_title_text="")
    return figure


def improvement_forest(data: dict[str, object], metrics: list[str]) -> go.Figure:
    summary = data["metric_summary"]
    summary = summary[summary.diagnosis.eq("overall")].copy()
    rows = []
    higher = {"dice", "iou", "normal_change_pearson", "normal_change_spearman", "hotspot_dice"}
    for item in summary.itertuples(index=False):
        for metric in metrics:
            model, baseline = float(getattr(item, metric)), float(getattr(item, f"nochange_{metric}"))
            improvement = model - baseline if metric in higher else 100.0 * (1.0 - model / baseline)
            rows.append({"Method": item.method_label, "Metric": metric.replace("_", " "), "Improvement": improvement})
    frame = pd.DataFrame(rows)
    figure = px.scatter(frame, x="Improvement", y="Metric", color="Method", symbol="Method", color_discrete_map=COLORS)
    figure.add_vline(x=0.0, line_dash="dash", line_color="black")
    figure.update_layout(template="plotly_white", title="Improvement over no change", xaxis_title="Percent reduction (errors) or absolute gain (overlaps/correlation)")
    return figure


def volume_trends(data: dict[str, object]) -> go.Figure:
    endpoint = data["endpoint"].copy()
    grouped = endpoint.groupby(["method_label", "diagnosis"], as_index=False).agg(
        predicted=("predicted_signed_log_volume_rate_per_year", "mean"), observed=("observed_signed_log_volume_rate_per_year", "mean")
    )
    grouped["predicted_percent"] = 100.0 * np.expm1(grouped.predicted)
    grouped["observed_percent"] = 100.0 * np.expm1(grouped.observed)
    long = grouped.melt(id_vars=["method_label", "diagnosis"], value_vars=["predicted_percent", "observed_percent"], var_name="Series", value_name="Percent/year")
    long["Series"] = long.Series.map({"predicted_percent": "Predicted", "observed_percent": "Observed"})
    figure = px.bar(long, x="method_label", y="Percent/year", color="Series", facet_col="diagnosis", barmode="group", title="CN and AD annual volume trends")
    figure.update_layout(template="plotly_white", xaxis_title="", legend_title_text="")
    return figure


def volume_calibration(data: dict[str, object]) -> go.Figure:
    endpoint = data["endpoint"]
    figure = px.scatter(endpoint, x="observed_signed_log_volume_rate_per_year", y="predicted_signed_log_volume_rate_per_year", color="method_label", symbol="diagnosis", hover_data=["subject_id"], color_discrete_map=COLORS, title="Predicted versus observed volume rate")
    bounds = [min(endpoint.observed_signed_log_volume_rate_per_year.min(), endpoint.predicted_signed_log_volume_rate_per_year.min()), max(endpoint.observed_signed_log_volume_rate_per_year.max(), endpoint.predicted_signed_log_volume_rate_per_year.max())]
    figure.add_shape(type="line", x0=bounds[0], y0=bounds[0], x1=bounds[1], y1=bounds[1], line=dict(color="black", dash="dash"))
    figure.update_layout(template="plotly_white", xaxis_title="Observed signed log-volume rate/year", yaxis_title="Predicted signed log-volume rate/year", legend_title_text="")
    return figure


def velocity_scatter(data: dict[str, object]) -> go.Figure:
    velocity = data["velocity"]
    figure = px.scatter(velocity, x="real_rms_per_coordinate_per_year", y="model_rms_per_coordinate_per_year", color="method_label", symbol="diagnosis", opacity=0.65, color_discrete_map=COLORS, title="Latent instantaneous speed: Model versus Observed (GT estimate)")
    maximum = max(velocity.real_rms_per_coordinate_per_year.max(), velocity.model_rms_per_coordinate_per_year.max())
    figure.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum, line=dict(color="black", dash="dash"))
    figure.update_layout(template="plotly_white", xaxis_title="Observed (GT estimate): RMS standardized coordinate/year", yaxis_title="Model generator: RMS standardized coordinate/year", legend_title_text="")
    return figure


def velocity_alignment(data: dict[str, object]) -> go.Figure:
    velocity = data["velocity"]
    figure = px.box(velocity, x="method_label", y="velocity_cosine", color="diagnosis", points="outliers", title="Latent direction agreement with Observed (GT estimate)")
    figure.add_hline(y=0.0, line_dash="dash", line_color="black")
    figure.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Cosine similarity: Model versus Observed (GT estimate)", legend_title_text="")
    return figure


def velocity_by_age(data: dict[str, object]) -> go.Figure:
    velocity = data["velocity"].copy()
    velocity["age_bin"] = pd.cut(velocity.age_years, bins=[0, 70, 75, 80, 85, 200], labels=["<70", "70–75", "75–80", "80–85", "85+"])
    grouped = velocity.groupby(["method_label", "diagnosis", "age_bin"], observed=True, as_index=False).agg(model=("model_rms_per_coordinate_per_year", "median"), real=("real_rms_per_coordinate_per_year", "median"), scans=("scan_id", "size"), subjects=("subject_id", "nunique"))
    long = grouped.melt(id_vars=["method_label", "diagnosis", "age_bin", "scans", "subjects"], value_vars=["model", "real"], var_name="Series", value_name="Speed")
    long["Series"] = long.Series.map({"model": "Model generator", "real": "Observed (GT estimate)"})
    figure = px.line(long, x="age_bin", y="Speed", color="method_label", line_dash="Series", facet_col="diagnosis", markers=True, hover_data=["subjects", "scans"], color_discrete_map=COLORS, title="Latent instantaneous speed by age")
    figure.update_layout(template="plotly_white", xaxis_title="Age (years)", yaxis_title="Median RMS standardized coordinate/year", legend_title_text="")
    return figure


def diagonal_check(data: dict[str, object]) -> go.Figure:
    velocity = data["velocity"]
    figure = px.box(velocity, x="method_label", y="diagonal_fd_rmse_per_coordinate_per_year", color="method_label", color_discrete_map=COLORS, log_y=True, title="Internal generator consistency: analytic value versus small-step transport")
    figure.update_layout(template="plotly_white", xaxis_title="", yaxis_title="RMS difference/year (log scale)", showlegend=False)
    return figure


def consistency_plot(data: dict[str, object]) -> go.Figure:
    frame = data["consistency"].melt(id_vars=["method_label"], value_vars=["semigroup_mean", "inverse_mean"], var_name="Defect", value_name="Relative defect")
    frame["Defect"] = frame.Defect.map({"semigroup_mean": "Semigroup", "inverse_mean": "Inverse"})
    figure = px.bar(frame, x="method_label", y="Relative defect", color="Defect", barmode="group", log_y=True, title="Cocycle consistency defects")
    figure.update_layout(template="plotly_white", xaxis_title="", legend_title_text="")
    return figure


def condition_velocity_gap(data: dict[str, object]) -> go.Figure:
    velocity = data["velocity"]
    figure = px.box(velocity, x="method_label", y="ad_minus_cn_condition_velocity_rms_per_coordinate_per_year", color="diagnosis", points=False, title="Same-shape AD-versus-CN condition effect in latent space")
    figure.update_layout(template="plotly_white", xaxis_title="", yaxis_title="RMS of AD-condition minus CN-condition field/coordinate/year", legend_title_text="")
    return figure


def legacy_endpoint_plot(data: dict[str, object], metric: str = "assd") -> go.Figure:
    frame = data["legacy_endpoint"].copy()
    grouped = frame.groupby(["method_label", "diagnosis"], as_index=False)[metric].mean()
    figure = px.bar(grouped, x="method_label", y=metric, color="diagnosis", barmode="group", title="Existing completed architecture comparison (separate cohort)")
    figure.update_layout(template="plotly_white", xaxis_title="", legend_title_text="")
    return figure


def legacy_horizon_plot(data: dict[str, object], metric: str = "assd") -> go.Figure:
    frame = data["legacy_endpoint"].copy()
    frame["horizon"] = pd.cut(frame.gap_years, bins=[0, 1, 2, np.inf], labels=["≤1 year", "1–2 years", ">2 years"], include_lowest=True)
    grouped = frame.groupby(["method_label", "horizon"], observed=True, as_index=False)[metric].mean()
    figure = px.line(grouped, x="horizon", y=metric, color="method_label", markers=True, color_discrete_map=LEGACY_COLORS, title="Existing architecture performance by follow-up horizon")
    figure.update_layout(template="plotly_white", xaxis_title="Follow-up horizon", legend_title_text="")
    return figure


def legacy_velocity_plot(data: dict[str, object]) -> go.Figure:
    frame = data["legacy_velocity"].copy()
    scale = np.sqrt(frame.latent_dim.astype(float))
    frame["model_rms"] = frame.model_velocity_z_l2_per_year / scale
    frame["observed_rms"] = frame.real_velocity_z_l2_per_year / scale
    grouped = frame.groupby(["method_label", "diagnosis"], as_index=False).agg(model=("model_rms", "median"), real=("observed_rms", "median"))
    figure = px.scatter(grouped, x="real", y="model", color="method_label", symbol="diagnosis", color_discrete_map=LEGACY_COLORS, title="Existing architectures: latent speed, Model versus Observed (GT estimate)")
    maximum = max(grouped.real.max(), grouped.model.max())
    figure.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum, line=dict(color="black", dash="dash"))
    figure.update_layout(template="plotly_white", xaxis_title="Observed (GT estimate): RMS standardized coordinate/year", yaxis_title="Model generator: RMS standardized coordinate/year", legend_title_text="")
    return figure


def legacy_velocity_alignment(data: dict[str, object]) -> go.Figure:
    frame = data["legacy_velocity"].copy()
    figure = px.box(frame, x="method_label", y="velocity_cosine_real_vs_model", color="diagnosis", points="outliers", color_discrete_map=LEGACY_COLORS, title="Existing architectures: latent direction agreement with Observed (GT estimate)")
    figure.add_hline(y=0.0, line_dash="dash", line_color="black")
    figure.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Cosine similarity: Model versus Observed (GT estimate)", legend_title_text="")
    return figure


def legacy_velocity_by_age(data: dict[str, object]) -> go.Figure:
    frame = data["legacy_velocity"].copy()
    scale = np.sqrt(frame.latent_dim.astype(float))
    frame["model_rms"] = frame.model_velocity_z_l2_per_year / scale
    frame["observed_rms"] = frame.real_velocity_z_l2_per_year / scale
    frame["age_bin"] = pd.cut(frame.age_years, bins=[0, 70, 75, 80, 85, 200], labels=["<70", "70–75", "75–80", "80–85", "85+"])
    grouped = frame.groupby(["method_label", "diagnosis", "age_bin"], observed=True, as_index=False).agg(model=("model_rms", "median"), observed_gt=("observed_rms", "median"), scans=("scan_id", "size"), subjects=("subject_id", "nunique"))
    long = grouped.melt(id_vars=["method_label", "diagnosis", "age_bin", "scans", "subjects"], value_vars=["model", "observed_gt"], var_name="Series", value_name="Speed")
    long["Series"] = long.Series.map({"model": "Model generator", "observed_gt": "Observed (GT estimate)"})
    figure = px.line(long, x="age_bin", y="Speed", color="method_label", line_dash="Series", facet_col="diagnosis", markers=True, hover_data=["subjects", "scans"], color_discrete_map=LEGACY_COLORS, title="Existing architectures: latent instantaneous speed by age")
    figure.update_layout(template="plotly_white", xaxis_title="Age (years)", yaxis_title="Median RMS standardized coordinate/year", legend_title_text="")
    return figure


def _surface_velocity_frame(data: dict[str, object], legacy: bool) -> tuple[pd.DataFrame, dict[str, str], str]:
    if legacy:
        return data["legacy_surface_velocity"].copy(), LEGACY_COLORS, "existing architecture cohort"
    return data["current_surface_velocity"].copy(), COLORS, "matched current cohort"


def surface_speed_by_age(data: dict[str, object], legacy: bool = False) -> go.Figure:
    frame, colors, cohort = _surface_velocity_frame(data, legacy)
    frame["age_bin"] = pd.cut(frame.age_years, bins=[0, 70, 75, 80, 85, 200], labels=["<70", "70–75", "75–80", "80–85", "85+"])
    grouped = frame.groupby(["method_label", "diagnosis", "age_bin"], observed=True, as_index=False).agg(
        model=("model_normal_rms_mm_per_year", "median"),
        observed_gt=("observed_gt_normal_rms_mm_per_year", "median"),
        scans=("scan_id", "size"),
        subjects=("subject_id", "nunique"),
    )
    long = grouped.melt(id_vars=["method_label", "diagnosis", "age_bin", "scans", "subjects"], value_vars=["model", "observed_gt"], var_name="Series", value_name="Speed")
    long["Series"] = long.Series.map({"model": "Model generator", "observed_gt": "Observed (GT estimate)"})
    figure = px.line(long, x="age_bin", y="Speed", color="method_label", line_dash="Series", facet_col="diagnosis", markers=True, hover_data=["subjects", "scans"], color_discrete_map=colors, title=f"Surface instantaneous speed by age — {cohort}")
    figure.update_layout(template="plotly_white", xaxis_title="Age (years)", yaxis_title="Area-weighted RMS normal speed (mm/year)", legend_title_text="")
    return figure


def surface_signed_trend_by_age(data: dict[str, object], legacy: bool = False) -> go.Figure:
    frame, colors, cohort = _surface_velocity_frame(data, legacy)
    frame["age_bin"] = pd.cut(frame.age_years, bins=[0, 70, 75, 80, 85, 200], labels=["<70", "70–75", "75–80", "80–85", "85+"])
    grouped = frame.groupby(["method_label", "diagnosis", "age_bin"], observed=True, as_index=False).agg(
        model=("model_normal_mean_mm_per_year", "median"),
        observed_gt=("observed_gt_normal_mean_mm_per_year", "median"),
        scans=("scan_id", "size"),
        subjects=("subject_id", "nunique"),
    )
    long = grouped.melt(id_vars=["method_label", "diagnosis", "age_bin", "scans", "subjects"], value_vars=["model", "observed_gt"], var_name="Series", value_name="Velocity")
    long["Series"] = long.Series.map({"model": "Model generator", "observed_gt": "Observed (GT estimate)"})
    figure = px.line(long, x="age_bin", y="Velocity", color="method_label", line_dash="Series", facet_col="diagnosis", markers=True, hover_data=["subjects", "scans"], color_discrete_map=colors, title=f"Signed surface-normal velocity by age — {cohort}")
    figure.add_hline(y=0.0, line_dash="dash", line_color="black")
    figure.update_layout(template="plotly_white", xaxis_title="Age (years)", yaxis_title="Area-weighted mean normal velocity (mm/year)", legend_title_text="")
    return figure


def surface_condition_effect_by_age(data: dict[str, object], legacy: bool = False) -> go.Figure:
    frame, colors, cohort = _surface_velocity_frame(data, legacy)
    frame["age_bin"] = pd.cut(frame.age_years, bins=[0, 70, 75, 80, 85, 200], labels=["<70", "70–75", "75–80", "80–85", "85+"])
    grouped = frame.groupby(["method_label", "diagnosis", "age_bin"], observed=True, as_index=False).agg(
        effect=("conditional_ad_minus_cn_normal_rms_mm_per_year", "median"),
        scans=("scan_id", "size"),
        subjects=("subject_id", "nunique"),
    )
    figure = px.line(grouped, x="age_bin", y="effect", color="method_label", facet_col="diagnosis", markers=True, hover_data=["subjects", "scans"], color_discrete_map=colors, title=f"Same-shape AD-versus-CN condition effect on the surface — {cohort}")
    figure.update_layout(template="plotly_white", xaxis_title="Age (years)", yaxis_title="RMS AD-condition minus CN-condition normal velocity (mm/year)", legend_title_text="")
    return figure


def surface_velocity_agreement(data: dict[str, object], legacy: bool = False) -> go.Figure:
    frame, colors, cohort = _surface_velocity_frame(data, legacy)
    grouped = frame.groupby(["method_label", "diagnosis"], as_index=False).agg(
        MAE=("normal_mae_mm_per_year", "mean"),
        Pearson=("normal_pearson", "mean"),
        Sign_agreement=("normal_sign_agreement", "mean"),
        Speed_ratio=("model_to_observed_speed_ratio", "median"),
    )
    figure = make_subplots(rows=2, cols=2, subplot_titles=("Normal-velocity MAE", "Spatial Pearson correlation", "Inward/outward sign agreement", "Model / Observed speed ratio"))
    specifications = (("MAE", 1, 1), ("Pearson", 1, 2), ("Sign_agreement", 2, 1), ("Speed_ratio", 2, 2))
    for metric, row, column in specifications:
        for diagnosis, pattern in (("CN", ""), ("AD", "/")):
            current = grouped[grouped.diagnosis.eq(diagnosis)]
            figure.add_trace(go.Bar(
                x=current.method_label, y=current[metric], name=diagnosis,
                marker={"color": [colors.get(label, "#777777") for label in current.method_label], "pattern": {"shape": pattern}},
                legendgroup=diagnosis, showlegend=(row, column) == (1, 1),
            ), row=row, col=column)
    figure.add_hline(y=0.0, line_dash="dash", line_color="black", row=1, col=2)
    figure.add_hline(y=1.0, line_dash="dash", line_color="black", row=2, col=2)
    figure.update_yaxes(title_text="mm/year (lower is better)", row=1, col=1)
    figure.update_yaxes(title_text="correlation (higher is better)", row=1, col=2)
    figure.update_yaxes(title_text="fraction (higher is better)", range=[0, 1], row=2, col=1)
    figure.update_yaxes(title_text="ratio (1 is matched)", row=2, col=2)
    figure.update_layout(template="plotly_white", title=f"Surface velocity agreement metrics — {cohort}", height=720, barmode="group", legend_title_text="Diagnosis")
    return figure


def instantaneous_surface_group_maps(data: dict[str, object], legacy: bool = False) -> go.Figure:
    if legacy:
        arrays = data["legacy_instantaneous_maps"]
        methods = [("observed_gt", "Observed (GT estimate)"), ("pca_cocycle", "PCA Cocycle"), ("inr_cocycle", "INR Cocycle"), ("latent_ode", "Latent ODE"), ("brainode", "BrainODE")]
        title = "Instantaneous surface velocity: AD-minus-CN observed-group difference — existing architecture cohort"
    else:
        arrays = data["current_instantaneous_maps"]
        methods = [("observed_gt", "Observed (GT estimate)"), ("pca", "PCA"), ("spiral", "Spiral"), ("adaptive", "Adaptive"), ("inr", "INR")]
        title = "Instantaneous surface velocity: AD-minus-CN observed-group difference — matched current cohort"
    vertices, faces = arrays["template_vertices"], arrays["faces"]
    values = [arrays["observed_gt_group_gap"] if key == "observed_gt" else arrays[f"{key}_model_group_gap"] for key, _ in methods]
    bound = max(float(np.quantile(np.abs(np.concatenate(values)), 0.98)), 1.0e-6)
    figure = make_subplots(rows=1, cols=len(methods), specs=[[{"type": "scene"}] * len(methods)], subplot_titles=[label for _, label in methods], horizontal_spacing=0.01)
    for column, (item, intensity) in enumerate(zip(methods, values), start=1):
        figure.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=intensity, colorscale="RdBu_r", cmin=-bound, cmax=bound, showscale=column == len(methods),
            colorbar={"title": "AD−CN<br>mm/year"}, hovertemplate="%{intensity:.4f} mm/year<extra></extra>",
        ), row=1, col=column)
    figure.update_layout(template="plotly_white", title=title, height=430, margin={"l": 0, "r": 0, "b": 0, "t": 70})
    for index in range(1, len(methods) + 1):
        figure.layout[f"scene{index if index > 1 else ''}"].update(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, camera={"eye": {"x": 1.5, "y": 1.2, "z": 0.8}})
    return figure


def surface_map_figure(arrays, methods: list[tuple[str, str]], title: str) -> go.Figure:
    vertices, faces = arrays["template_vertices"], arrays["faces"]
    values = [arrays[f"{key}_gap"] for key, _ in methods]
    bound = float(np.quantile(np.abs(np.concatenate(values)), 0.98))
    figure = make_subplots(rows=1, cols=len(methods), specs=[[{"type": "scene"}] * len(methods)], subplot_titles=[label for _, label in methods], horizontal_spacing=0.01)
    for column, ((key, _), intensity) in enumerate(zip(methods, values), start=1):
        figure.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=intensity, colorscale="RdBu_r", cmin=-bound, cmax=bound, showscale=column == len(methods), colorbar=dict(title="AD−CN<br>mm/year"), hovertemplate="%{intensity:.4f} mm/year<extra></extra>"), row=1, col=column)
    figure.update_layout(template="plotly_white", title=title, height=430, margin=dict(l=0, r=0, b=0, t=70))
    for index in range(1, len(methods) + 1):
        figure.layout[f"scene{index if index > 1 else ''}"].update(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, camera=dict(eye=dict(x=1.5, y=1.2, z=0.8)))
    return figure


def current_surface_maps(data: dict[str, object]) -> go.Figure:
    return surface_map_figure(data["current_maps"], [("observed", "Observed"), ("pca", "PCA"), ("spiral", "Spiral"), ("adaptive", "Adaptive"), ("inr", "INR")], "Average AD-minus-CN annual surface change — matched current cohort")


def legacy_surface_maps(data: dict[str, object]) -> go.Figure:
    return surface_map_figure(data["legacy_maps"], [("observed", "Observed"), ("inr_cocycle", "INR Cocycle"), ("latent_ode", "Latent ODE"), ("brainode", "BrainODE")], "Average AD-minus-CN annual surface change — existing architecture cohort")
