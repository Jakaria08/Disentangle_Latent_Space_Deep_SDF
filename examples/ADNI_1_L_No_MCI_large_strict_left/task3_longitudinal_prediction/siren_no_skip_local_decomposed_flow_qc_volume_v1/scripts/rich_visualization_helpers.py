from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import direct_flow_rich_notebook_helpers as base


DIAGNOSIS_ORDER = ("CN", "AD")
METHOD_ORDER = (
    "real_observed",
    "composed_from_base",
    "direct_from_base",
    "base_reconstruction_no_change",
)
METHOD_LABELS = {
    "real_observed": "real observed",
    "composed_from_base": "composed prediction",
    "direct_from_base": "direct prediction",
    "base_reconstruction_no_change": "no-change",
}
METHOD_COLORS = {
    "real_observed": "#111111",
    "composed_from_base": "#d62728",
    "direct_from_base": "#1f77b4",
    "base_reconstruction_no_change": "#7f7f7f",
}
DIAGNOSIS_COLORS = {
    "CN": "#2ca02c",
    "AD": "#d62728",
    "MCI": "#ff7f0e",
}


def subject_summary_table(bundle: base.DirectFlowNotebookBundle) -> pd.DataFrame:
    metadata = bundle.contract.metadata.copy()
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    records: List[Dict[str, object]] = []
    sort_cols = ["split", "subject_id", "continuous_age_years", "visit_order"]
    for (split, subject_id), group in metadata.sort_values(sort_cols).groupby(
        ["split", "subject_id"],
        sort=False,
    ):
        diagnoses = sorted(group["diagnosis"].astype(str).unique().tolist())
        label_ads = sorted(group["label_ad"].astype(int).unique().tolist())
        first = group.iloc[0]
        last = group.iloc[-1]
        records.append(
            {
                "split": str(split),
                "subject_id": str(subject_id),
                "diagnosis": diagnoses[0] if diagnoses else "",
                "label_ad": int(label_ads[0]) if label_ads else -1,
                "num_scans": int(len(group)),
                "first_scan_id": str(first["scan_id"]),
                "last_scan_id": str(last["scan_id"]),
                "first_age_years": float(first["continuous_age_years"]),
                "last_age_years": float(last["continuous_age_years"]),
                "span_years": float(last["continuous_age_years"])
                - float(first["continuous_age_years"]),
                "diagnosis_values": ",".join(diagnoses),
                "label_ad_values": ",".join(str(value) for value in label_ads),
            }
        )
    return pd.DataFrame.from_records(records)


def select_balanced_subjects(
    bundle: base.DirectFlowNotebookBundle,
    *,
    subjects_per_split: int = 50,
    min_scans: int = 3,
    diagnoses: Sequence[str] = DIAGNOSIS_ORDER,
) -> pd.DataFrame:
    summary = subject_summary_table(bundle)
    per_diagnosis = int(subjects_per_split) // len(diagnoses)
    if per_diagnosis < 1:
        raise ValueError("subjects_per_split is too small for balanced selection.")

    selected: List[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        for diagnosis in diagnoses:
            pool = summary.loc[
                (summary["split"] == split)
                & (summary["diagnosis"] == diagnosis)
                & (summary["num_scans"] >= int(min_scans))
                & (summary["diagnosis_values"] == diagnosis)
            ].copy()
            if len(pool) < per_diagnosis:
                raise ValueError(
                    f"Need {per_diagnosis} {diagnosis} subjects with at least "
                    f"{min_scans} scans in split {split}, found {len(pool)}."
                )
            pool = pool.sort_values(
                ["span_years", "num_scans", "first_age_years", "subject_id"],
                ascending=[False, False, True, True],
            ).head(per_diagnosis)
            selected.append(pool)

    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["split", "diagnosis", "subject_id"]).reset_index(drop=True)
    result["selection_rank"] = (
        result.groupby(["split", "diagnosis"], sort=False).cumcount() + 1
    )
    return result


def selection_summary_table(selected_subjects: pd.DataFrame) -> pd.DataFrame:
    return (
        selected_subjects.groupby(["split", "diagnosis"], sort=False)
        .agg(
            subjects=("subject_id", "nunique"),
            mean_scans=("num_scans", "mean"),
            median_scans=("num_scans", "median"),
            mean_span_years=("span_years", "mean"),
            median_span_years=("span_years", "median"),
            min_span_years=("span_years", "min"),
            max_span_years=("span_years", "max"),
        )
        .reset_index()
    )


def selected_subject_id_map(selected_subjects: pd.DataFrame) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for split, group in selected_subjects.groupby("split", sort=False):
        mapping[str(split)] = group["subject_id"].astype(str).tolist()
    return mapping


def build_selected_observed_age_volume_trend_dataset(
    bundle: base.DirectFlowNotebookBundle,
    selected_subjects: pd.DataFrame,
    *,
    composed_step_years: float = 0.5,
    mesh_resolution: int = 80,
    mesh_max_batch: int = 2**18,
) -> Dict[str, pd.DataFrame]:
    trend_rows: List[Dict[str, object]] = []
    subject_rows_out: List[Dict[str, object]] = []
    decode_rows: List[Dict[str, object]] = []

    for _, selected in selected_subjects.iterrows():
        split = str(selected["split"])
        subject_id = str(selected["subject_id"])
        rows = base.subject_rows(bundle, subject_id, split=split)
        if rows.empty:
            continue
        base_row = rows.iloc[0]
        label_ad = int(base_row["label_ad"])
        diagnosis = str(base_row["diagnosis"])
        base_scan_id = str(base_row["scan_id"])
        base_age_years = float(base_row["continuous_age_years"])
        base_age_norm = float(base_row["continuous_age_norm"])
        base_latent = base._latent_tensor(bundle, split, base_scan_id)
        base_time = base._time_tensor(bundle, base_age_norm)
        condition = base._condition_tensor(bundle, label_ad)

        base_real_volume = base.mesh_volume(base_row["mesh_path"])
        base_recon_mesh, base_recon_error = base.try_decode_mesh(
            bundle,
            base_latent,
            resolution=mesh_resolution,
            max_batch=mesh_max_batch,
        )
        base_recon_volume = base.mesh_volume_or_nan(base_recon_mesh)

        per_subject_predicted: List[Tuple[float, float, bool]] = []
        per_subject_composed: List[Tuple[float, float, bool]] = []
        per_subject_real: List[Tuple[float, float]] = []
        direct_decode_success_count = 0
        composed_decode_success_count = 0

        for _, row in rows.iterrows():
            scan_id = str(row["scan_id"])
            age_years = float(row["continuous_age_years"])
            age_norm = float(row["continuous_age_norm"])
            years_from_baseline = age_years - base_age_years
            real_volume = base.mesh_volume(row["mesh_path"])
            per_subject_real.append((age_years, real_volume))

            if abs(age_norm - base_age_norm) <= 1e-8:
                predicted_mesh = base_recon_mesh
                predicted_error = base_recon_error
                composed_mesh = base_recon_mesh
                composed_error = base_recon_error
            else:
                predicted_latent = base.transport_direct(
                    bundle,
                    base_latent,
                    base_time,
                    base._time_tensor(bundle, age_norm),
                    condition,
                )
                predicted_mesh, predicted_error = base.try_decode_mesh(
                    bundle,
                    predicted_latent,
                    resolution=mesh_resolution,
                    max_batch=mesh_max_batch,
                )
                composed_latent = base.transport_composed_fixed_step(
                    bundle,
                    base_latent,
                    start_age_years=base_age_years,
                    end_age_years=age_years,
                    label_ad=label_ad,
                    step_years=float(composed_step_years),
                )
                composed_mesh, composed_error = base.try_decode_mesh(
                    bundle,
                    composed_latent,
                    resolution=mesh_resolution,
                    max_batch=mesh_max_batch,
                )
            predicted_volume = base.mesh_volume_or_nan(predicted_mesh)
            predicted_success = predicted_mesh is not None
            direct_decode_success_count += int(predicted_success)
            per_subject_predicted.append((age_years, predicted_volume, predicted_success))
            composed_volume = base.mesh_volume_or_nan(composed_mesh)
            composed_success = composed_mesh is not None
            composed_decode_success_count += int(composed_success)
            per_subject_composed.append((age_years, composed_volume, composed_success))

            real_relative = (
                real_volume / base_real_volume
                if np.isfinite(base_real_volume) and base_real_volume > 0.0
                else float("nan")
            )
            predicted_relative = (
                predicted_volume / base_recon_volume
                if np.isfinite(predicted_volume)
                and np.isfinite(base_recon_volume)
                and base_recon_volume > 0.0
                else float("nan")
            )
            composed_relative = (
                composed_volume / base_recon_volume
                if np.isfinite(composed_volume)
                and np.isfinite(base_recon_volume)
                and base_recon_volume > 0.0
                else float("nan")
            )

            common = {
                "split": split,
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "label_ad": label_ad,
                "scan_id": scan_id,
                "base_scan_id": base_scan_id,
                "visit_order": int(row["visit_order"]),
                "age_years": age_years,
                "years_from_baseline": years_from_baseline,
                "base_age_years": base_age_years,
            }
            trend_rows.append(
                {
                    **common,
                    "transport_method": "real_observed",
                    "volume": real_volume,
                    "relative_volume": real_relative,
                    "decode_success": True,
                }
            )
            trend_rows.append(
                {
                    **common,
                    "transport_method": "composed_from_base",
                    "volume": composed_volume,
                    "relative_volume": composed_relative,
                    "decode_success": composed_success,
                }
            )
            trend_rows.append(
                {
                    **common,
                    "transport_method": "direct_from_base",
                    "volume": predicted_volume,
                    "relative_volume": predicted_relative,
                    "decode_success": predicted_success,
                }
            )
            trend_rows.append(
                {
                    **common,
                    "transport_method": "base_reconstruction_no_change",
                    "volume": base_recon_volume,
                    "relative_volume": 1.0 if np.isfinite(base_recon_volume) else float("nan"),
                    "decode_success": base_recon_mesh is not None,
                }
            )
            decode_rows.append(
                {
                    "split": split,
                    "subject_id": subject_id,
                    "diagnosis": diagnosis,
                    "scan_id": scan_id,
                    "age_years": age_years,
                    "years_from_baseline": years_from_baseline,
                    "transport_method": "direct_from_base",
                    "decode_success": predicted_success,
                    "error": predicted_error,
                }
            )
            decode_rows.append(
                {
                    "split": split,
                    "subject_id": subject_id,
                    "diagnosis": diagnosis,
                    "scan_id": scan_id,
                    "age_years": age_years,
                    "years_from_baseline": years_from_baseline,
                    "transport_method": "composed_from_base",
                    "decode_success": composed_success,
                    "error": composed_error,
                }
            )

        final_real_age, final_real_volume = per_subject_real[-1]
        _, final_predicted_volume, final_predicted_success = per_subject_predicted[-1]
        _, final_composed_volume, final_composed_success = per_subject_composed[-1]
        span_years = final_real_age - base_age_years
        real_delta_pct = (
            100.0 * (final_real_volume - base_real_volume) / base_real_volume
            if base_real_volume > 0.0
            else float("nan")
        )
        predicted_delta_pct = (
            100.0 * (final_predicted_volume - base_recon_volume) / base_recon_volume
            if np.isfinite(final_predicted_volume)
            and np.isfinite(base_recon_volume)
            and base_recon_volume > 0.0
            else float("nan")
        )
        composed_delta_pct = (
            100.0 * (final_composed_volume - base_recon_volume) / base_recon_volume
            if np.isfinite(final_composed_volume)
            and np.isfinite(base_recon_volume)
            and base_recon_volume > 0.0
            else float("nan")
        )
        real_atrophy_pct_per_year = (
            100.0 * np.log(base_real_volume / final_real_volume) / span_years
            if span_years > 0.0 and base_real_volume > 0.0 and final_real_volume > 0.0
            else float("nan")
        )
        predicted_atrophy_pct_per_year = (
            100.0 * np.log(base_recon_volume / final_predicted_volume) / span_years
            if span_years > 0.0
            and np.isfinite(base_recon_volume)
            and np.isfinite(final_predicted_volume)
            and base_recon_volume > 0.0
            and final_predicted_volume > 0.0
            else float("nan")
        )
        composed_atrophy_pct_per_year = (
            100.0 * np.log(base_recon_volume / final_composed_volume) / span_years
            if span_years > 0.0
            and np.isfinite(base_recon_volume)
            and np.isfinite(final_composed_volume)
            and base_recon_volume > 0.0
            and final_composed_volume > 0.0
            else float("nan")
        )
        sign_match = (
            np.sign(real_delta_pct) == np.sign(predicted_delta_pct)
            if np.isfinite(real_delta_pct)
            and np.isfinite(predicted_delta_pct)
            and abs(real_delta_pct) > 1e-8
            else False
        )
        composed_sign_match = (
            np.sign(real_delta_pct) == np.sign(composed_delta_pct)
            if np.isfinite(real_delta_pct)
            and np.isfinite(composed_delta_pct)
            and abs(real_delta_pct) > 1e-8
            else False
        )
        subject_rows_out.append(
            {
                "split": split,
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "num_scans": int(len(rows)),
                "base_age_years": base_age_years,
                "final_age_years": final_real_age,
                "span_years": span_years,
                "base_real_volume": base_real_volume,
                "base_recon_volume": base_recon_volume,
                "base_recon_abs_error": abs(base_recon_volume - base_real_volume)
                if np.isfinite(base_recon_volume)
                else float("nan"),
                "final_real_volume": final_real_volume,
                "final_predicted_volume": final_predicted_volume,
                "final_predicted_decode_success": final_predicted_success,
                "final_abs_volume_error": abs(final_predicted_volume - final_real_volume)
                if np.isfinite(final_predicted_volume)
                else float("nan"),
                "final_composed_volume": final_composed_volume,
                "final_composed_decode_success": final_composed_success,
                "final_composed_abs_volume_error": abs(final_composed_volume - final_real_volume)
                if np.isfinite(final_composed_volume)
                else float("nan"),
                "real_delta_pct": real_delta_pct,
                "predicted_delta_pct": predicted_delta_pct,
                "composed_delta_pct": composed_delta_pct,
                "real_atrophy_pct_per_year": real_atrophy_pct_per_year,
                "predicted_atrophy_pct_per_year": predicted_atrophy_pct_per_year,
                "composed_atrophy_pct_per_year": composed_atrophy_pct_per_year,
                "final_delta_sign_match": bool(sign_match),
                "composed_final_delta_sign_match": bool(composed_sign_match),
                "decode_success_fraction": direct_decode_success_count
                / max(1, len(rows)),
                "composed_decode_success_fraction": composed_decode_success_count
                / max(1, len(rows)),
                "composed_step_years": float(composed_step_years),
            }
        )

    return {
        "trend_frame": pd.DataFrame(trend_rows),
        "subject_summary": pd.DataFrame(subject_rows_out),
        "decode_status": pd.DataFrame(decode_rows),
    }


def adjacent_volume_speed_frame(trend_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    required = [
        "split",
        "subject_id",
        "diagnosis",
        "transport_method",
        "visit_order",
        "age_years",
        "years_from_baseline",
        "volume",
    ]
    if trend_frame.empty or any(col not in trend_frame.columns for col in required):
        return pd.DataFrame()
    for keys, group in trend_frame.groupby(
        ["split", "subject_id", "diagnosis", "transport_method"],
        sort=False,
    ):
        split, subject_id, diagnosis, method = keys
        group = group.sort_values("visit_order")
        records = list(group.to_dict("records"))
        for source, target in zip(records[:-1], records[1:]):
            dt = float(target["age_years"]) - float(source["age_years"])
            if dt <= 0.0:
                continue
            source_volume = float(source["volume"])
            target_volume = float(target["volume"])
            if not (np.isfinite(source_volume) and np.isfinite(target_volume)):
                continue
            rows.append(
                {
                    "split": str(split),
                    "subject_id": str(subject_id),
                    "diagnosis": str(diagnosis),
                    "transport_method": str(method),
                    "source_visit_order": int(source["visit_order"]),
                    "target_visit_order": int(target["visit_order"]),
                    "source_age_years": float(source["age_years"]),
                    "target_age_years": float(target["age_years"]),
                    "midpoint_age_years": 0.5
                    * (float(source["age_years"]) + float(target["age_years"])),
                    "delta_years": dt,
                    "source_volume": source_volume,
                    "target_volume": target_volume,
                    "signed_volume_rate": (target_volume - source_volume) / dt,
                    "atrophy_volume_rate": (source_volume - target_volume) / dt,
                    "atrophy_pct_per_year": 100.0
                    * np.log(source_volume / target_volume)
                    / dt
                    if source_volume > 0.0 and target_volume > 0.0
                    else float("nan"),
                }
            )
    return pd.DataFrame.from_records(rows)


def speed_summary_table(speed_frame: pd.DataFrame) -> pd.DataFrame:
    if speed_frame.empty:
        return pd.DataFrame()
    return (
        speed_frame.groupby(["split", "diagnosis", "transport_method"], sort=False)
        .agg(
            intervals=("subject_id", "size"),
            subjects=("subject_id", "nunique"),
            mean_atrophy_pct_per_year=("atrophy_pct_per_year", "mean"),
            median_atrophy_pct_per_year=("atrophy_pct_per_year", "median"),
            mean_signed_volume_rate=("signed_volume_rate", "mean"),
            median_signed_volume_rate=("signed_volume_rate", "median"),
        )
        .reset_index()
    )


def prediction_consistency_table(subject_summary: pd.DataFrame) -> pd.DataFrame:
    if subject_summary.empty:
        return pd.DataFrame()
    return (
        subject_summary.groupby(["split", "diagnosis"], sort=False)
        .agg(
            subjects=("subject_id", "nunique"),
            mean_real_final_delta_pct=("real_delta_pct", "mean"),
            mean_predicted_final_delta_pct=("predicted_delta_pct", "mean"),
            mean_composed_final_delta_pct=("composed_delta_pct", "mean"),
            mean_real_atrophy_pct_per_year=("real_atrophy_pct_per_year", "mean"),
            mean_predicted_atrophy_pct_per_year=("predicted_atrophy_pct_per_year", "mean"),
            mean_composed_atrophy_pct_per_year=("composed_atrophy_pct_per_year", "mean"),
            final_delta_sign_match_fraction=("final_delta_sign_match", "mean"),
            composed_final_delta_sign_match_fraction=(
                "composed_final_delta_sign_match",
                "mean",
            ),
            mean_final_abs_volume_error=("final_abs_volume_error", "mean"),
            mean_final_composed_abs_volume_error=(
                "final_composed_abs_volume_error",
                "mean",
            ),
            mean_decode_success_fraction=("decode_success_fraction", "mean"),
            mean_composed_decode_success_fraction=(
                "composed_decode_success_fraction",
                "mean",
            ),
        )
        .reset_index()
    )


def make_selection_figure(selected_subjects: pd.DataFrame) -> go.Figure:
    summary = selection_summary_table(selected_subjects)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Selected subjects", "Selected follow-up span"),
    )
    for diagnosis in DIAGNOSIS_ORDER:
        group = summary.loc[summary["diagnosis"] == diagnosis]
        figure.add_trace(
            go.Bar(
                x=group["split"],
                y=group["subjects"],
                name=diagnosis,
                marker_color=DIAGNOSIS_COLORS[diagnosis],
                legendgroup=diagnosis,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=group["split"],
                y=group["mean_span_years"],
                name=f"{diagnosis} span",
                marker_color=DIAGNOSIS_COLORS[diagnosis],
                legendgroup=diagnosis,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    figure.update_yaxes(title_text="subjects", row=1, col=1)
    figure.update_yaxes(title_text="mean span years", row=1, col=2)
    figure.update_layout(
        title="Balanced 50-subject selection per split",
        barmode="group",
        template="plotly_white",
        width=1100,
        height=480,
    )
    return figure


def make_final_delta_comparison_figure(subject_summary: pd.DataFrame) -> go.Figure:
    summary = prediction_consistency_table(subject_summary)
    splits = [split for split in ("train", "val", "test") if split in set(summary["split"])]
    figure = make_subplots(
        rows=1,
        cols=len(splits),
        subplot_titles=[split.upper() for split in splits],
        shared_yaxes=True,
    )
    for col, split in enumerate(splits, start=1):
        group = summary.loc[summary["split"] == split].copy()
        for method, y_col, color in [
            ("real observed", "mean_real_final_delta_pct", "#111111"),
            ("composed prediction", "mean_composed_final_delta_pct", "#d62728"),
            ("direct prediction", "mean_predicted_final_delta_pct", "#1f77b4"),
        ]:
            figure.add_trace(
                go.Bar(
                    x=group["diagnosis"],
                    y=group[y_col],
                    name=method,
                    marker_color=color,
                    legendgroup=method,
                    showlegend=col == 1,
                    hovertemplate="diagnosis=%{x}<br>final delta=%{y:.3f}%<extra></extra>",
                ),
                row=1,
                col=col,
            )
    figure.update_yaxes(title_text="final volume change from baseline (%)", row=1, col=1)
    figure.update_layout(
        title="Final observed-vs-predicted volume change for selected subjects",
        barmode="group",
        template="plotly_white",
        width=430 * max(1, len(splits)),
        height=500,
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def _binned_mean_frame(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    bin_width: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "diagnosis",
                "transport_method",
                "x_bin",
                "mean_value",
                "std_value",
                "n",
            ]
        )
    data = frame.copy()
    data["x_bin"] = (
        np.round(data[x_col].astype(float) / float(bin_width)) * float(bin_width)
    )
    return (
        data.groupby(["split", "diagnosis", "transport_method", "x_bin"], sort=False)
        .agg(
            mean_value=(y_col, "mean"),
            std_value=(y_col, "std"),
            n=(y_col, "size"),
        )
        .reset_index()
    )


def _split_trend_figure(
    binned: pd.DataFrame,
    *,
    x_title: str,
    y_title: str,
    title: str,
) -> go.Figure:
    if binned.empty:
        return go.Figure()
    split_order = [split for split in ("train", "val", "test") if split in set(binned["split"])]
    style = {
        "real_observed": {"color": "#111111", "dash": "solid", "symbol": "diamond"},
        "composed_from_base": {"color": "#d62728", "dash": "dash", "symbol": "square"},
        "direct_from_base": {"color": "#1f77b4", "dash": "solid", "symbol": "circle"},
        "base_reconstruction_no_change": {
            "color": "#7f7f7f",
            "dash": "dot",
            "symbol": "x",
        },
    }
    figure = make_subplots(
        rows=1,
        cols=len(split_order),
        shared_yaxes=False,
        subplot_titles=[split.upper() for split in split_order],
    )
    for col, split in enumerate(split_order, start=1):
        split_frame = binned.loc[binned["split"] == split]
        for method in METHOD_ORDER:
            group = split_frame.loc[
                split_frame["transport_method"] == method
            ].sort_values("x_bin")
            if group.empty:
                continue
            spec = style[method]
            figure.add_trace(
                go.Scatter(
                    x=group["x_bin"],
                    y=group["mean_value"],
                    mode="lines+markers",
                    name=METHOD_LABELS[method],
                    legendgroup=method,
                    showlegend=(col == 1),
                    line={"color": spec["color"], "dash": spec["dash"], "width": 3},
                    marker={"color": spec["color"], "symbol": spec["symbol"], "size": 8},
                    customdata=np.stack([group["n"]], axis=-1),
                    hovertemplate=(
                        f"{x_title}=%{{x:.2f}}<br>"
                        f"{y_title}=%{{y:.6f}}<br>"
                        "n=%{customdata[0]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        figure.update_xaxes(title_text=x_title, row=1, col=col)
        figure.update_yaxes(title_text=y_title if col == 1 else None, row=1, col=col)
    figure.update_layout(
        title=title,
        width=max(1100, 430 * max(1, len(split_order))),
        height=520,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
    )
    return figure


def observed_age_volume_trend_figure(
    trend_frame: pd.DataFrame,
    *,
    age_bin_width: float = 1.0,
) -> go.Figure:
    binned = _binned_mean_frame(
        trend_frame,
        x_col="age_years",
        y_col="volume",
        bin_width=age_bin_width,
    )
    return _split_trend_figure(
        binned,
        x_title="age years",
        y_title="mean mesh volume",
        title="Observed-age volume trend: real vs composed/direct baseline forecast",
    )


def elapsed_relative_volume_trend_figure(
    trend_frame: pd.DataFrame,
    *,
    elapsed_bin_width: float = 0.5,
) -> go.Figure:
    binned = _binned_mean_frame(
        trend_frame,
        x_col="years_from_baseline",
        y_col="relative_volume",
        bin_width=elapsed_bin_width,
    )
    return _split_trend_figure(
        binned,
        x_title="years from baseline",
        y_title="mean relative volume",
        title="Within-subject relative volume trend: real vs composed/direct forecast",
    )


def make_speed_box_figure(speed_frame: pd.DataFrame) -> go.Figure:
    splits = [split for split in ("train", "val", "test") if split in set(speed_frame["split"])]
    figure = make_subplots(
        rows=1,
        cols=len(splits),
        subplot_titles=[split.upper() for split in splits],
        shared_yaxes=True,
    )
    usable_methods = ("real_observed", "composed_from_base", "direct_from_base")
    for col, split in enumerate(splits, start=1):
        split_frame = speed_frame.loc[
            (speed_frame["split"] == split)
            & (speed_frame["transport_method"].isin(usable_methods))
        ].copy()
        for diagnosis in DIAGNOSIS_ORDER:
            for method in usable_methods:
                group = split_frame.loc[
                    (split_frame["diagnosis"] == diagnosis)
                    & (split_frame["transport_method"] == method)
                ]
                if group.empty:
                    continue
                name = f"{diagnosis} {METHOD_LABELS[method]}"
                figure.add_trace(
                    go.Box(
                        y=group["atrophy_pct_per_year"],
                        name=name,
                        marker_color=METHOD_COLORS[method],
                        legendgroup=name,
                        showlegend=col == 1,
                        boxmean=True,
                        boxpoints=False,
                    ),
                    row=1,
                    col=col,
                )
    figure.update_yaxes(title_text="atrophy speed (%/year)", row=1, col=1)
    figure.update_layout(
        title="Adjacent-interval volume-change speed: real data vs composed/direct prediction",
        template="plotly_white",
        width=520 * max(1, len(splits)),
        height=560,
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def make_gap_bin_figure(gap_summary: pd.DataFrame) -> go.Figure:
    frame = gap_summary.loc[gap_summary["grouping"] == "split_gap_bin"].copy()
    if frame.empty:
        return go.Figure()
    frame["direct_relative_improvement_pct"] = (
        100.0
        * frame["sdf_l1_improvement_mean"].astype(float)
        / frame["no_change_target_sdf_l1_mean"].astype(float)
    )
    has_composed = "composed_sdf_l1_improvement_mean" in frame.columns
    if has_composed:
        frame["composed_relative_improvement_pct"] = (
            100.0
            * frame["composed_sdf_l1_improvement_mean"].astype(float)
            / frame["no_change_target_sdf_l1_mean"].astype(float)
        )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "SDF improvement by gap",
            "Beat no-change fraction by gap",
        ),
    )
    for split in ("train", "val", "test"):
        group = frame.loc[frame["split"] == split].sort_values("gap_bin")
        if group.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=group["gap_bin"],
                y=group["direct_relative_improvement_pct"],
                mode="lines+markers",
                name=f"{split} direct",
                legendgroup=f"{split}_direct",
            ),
            row=1,
            col=1,
        )
        if has_composed:
            figure.add_trace(
                go.Scatter(
                    x=group["gap_bin"],
                    y=group["composed_relative_improvement_pct"],
                    mode="lines+markers",
                    name=f"{split} composed",
                    legendgroup=f"{split}_composed",
                    line={"dash": "dash"},
                ),
                row=1,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=group["gap_bin"],
                y=group["model_beats_no_change_fraction"],
                mode="lines+markers",
                name=f"{split} direct beat",
                legendgroup=f"{split}_direct",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        if has_composed and "composed_beats_no_change_fraction" in group.columns:
            figure.add_trace(
                go.Scatter(
                    x=group["gap_bin"],
                    y=group["composed_beats_no_change_fraction"],
                    mode="lines+markers",
                    name=f"{split} composed beat",
                    legendgroup=f"{split}_composed",
                    showlegend=False,
                    line={"dash": "dash"},
                ),
                row=1,
                col=2,
            )
    figure.update_xaxes(title_text="gap bin", row=1, col=1)
    figure.update_xaxes(title_text="gap bin", row=1, col=2)
    figure.update_yaxes(title_text="relative SDF improvement (%)", row=1, col=1)
    figure.update_yaxes(title_text="fraction", row=1, col=2)
    figure.update_layout(
        title="Prediction quality by source-to-target time gap",
        template="plotly_white",
        width=1200,
        height=500,
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def make_mci_reference_speed_comparison_figure(
    selected_speed_frame: pd.DataFrame,
    mci_pair_frame: pd.DataFrame,
) -> go.Figure:
    rows: List[Dict[str, object]] = []
    if (
        not selected_speed_frame.empty
        and "transport_method" in selected_speed_frame.columns
        and "atrophy_pct_per_year" in selected_speed_frame.columns
    ):
        selected = selected_speed_frame.loc[
            selected_speed_frame["transport_method"].isin(
                ("real_observed", "composed_from_base", "direct_from_base")
            )
        ].copy()
        for (diagnosis, method), group in selected.groupby(
            ["diagnosis", "transport_method"],
            sort=False,
        ):
            rows.append(
                {
                    "diagnosis": diagnosis,
                    "source": f"large strict {METHOD_LABELS[method]}",
                    "mean_atrophy_pct_per_year": float(
                        group["atrophy_pct_per_year"].mean()
                    ),
                    "n": int(len(group)),
                }
            )
    for diagnosis, group in mci_pair_frame.groupby("diagnosis", sort=False):
        if diagnosis not in ("CN", "MCI", "AD"):
            continue
        rows.append(
            {
                "diagnosis": diagnosis,
                "source": "with-MCI real reference",
                "mean_atrophy_pct_per_year": float(group["atrophy_pct_per_year"].mean()),
                "n": int(len(group)),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    figure = go.Figure()
    source_order = [
        "large strict real observed",
        "large strict composed prediction",
        "large strict direct prediction",
        "with-MCI real reference",
    ]
    colors = {
        "large strict real observed": "#111111",
        "large strict composed prediction": "#d62728",
        "large strict direct prediction": "#1f77b4",
        "with-MCI real reference": "#ff7f0e",
    }
    for source in source_order:
        group = frame.loc[frame["source"] == source]
        if group.empty:
            continue
        figure.add_trace(
            go.Bar(
                x=group["diagnosis"],
                y=group["mean_atrophy_pct_per_year"],
                name=source,
                marker_color=colors[source],
                customdata=group[["n"]].to_numpy(),
                hovertemplate=(
                    "diagnosis=%{x}<br>mean atrophy=%{y:.3f}%/year"
                    "<br>intervals=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Volume-change speed trend: selected large-strict model vs with-MCI real-data reference",
        xaxis_title="diagnosis",
        yaxis_title="mean atrophy speed (%/year)",
        barmode="group",
        template="plotly_white",
        width=1050,
        height=520,
        legend={"orientation": "h", "y": -0.18},
    )
    return figure


def select_counterfactual_source_rows(
    bundle: base.DirectFlowNotebookBundle,
    selected_subjects: pd.DataFrame,
    *,
    per_split_per_diagnosis: int = 1,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    ranked = selected_subjects.sort_values(
        ["split", "diagnosis", "span_years", "num_scans", "subject_id"],
        ascending=[True, True, False, False, True],
    )
    for (split, diagnosis), group in ranked.groupby(["split", "diagnosis"], sort=False):
        for _, selected in group.head(int(per_split_per_diagnosis)).iterrows():
            rows = base.subject_rows(bundle, str(selected["subject_id"]), split=str(split))
            if rows.empty:
                continue
            source = rows.iloc[0].copy()
            records.append(
                {
                    "split": str(split),
                    "diagnosis": str(diagnosis),
                    "subject_id": str(selected["subject_id"]),
                    "source_scan_id": str(source["scan_id"]),
                    "source_age_years": float(source["continuous_age_years"]),
                    "span_years": float(selected["span_years"]),
                }
            )
    return pd.DataFrame.from_records(records)


def build_counterfactual_condition_case(
    bundle: base.DirectFlowNotebookBundle,
    *,
    split: str,
    subject_id: str,
    horizon_years: float = 10.0,
    evaluation_step_years: float = 2.0,
    mesh_resolution: int = 80,
    mesh_max_batch: int = 2**18,
) -> Dict[str, object]:
    rows = base.subject_rows(bundle, str(subject_id), split=str(split))
    if rows.empty:
        raise ValueError(f"No rows for split={split!r}, subject_id={subject_id!r}.")
    source_row = rows.iloc[0].copy()
    source_age = float(source_row["continuous_age_years"])
    source_scan_id = str(source_row["scan_id"])
    source_latent = base._latent_tensor(bundle, str(split), source_scan_id)
    source_time = base._time_tensor(bundle, float(source_row["continuous_age_norm"]))
    source_mesh = base.load_mesh(source_row["mesh_path"])
    source_volume = base.mesh_volume(source_mesh)

    age_grid = np.arange(
        source_age,
        source_age + float(horizon_years) + 1e-8,
        float(evaluation_step_years),
    )
    final_age = source_age + float(horizon_years)
    if age_grid[-1] < final_age - 1e-8:
        age_grid = np.append(age_grid, final_age)
    age_grid = base._unique_sorted_ages(age_grid.tolist())

    trend_rows: List[Dict[str, object]] = []
    decode_rows: List[Dict[str, object]] = []
    meshes: Dict[Tuple[int, str], object] = {}

    for label_ad, label in [(0, "CN_condition"), (1, "AD_condition")]:
        condition = base._condition_tensor(bundle, label_ad)
        for age in age_grid:
            key = base._age_key(age)
            if abs(float(age) - source_age) <= 1e-8:
                mesh = source_mesh
                error = None
            else:
                latent = base.transport_direct(
                    bundle,
                    source_latent,
                    source_time,
                    base._time_tensor(bundle, base.normalize_age_years(bundle, float(age))),
                    condition,
                )
                mesh, error = base.try_decode_mesh(
                    bundle,
                    latent,
                    resolution=mesh_resolution,
                    max_batch=mesh_max_batch,
                )
            meshes[(label_ad, key)] = mesh
            volume = base.mesh_volume_or_nan(mesh)
            trend_rows.append(
                {
                    "split": str(split),
                    "subject_id": str(subject_id),
                    "source_diagnosis": str(source_row["diagnosis"]),
                    "source_scan_id": source_scan_id,
                    "source_age_years": source_age,
                    "condition_label_ad": label_ad,
                    "condition_name": label,
                    "age_years": float(age),
                    "years_from_source": float(age) - source_age,
                    "volume": volume,
                    "relative_volume": volume / source_volume
                    if np.isfinite(volume) and source_volume > 0.0
                    else float("nan"),
                }
            )
            decode_rows.append(
                {
                    "split": str(split),
                    "subject_id": str(subject_id),
                    "source_diagnosis": str(source_row["diagnosis"]),
                    "condition_name": label,
                    "age_years": float(age),
                    "decode_success": mesh is not None,
                    "error": error,
                }
            )

    return {
        "split": str(split),
        "subject_id": str(subject_id),
        "source_row": source_row,
        "source_diagnosis": str(source_row["diagnosis"]),
        "source_age_years": source_age,
        "final_age_years": float(final_age),
        "source_mesh": source_mesh,
        "source_volume": source_volume,
        "age_grid": age_grid,
        "meshes": meshes,
        "trend_frame": pd.DataFrame.from_records(trend_rows),
        "decode_status": pd.DataFrame.from_records(decode_rows),
    }


def counterfactual_volume_figure(case: Mapping[str, object]) -> go.Figure:
    frame = case["trend_frame"].copy()
    figure = go.Figure()
    colors = {"CN_condition": "#2ca02c", "AD_condition": "#d62728"}
    for condition, group in frame.groupby("condition_name", sort=False):
        group = group.sort_values("age_years")
        figure.add_trace(
            go.Scatter(
                x=group["age_years"],
                y=group["volume"],
                mode="lines+markers",
                name=str(condition),
                line={"color": colors.get(str(condition), "#1f77b4"), "width": 3},
                marker={"size": 7},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[float(frame["age_years"].min()), float(frame["age_years"].max())],
            y=[float(case["source_volume"]), float(case["source_volume"])],
            mode="lines",
            name="source no-change",
            line={"color": "#7f7f7f", "dash": "dot", "width": 2},
        )
    )
    figure.update_layout(
        title=(
            f"Counterfactual condition forecast | {case['split']} subject "
            f"{case['subject_id']} ({case['source_diagnosis']})"
        ),
        xaxis_title="age years",
        yaxis_title="mesh volume",
        template="plotly_white",
        width=950,
        height=520,
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def counterfactual_final_change_figure(
    case: Mapping[str, object],
    *,
    sample_count: int = 3000,
) -> Tuple[go.Figure, pd.DataFrame]:
    final_key = base._age_key(float(case["final_age_years"]))
    comparisons = []
    for label_ad, label in [(0, "CN condition final"), (1, "AD condition final")]:
        mesh = case["meshes"].get((label_ad, final_key))
        if mesh is not None:
            comparisons.append((label, mesh))
    if not comparisons:
        message = "No final counterfactual meshes decoded successfully."
        return base.message_figure(
            title=message,
            message=message,
        ), pd.DataFrame(
            [
                {
                    "label": "none",
                    "mean_surface_shift": float("nan"),
                    "p90_surface_shift": float("nan"),
                    "p95_surface_shift": float("nan"),
                    "max_surface_shift": float("nan"),
                    "icp_cost": float("nan"),
                    "message": message,
                }
            ]
        )
    return base.change_heatmap_figure(
        case["source_mesh"],
        comparisons,
        title=(
            f"Counterfactual final change | {case['split']} subject {case['subject_id']} "
            f"({case['source_diagnosis']}) | source age {float(case['source_age_years']):.1f} "
            f"to {float(case['final_age_years']):.1f}"
        ),
        sample_count=sample_count,
    )


def save_tables(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def write_html_index(
    output_dir: Path,
    *,
    title: str,
    sections: Sequence[Tuple[str, Sequence[Path]]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    body: List[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.45;color:#222}",
        "h1{font-size:28px;margin-bottom:8px} h2{font-size:20px;margin-top:28px}",
        "li{margin:6px 0} code{background:#f4f4f4;padding:2px 4px;border-radius:3px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>Generated by the rich direct-flow visualization notebook. Large numeric trend tables cover the selected 50 subjects per split; 3D shape pages are representative examples only.</p>",
    ]
    for section_title, paths in sections:
        body.append(f"<h2>{html.escape(section_title)}</h2>")
        body.append("<ul>")
        for path in paths:
            rel = path.relative_to(output_dir) if path.is_absolute() and output_dir in path.parents else path.name
            body.append(
                f"<li><a href='{html.escape(str(rel))}'>{html.escape(path.name)}</a></li>"
            )
        body.append("</ul>")
    body.extend(["</body>", "</html>"])
    path = output_dir / "index.html"
    path.write_text("\n".join(body), encoding="utf-8")
    return path
