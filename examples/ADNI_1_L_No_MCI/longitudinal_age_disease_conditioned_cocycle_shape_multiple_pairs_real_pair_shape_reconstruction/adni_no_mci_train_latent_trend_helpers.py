from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
WITH_MCI_HELPER_DIR = REPO_ROOT / "examples" / "ADNI_1_L_With_MCI"
EXPERIMENT_DIR = Path(__file__).resolve().parent

for extra_path in (REPO_ROOT, WITH_MCI_HELPER_DIR, EXPERIMENT_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import adni_no_mci_longitudinal_model_helpers as model_helpers
import adni_no_mci_longitudinal_notebook_helpers as base_helpers


DEFAULT_CHECKPOINT = "1000"
DEFAULT_SELECTION_COUNT = 20
DEFAULT_SELECTION_SEED = 7
DEFAULT_AGE_BIN_WIDTH = 5.0
DEFAULT_MESH_RESOLUTION = 96
DEFAULT_METRIC_SURFACE_SAMPLES = 12000
DEFAULT_MAX_BATCH = 2 ** 17
DEFAULT_SWEEP_STEP_YEARS = 1.0
TRAINING_AGE_MAX_IN_DISTRIBUTION = model_helpers.TRAINING_AGE_MAX


def _age_bin_left(age_years: float, bin_width_years: float = DEFAULT_AGE_BIN_WIDTH) -> float:
    return float(bin_width_years) * math.floor(float(age_years) / float(bin_width_years))


def _age_bin_label(age_left: float, bin_width_years: float = DEFAULT_AGE_BIN_WIDTH) -> str:
    age_right = float(age_left) + float(bin_width_years)
    return f"{int(round(age_left)):02d}-{int(round(age_right)):02d}"


def _add_age_bin_columns(
    frame: pd.DataFrame,
    age_col: str,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
) -> pd.DataFrame:
    out = frame.copy()
    out["age_bin_left"] = out[age_col].map(
        lambda value: _age_bin_left(value, bin_width_years=bin_width_years)
    )
    out["age_bin_label"] = out["age_bin_left"].map(
        lambda value: _age_bin_label(value, bin_width_years=bin_width_years)
    )
    out["age_bin_center"] = out["age_bin_left"] + 0.5 * float(bin_width_years)
    return out


@lru_cache(maxsize=8)
def _cached_bundle(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
):
    return model_helpers.load_model_bundle(EXPERIMENT_DIR, checkpoint, device=device)


@lru_cache(maxsize=1)
def _records_df() -> pd.DataFrame:
    return base_helpers.load_records_df().copy()


@lru_cache(maxsize=1)
def _train_records_df() -> pd.DataFrame:
    return model_helpers.split_frame(_records_df(), "train").copy()


@lru_cache(maxsize=1)
def _train_subject_groups() -> Dict[str, pd.DataFrame]:
    return model_helpers.grouped_subject_rows(_train_records_df())


@lru_cache(maxsize=8)
def _train_anchor_index_frame(checkpoint: str = DEFAULT_CHECKPOINT) -> pd.DataFrame:
    bundle = _cached_bundle(checkpoint, "cpu")
    if bundle.train_latents is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint} does not have train latents loaded under {EXPERIMENT_DIR}."
        )
    subject_ids = sorted(_train_records_df()["subject_id"].astype(str).unique().tolist())
    if len(subject_ids) != int(bundle.train_latents.shape[0]):
        raise RuntimeError(
            "Train subject / anchor count mismatch: "
            f"{len(subject_ids)} subjects vs {int(bundle.train_latents.shape[0])} anchor rows"
        )
    rows: List[Dict[str, object]] = []
    grouped = _train_subject_groups()
    for idx, subject_id in enumerate(subject_ids):
        group = grouped[str(subject_id)].sort_values(["visit_order", "continuous_age_norm"]).reset_index(drop=True)
        first_row = group.iloc[0]
        rows.append(
            {
                "latent_index": int(idx),
                "subject_id": str(subject_id),
                "baseline_time": float(group["continuous_age_norm"].min()),
                "baseline_age_years": float(first_row["baseline_age_years"]),
                "first_diagnosis": str(first_row["diagnosis"]),
                "num_scans": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["subject_id"]).reset_index(drop=True)


def load_train_subject_anchor(
    subject_id: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
) -> torch.Tensor:
    bundle = _cached_bundle(checkpoint, device)
    meta = _train_anchor_index_frame(checkpoint).set_index("subject_id", drop=False)
    if str(subject_id) not in meta.index:
        raise KeyError(f"subject_id is not part of the training anchor table: {subject_id}")
    idx = int(meta.loc[str(subject_id)]["latent_index"])
    return bundle.train_latents[idx : idx + 1].to(bundle.device)


@lru_cache(maxsize=1)
def train_subject_overview_frame() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for subject_id, group in _train_subject_groups().items():
        ordered = group.sort_values(["visit_order", "continuous_age_norm"]).reset_index(drop=True)
        first_row = ordered.iloc[0]
        last_row = ordered.iloc[-1]
        rows.append(
            {
                "subject_id": str(subject_id),
                "num_scans": int(len(ordered)),
                "baseline_age_years": float(first_row["baseline_age_years"]),
                "first_observed_age_years": float(first_row["continuous_age_years"]),
                "last_observed_age_years": float(last_row["continuous_age_years"]),
                "age_span_years": float(
                    float(last_row["continuous_age_years"]) - float(first_row["continuous_age_years"])
                ),
                "first_diagnosis": str(first_row["diagnosis"]),
                "last_diagnosis": str(last_row["diagnosis"]),
                "ever_ad": bool(int(ordered["label_ad"].max()) == 1),
            }
        )
    out = pd.DataFrame(rows)
    out = _add_age_bin_columns(
        out,
        age_col="baseline_age_years",
        bin_width_years=DEFAULT_AGE_BIN_WIDTH,
    )
    return out.sort_values(["age_bin_left", "baseline_age_years", "subject_id"]).reset_index(drop=True)


@lru_cache(maxsize=32)
def selected_train_subjects(
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
) -> pd.DataFrame:
    subject_df = train_subject_overview_frame().copy()
    subject_df = _add_age_bin_columns(
        subject_df,
        age_col="baseline_age_years",
        bin_width_years=bin_width_years,
    )
    rng = np.random.default_rng(int(seed))
    bin_to_subjects: Dict[str, List[str]] = {}
    for label, group in subject_df.groupby("age_bin_label", sort=True):
        subject_ids = group["subject_id"].astype(str).tolist()
        rng.shuffle(subject_ids)
        bin_to_subjects[str(label)] = subject_ids

    ordered_bins = sorted(bin_to_subjects.keys())
    chosen: List[str] = []
    while len(chosen) < int(n_subjects):
        progressed = False
        for label in ordered_bins:
            if not bin_to_subjects[label]:
                continue
            chosen.append(bin_to_subjects[label].pop())
            progressed = True
            if len(chosen) >= int(n_subjects):
                break
        if not progressed:
            break

    selected = (
        subject_df.loc[subject_df["subject_id"].isin(chosen)]
        .copy()
        .assign(selection_seed=int(seed), selection_rank=lambda df: df["subject_id"].map({sid: i for i, sid in enumerate(chosen)}))
        .sort_values(["selection_rank", "subject_id"])
        .reset_index(drop=True)
    )
    return selected


def selected_train_scan_frame(
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
) -> pd.DataFrame:
    selected_ids = selected_train_subjects(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )["subject_id"].tolist()
    frame = _train_records_df().copy()
    frame = frame.loc[frame["subject_id"].isin(selected_ids)].copy()
    frame["volume"] = frame["mesh_path"].map(model_helpers.mesh_volume)
    return _add_age_bin_columns(
        frame.sort_values(["subject_id", "visit_order"]).reset_index(drop=True),
        age_col="continuous_age_years",
        bin_width_years=bin_width_years,
    )


def selected_train_subject_overview(
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
) -> Dict[str, object]:
    subject_df = selected_train_subjects(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    ).copy()
    scan_df = selected_train_scan_frame(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    ).copy()
    age_bin_counts = (
        subject_df.groupby(["age_bin_label", "first_diagnosis"], sort=True)["subject_id"]
        .nunique()
        .reset_index(name="num_subjects")
    )
    return {
        "subject_df": subject_df,
        "scan_df": scan_df,
        "age_bin_counts_df": age_bin_counts,
    }


def selected_train_overview_figures(
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
) -> List[go.Figure]:
    payload = selected_train_subject_overview(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )
    age_counts = payload["age_bin_counts_df"].copy()
    scan_df = payload["scan_df"].copy()

    counts_fig = go.Figure()
    for diagnosis, group in age_counts.groupby("first_diagnosis", sort=True):
        counts_fig.add_trace(
            go.Bar(
                x=group["age_bin_label"],
                y=group["num_subjects"],
                name=str(diagnosis),
            )
        )
    counts_fig.update_layout(
        title=f"Selected training subjects by 5-year baseline age bin (n={n_subjects})",
        xaxis_title="Baseline age bin (years)",
        yaxis_title="Number of selected subjects",
        barmode="group",
        template="plotly_white",
    )

    volume_fig = model_helpers.trajectory_figure(
        scan_df,
        x_col="continuous_age_years",
        y_col="volume",
        color_col="diagnosis",
        line_group_col="subject_id",
        title="Observed training-set longitudinal volume trajectories for selected subjects",
    )
    volume_fig.update_layout(
        xaxis_title="Observed age (years)",
        yaxis_title="Mesh volume",
    )

    return [counts_fig, volume_fig]


def _probe_decoder_range(
    bundle,
    latent: torch.Tensor,
    grid_n: int = 14,
    cube_half_extent: float = 1.0,
) -> Dict[str, float]:
    lin = np.linspace(
        -float(cube_half_extent),
        float(cube_half_extent),
        int(grid_n),
        dtype=np.float32,
    )
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    points = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)
    pts_t = torch.from_numpy(points).to(bundle.device)
    with torch.no_grad():
        z_rep = latent.expand(pts_t.shape[0], -1)
        sdf = bundle.decoder(torch.cat([z_rep, pts_t], dim=1)).squeeze(1)
    sdf_min = float(sdf.min().item())
    sdf_max = float(sdf.max().item())
    return {
        "probe_sdf_min": sdf_min,
        "probe_sdf_max": sdf_max,
        "probe_zero_cross": bool(sdf_min <= 0.0 <= sdf_max),
        "latent_l2_norm": float(latent.norm().item()),
        "latent_abs_max": float(latent.abs().max().item()),
    }


def _safe_decode_mesh(
    bundle,
    latent: torch.Tensor,
    resolution: int = DEFAULT_MESH_RESOLUTION,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> Dict[str, object]:
    probe = _probe_decoder_range(bundle, latent)
    try:
        mesh_obj = model_helpers.decode_mesh(
            bundle,
            latent,
            resolution=resolution,
            max_batch=max_batch,
        )
        return {
            "mesh_valid": True,
            "mesh": mesh_obj,
            "mesh_error": "",
            "pred_volume": model_helpers.mesh_volume(mesh_obj),
            **probe,
        }
    except Exception as exc:  # pragma: no cover - depends on decoded geometry
        return {
            "mesh_valid": False,
            "mesh": None,
            "mesh_error": str(exc),
            "pred_volume": float("nan"),
            **probe,
        }


def _compose_latent_over_rows(
    bundle,
    start_latent: torch.Tensor,
    source_row: pd.Series,
    future_rows: Sequence[pd.Series],
) -> torch.Tensor:
    latent_now = start_latent
    current_time = float(source_row["continuous_age_norm"])
    for row in future_rows:
        next_time = float(row["continuous_age_norm"])
        next_label = int(row["label_ad"])
        latent_now = model_helpers.transport_direct(
            bundle,
            latent_now,
            baseline_time=current_time,
            target_time=next_time,
            target_label_ad=next_label,
        )
        current_time = next_time
    return latent_now


def _compose_latent_uniform(
    bundle,
    start_latent: torch.Tensor,
    start_age_years: float,
    final_age_years: float,
    label_ad: int,
    step_years: float = DEFAULT_SWEEP_STEP_YEARS,
) -> torch.Tensor:
    if abs(float(final_age_years) - float(start_age_years)) <= 1e-12:
        return start_latent
    latent_now = start_latent
    current_age = float(start_age_years)
    while current_age + 1e-8 < float(final_age_years):
        next_age = min(current_age + float(step_years), float(final_age_years))
        latent_now = model_helpers.transport_direct(
            bundle,
            latent_now,
            baseline_time=(current_age - model_helpers.TRAINING_AGE_MIN)
            / model_helpers.TRAINING_AGE_RANGE_YEARS,
            target_time=(next_age - model_helpers.TRAINING_AGE_MIN)
            / model_helpers.TRAINING_AGE_RANGE_YEARS,
            target_label_ad=int(label_ad),
        )
        current_age = next_age
    return latent_now


def _selected_subject_groups(
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
) -> Dict[str, pd.DataFrame]:
    selected_ids = selected_train_subjects(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )["subject_id"].tolist()
    all_groups = _train_subject_groups()
    return {subject_id: all_groups[subject_id] for subject_id in selected_ids}


@lru_cache(maxsize=16)
def evaluate_selected_train_forecasts(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    mesh_resolution: int = DEFAULT_MESH_RESOLUTION,
    metric_surface_samples: int = DEFAULT_METRIC_SURFACE_SAMPLES,
) -> Dict[str, object]:
    bundle = _cached_bundle(checkpoint, device)
    subject_groups = _selected_subject_groups(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )
    rows: List[Dict[str, object]] = []
    for subject_id, subject_rows in subject_groups.items():
        ordered = subject_rows.sort_values(["visit_order", "continuous_age_norm"]).reset_index(drop=True)
        baseline_time = float(ordered["continuous_age_norm"].min())
        subject_anchor = load_train_subject_anchor(
            str(subject_id),
            checkpoint=checkpoint,
            device=device,
        )
        for source_idx in range(len(ordered) - 1):
            source_row = ordered.iloc[source_idx]
            source_latent = model_helpers.transport_direct(
                bundle,
                subject_anchor,
                baseline_time=baseline_time,
                target_time=float(source_row["continuous_age_norm"]),
                target_label_ad=int(source_row["label_ad"]),
            )
            source_mesh = model_helpers.load_mesh(source_row["mesh_path"])
            source_volume = model_helpers.mesh_volume(source_mesh)
            source_age = float(source_row["continuous_age_years"])
            source_time = float(source_row["continuous_age_norm"])
            source_diag = str(source_row["diagnosis"])
            for target_idx in range(source_idx + 1, len(ordered)):
                target_row = ordered.iloc[target_idx]
                target_state_latent = model_helpers.transport_direct(
                    bundle,
                    subject_anchor,
                    baseline_time=baseline_time,
                    target_time=float(target_row["continuous_age_norm"]),
                    target_label_ad=int(target_row["label_ad"]),
                )
                target_mesh = model_helpers.load_mesh(target_row["mesh_path"])
                target_age = float(target_row["continuous_age_years"])
                target_time = float(target_row["continuous_age_norm"])
                target_diag = str(target_row["diagnosis"])
                visit_gap = int(target_row["visit_order"]) - int(source_row["visit_order"])
                pair_kind = "adjacent" if visit_gap == 1 else "non_adjacent"
                target_label_ad = int(target_row["label_ad"])

                predicted_latents: Dict[str, torch.Tensor] = {
                    "direct": model_helpers.transport_direct(
                        bundle,
                        source_latent,
                        baseline_time=source_time,
                        target_time=target_time,
                        target_label_ad=target_label_ad,
                    ),
                    "composed": _compose_latent_over_rows(
                        bundle,
                        source_latent,
                        source_row,
                        [ordered.iloc[i] for i in range(source_idx + 1, target_idx + 1)],
                    ),
                    "no_change": source_latent,
                }

                for method, pred_latent in predicted_latents.items():
                    decode_info = (
                        {
                            "mesh_valid": True,
                            "mesh": source_mesh,
                            "mesh_error": "",
                            "pred_volume": source_volume,
                            **_probe_decoder_range(bundle, pred_latent),
                        }
                        if method == "no_change"
                        else _safe_decode_mesh(
                            bundle,
                            pred_latent,
                            resolution=mesh_resolution,
                            max_batch=DEFAULT_MAX_BATCH,
                        )
                    )
                    metrics: Dict[str, float]
                    if decode_info["mesh_valid"]:
                        metrics = model_helpers.deterministic_mesh_metrics(
                            target_mesh,
                            decode_info["mesh"],
                            num_samples=metric_surface_samples,
                            seed=0,
                            align_mode=bundle.align_mode,
                            align_iters=bundle.align_iters,
                            align_trim_quantile=bundle.align_trim_quantile,
                        )
                        pred_minus_target_volume = (
                            float(metrics["pred_volume"]) - float(metrics["target_volume"])
                        )
                    else:
                        metrics = {
                            "chamfer_aligned": float("nan"),
                            "assd_aligned": float("nan"),
                            "hd95_aligned": float("nan"),
                            "pred_volume": float("nan"),
                            "target_volume": model_helpers.mesh_volume(target_mesh),
                            "pred_point_count": 0,
                            "target_point_count": 0,
                        }
                        pred_minus_target_volume = float("nan")

                    rows.append(
                        {
                            "subject_id": str(subject_id),
                            "diagnosis": target_diag,
                            "source_diagnosis": source_diag,
                            "target_diagnosis": target_diag,
                            "source_scan_id": str(source_row["scan_id"]),
                            "target_scan_id": str(target_row["scan_id"]),
                            "source_visit_order": int(source_row["visit_order"]),
                            "target_visit_order": int(target_row["visit_order"]),
                            "source_age_years": source_age,
                            "target_age_years": target_age,
                            "horizon_years": target_age - source_age,
                            "pair_kind": pair_kind,
                            "visit_gap": int(visit_gap),
                            "method": method,
                            "source_label_ad": int(source_row["label_ad"]),
                            "target_label_ad": target_label_ad,
                            "source_mesh_path": str(source_row["mesh_path"]),
                            "target_mesh_path": str(target_row["mesh_path"]),
                            "mesh_valid": bool(decode_info["mesh_valid"]),
                            "mesh_error": str(decode_info["mesh_error"]),
                            "probe_sdf_min": float(decode_info["probe_sdf_min"]),
                            "probe_sdf_max": float(decode_info["probe_sdf_max"]),
                            "probe_zero_cross": bool(decode_info["probe_zero_cross"]),
                            "latent_l2_norm": float(decode_info["latent_l2_norm"]),
                            "latent_abs_max": float(decode_info["latent_abs_max"]),
                            "latent_l2_to_target_state": float(
                                torch.norm(pred_latent - target_state_latent).item()
                            ),
                            "pred_minus_target_volume": pred_minus_target_volume,
                            **metrics,
                        }
                    )

    forecast_df = pd.DataFrame(rows).sort_values(
        ["subject_id", "source_visit_order", "target_visit_order", "method"]
    ).reset_index(drop=True)

    metric_cols = [
        "chamfer_aligned",
        "assd_aligned",
        "hd95_aligned",
        "pred_minus_target_volume",
        "latent_l2_to_target_state",
    ]
    summary_frames: List[pd.DataFrame] = []
    for method in ("direct", "composed", "no_change"):
        for pair_kind in ("all", "adjacent", "non_adjacent"):
            subset = forecast_df.loc[forecast_df["method"] == method].copy()
            if pair_kind != "all":
                subset = subset.loc[subset["pair_kind"] == pair_kind]
            if subset.empty:
                continue
            metrics_subset = subset.copy()
            summary = model_helpers.compute_group_summary(
                metrics_subset,
                metric_cols=metric_cols,
                diagnosis_col="diagnosis",
                subject_col="subject_id",
                bootstrap_iterations=1000,
                seed=11,
            )
            summary.insert(0, "pair_kind", pair_kind)
            summary.insert(0, "method", method)
            valid_fraction_by_cohort = {}
            for cohort in summary["cohort"].tolist():
                cohort_subset = subset if cohort == "all" else subset.loc[subset["diagnosis"] == cohort]
                valid_fraction_by_cohort[cohort] = float(
                    cohort_subset["mesh_valid"].mean() if len(cohort_subset) > 0 else float("nan")
                )
            summary["mesh_valid_fraction"] = summary["cohort"].map(valid_fraction_by_cohort)
            summary_frames.append(summary)
    summary_df = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )

    return {
        "forecast_df": forecast_df,
        "summary_df": summary_df,
    }


def train_forecast_summary_figures(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
) -> List[go.Figure]:
    payload = evaluate_selected_train_forecasts(
        checkpoint=checkpoint,
        device=device,
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )
    summary_df = payload["summary_df"].copy()
    forecast_df = payload["forecast_df"].copy()

    chamfer_fig = go.Figure()
    subset = summary_df.loc[
        (summary_df["cohort"] == "all") & (summary_df["pair_kind"].isin(["adjacent", "non_adjacent"]))
    ]
    for method, group in subset.groupby("method", sort=True):
        chamfer_fig.add_trace(
            go.Bar(
                x=group["pair_kind"],
                y=group["chamfer_aligned_mean"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=group["chamfer_aligned_ci_high"] - group["chamfer_aligned_mean"],
                    arrayminus=group["chamfer_aligned_mean"] - group["chamfer_aligned_ci_low"],
                ),
                name=str(method),
            )
        )
    chamfer_fig.update_layout(
        title="Train-set existing-visit forecast error",
        xaxis_title="Pair type",
        yaxis_title="Aligned Chamfer",
        barmode="group",
        template="plotly_white",
    )

    valid_direct = forecast_df.loc[
        (forecast_df["method"] == "direct") & forecast_df["mesh_valid"]
    ].copy()
    valid_no_change = forecast_df.loc[forecast_df["method"] == "no_change"].copy()
    direct_vs_static = valid_direct.merge(
        valid_no_change[
            [
                "subject_id",
                "source_scan_id",
                "target_scan_id",
                "chamfer_aligned",
            ]
        ],
        on=["subject_id", "source_scan_id", "target_scan_id"],
        suffixes=("_direct", "_no_change"),
    )
    direct_vs_static_fig = go.Figure()
    for diagnosis, group in direct_vs_static.groupby("diagnosis", sort=True):
        direct_vs_static_fig.add_trace(
            go.Scatter(
                x=group["chamfer_aligned_no_change"],
                y=group["chamfer_aligned_direct"],
                mode="markers",
                name=str(diagnosis),
                text=group["subject_id"],
            )
        )
    if not direct_vs_static.empty:
        lo = float(
            min(
                direct_vs_static["chamfer_aligned_no_change"].min(),
                direct_vs_static["chamfer_aligned_direct"].min(),
            )
        )
        hi = float(
            max(
                direct_vs_static["chamfer_aligned_no_change"].max(),
                direct_vs_static["chamfer_aligned_direct"].max(),
            )
        )
        direct_vs_static_fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="y=x",
                line=dict(color="black", dash="dash"),
            )
        )
    direct_vs_static_fig.update_layout(
        title="Train-set direct forecast versus no-change baseline",
        xaxis_title="No-change aligned Chamfer",
        yaxis_title="Direct aligned Chamfer",
        template="plotly_white",
    )

    latent_fig = go.Figure()
    for method, group in forecast_df.groupby("method", sort=True):
        latent_fig.add_trace(
            go.Box(
                y=group["latent_l2_to_target_state"],
                name=str(method),
                boxmean=True,
            )
        )
    latent_fig.update_layout(
        title="Train-set latent prediction error to the learned target state",
        yaxis_title="Latent L2 to anchor-derived target state",
        template="plotly_white",
    )

    return [chamfer_fig, direct_vs_static_fig, latent_fig]


def representative_train_forecast_figures(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    mesh_resolution: int = DEFAULT_MESH_RESOLUTION,
) -> List[go.Figure]:
    payload = evaluate_selected_train_forecasts(
        checkpoint=checkpoint,
        device=device,
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
        mesh_resolution=mesh_resolution,
    )
    forecast_df = payload["forecast_df"].copy()
    bundle = _cached_bundle(checkpoint, device)
    figures: List[go.Figure] = []
    for diagnosis in ("CN", "AD"):
        group = forecast_df.loc[
            (forecast_df["diagnosis"] == diagnosis)
            & (forecast_df["method"] == "direct")
            & forecast_df["mesh_valid"]
        ].copy()
        if group.empty:
            continue
        rep_row = group.sort_values(["horizon_years", "chamfer_aligned"], ascending=[False, True]).iloc[0]
        subject_id = str(rep_row["subject_id"])
        subject_group = _train_subject_groups()[subject_id].sort_values(
            ["visit_order", "continuous_age_norm"]
        ).reset_index(drop=True)
        baseline_time = float(subject_group["continuous_age_norm"].min())
        source_row = _train_records_df().loc[
            _train_records_df()["scan_id"] == str(rep_row["source_scan_id"])
        ].iloc[0]
        target_row = _train_records_df().loc[
            _train_records_df()["scan_id"] == str(rep_row["target_scan_id"])
        ].iloc[0]
        subject_anchor = load_train_subject_anchor(
            subject_id,
            checkpoint=checkpoint,
            device=device,
        )
        source_latent = model_helpers.transport_direct(
            bundle,
            subject_anchor,
            baseline_time=baseline_time,
            target_time=float(source_row["continuous_age_norm"]),
            target_label_ad=int(source_row["label_ad"]),
        )
        target_time = float(target_row["continuous_age_norm"])
        direct_latent = model_helpers.transport_direct(
            bundle,
            source_latent,
            baseline_time=float(source_row["continuous_age_norm"]),
            target_time=target_time,
            target_label_ad=int(target_row["label_ad"]),
        )
        composed_latent = _compose_latent_over_rows(
            bundle,
            source_latent,
            source_row,
            [
                row
                for _, row in _train_subject_groups()[subject_id]
                .loc[
                    (_train_subject_groups()[subject_id]["visit_order"] > int(source_row["visit_order"]))
                    & (_train_subject_groups()[subject_id]["visit_order"] <= int(target_row["visit_order"]))
                ]
                .sort_values("visit_order")
                .iterrows()
            ],
        )
        direct_mesh = _safe_decode_mesh(
            bundle,
            direct_latent,
            resolution=mesh_resolution,
        )["mesh"]
        composed_mesh = _safe_decode_mesh(
            bundle,
            composed_latent,
            resolution=mesh_resolution,
        )["mesh"]
        figures.append(
            base_helpers.overlay_mesh_figure(
                [
                    ("source", source_row["mesh_path"], "#7f7f7f", 0.22),
                    ("ground truth", target_row["mesh_path"], "#2ca02c", 0.40),
                    ("direct", direct_mesh, "#d62728", 0.45),
                ],
                title=(
                    f"Train-set {diagnosis} example {subject_id}: direct forecast to existing future shape "
                    f"({float(rep_row['horizon_years']):.1f} years)"
                ),
            )
        )
        if composed_mesh is not None:
            figures.append(
                base_helpers.overlay_mesh_figure(
                    [
                        ("source", source_row["mesh_path"], "#7f7f7f", 0.22),
                        ("ground truth", target_row["mesh_path"], "#2ca02c", 0.40),
                        ("composed", composed_mesh, "#9467bd", 0.45),
                    ],
                    title=(
                        f"Train-set {diagnosis} example {subject_id}: composed forecast to existing future shape "
                        f"({float(rep_row['horizon_years']):.1f} years)"
                    ),
                )
            )
    return figures


@lru_cache(maxsize=16)
def compute_selected_train_velocities(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    velocity_method: str = "finite_difference",
) -> Dict[str, object]:
    bundle = _cached_bundle(checkpoint, device)
    scan_df = selected_train_scan_frame(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )
    rows: List[Dict[str, object]] = []
    for _, row in scan_df.iterrows():
        anchor = load_train_subject_anchor(
            str(row["subject_id"]),
            checkpoint=checkpoint,
            device=device,
        )
        subject_group = _train_subject_groups()[str(row["subject_id"])]
        baseline_time = float(subject_group["continuous_age_norm"].min())
        latent = model_helpers.transport_direct(
            bundle,
            anchor,
            baseline_time=baseline_time,
            target_time=float(row["continuous_age_norm"]),
            target_label_ad=int(row["label_ad"]),
        )
        mesh_obj = model_helpers.load_mesh(row["mesh_path"])
        vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
        speed = model_helpers.implicit_surface_normal_velocity(
            bundle,
            latent,
            float(row["continuous_age_norm"]),
            int(row["label_ad"]),
            vertices,
            yearly=True,
            method=velocity_method,
        )
        rows.append(
            {
                "subject_id": str(row["subject_id"]),
                "scan_id": str(row["scan_id"]),
                "diagnosis": str(row["diagnosis"]),
                "visit_order": int(row["visit_order"]),
                "age_years": float(row["continuous_age_years"]),
                "label_ad": int(row["label_ad"]),
                "volume": model_helpers.mesh_volume(mesh_obj),
                **model_helpers.summarize_speed_map(mesh_obj, speed),
            }
        )
    velocity_df = _add_age_bin_columns(
        pd.DataFrame(rows).sort_values(["subject_id", "visit_order"]).reset_index(drop=True),
        age_col="age_years",
        bin_width_years=bin_width_years,
    )

    summary_rows: List[Dict[str, object]] = []
    for age_bin_center, group in velocity_df.groupby("age_bin_center", sort=True):
        for diagnosis in ("all", "CN", "AD"):
            subset = group if diagnosis == "all" else group.loc[group["diagnosis"] == diagnosis]
            if subset.empty:
                continue
            summary_rows.append(
                {
                    "age_bin_center": float(age_bin_center),
                    "diagnosis": diagnosis,
                    "num_rows": int(len(subset)),
                    "num_subjects": int(subset["subject_id"].nunique()),
                    "area_weighted_rms_speed_per_year_mean": float(
                        subset["area_weighted_rms_speed_per_year"].mean()
                    ),
                    "mean_absolute_speed_per_year_mean": float(
                        subset["mean_absolute_speed_per_year"].mean()
                    ),
                    "net_volume_rate_per_year_mean": float(
                        subset["net_volume_rate_per_year"].mean()
                    ),
                }
            )
    age_bin_df = pd.DataFrame(summary_rows)
    return {
        "velocity_df": velocity_df,
        "age_bin_df": age_bin_df,
    }


def train_velocity_figures(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    velocity_method: str = "finite_difference",
) -> List[go.Figure]:
    payload = compute_selected_train_velocities(
        checkpoint=checkpoint,
        device=device,
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
        velocity_method=velocity_method,
    )
    velocity_df = payload["velocity_df"].copy()
    age_bin_df = payload["age_bin_df"].copy()
    figures: List[go.Figure] = []

    rms_fig = go.Figure()
    for diagnosis in ("CN", "AD", "all"):
        group = age_bin_df.loc[age_bin_df["diagnosis"] == diagnosis]
        if group.empty:
            continue
        rms_fig.add_trace(
            go.Scatter(
                x=group["age_bin_center"],
                y=group["area_weighted_rms_speed_per_year_mean"],
                mode="lines+markers",
                name=diagnosis,
            )
        )
    rms_fig.update_layout(
        title="Training-set model velocity trend by age bin",
        xaxis_title="Age-bin center (years)",
        yaxis_title="Mean area-weighted RMS speed / year",
        template="plotly_white",
    )
    figures.append(rms_fig)

    nvr_fig = go.Figure()
    for diagnosis in ("CN", "AD", "all"):
        group = age_bin_df.loc[age_bin_df["diagnosis"] == diagnosis]
        if group.empty:
            continue
        nvr_fig.add_trace(
            go.Scatter(
                x=group["age_bin_center"],
                y=group["net_volume_rate_per_year_mean"],
                mode="lines+markers",
                name=diagnosis,
            )
        )
    nvr_fig.update_layout(
        title="Training-set net volume-rate trend by age bin",
        xaxis_title="Age-bin center (years)",
        yaxis_title="Mean net volume rate / year",
        template="plotly_white",
    )
    figures.append(nvr_fig)

    per_scan_fig = go.Figure()
    for diagnosis, group in velocity_df.groupby("diagnosis", sort=True):
        per_scan_fig.add_trace(
            go.Box(
                y=group["area_weighted_rms_speed_per_year"],
                name=str(diagnosis),
                boxmean=True,
            )
        )
    per_scan_fig.update_layout(
        title="Training-set per-scan RMS speed distribution",
        yaxis_title="Area-weighted RMS speed / year",
        template="plotly_white",
    )
    figures.append(per_scan_fig)

    return figures


@lru_cache(maxsize=16)
def evaluate_selected_train_adjacent_pairs(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    velocity_method: str = "finite_difference",
) -> Dict[str, object]:
    bundle = _cached_bundle(checkpoint, device)
    subject_groups = _selected_subject_groups(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )
    rows: List[Dict[str, object]] = []
    for subject_id, subject_rows in subject_groups.items():
        ordered = subject_rows.sort_values(["visit_order", "continuous_age_norm"]).reset_index(drop=True)
        baseline_time = float(ordered["continuous_age_norm"].min())
        subject_anchor = load_train_subject_anchor(
            str(subject_id),
            checkpoint=checkpoint,
            device=device,
        )
        for idx in range(len(ordered) - 1):
            source_row = ordered.iloc[idx]
            target_row = ordered.iloc[idx + 1]
            delta_years = float(target_row["continuous_age_years"]) - float(
                source_row["continuous_age_years"]
            )
            if delta_years <= 0.0:
                continue
            try:
                observed = model_helpers.observed_correspondence_speed(
                    source_row["mesh_path"],
                    target_row["mesh_path"],
                    delta_years=delta_years,
                )
            except Exception:
                continue
            source_latent = model_helpers.transport_direct(
                bundle,
                subject_anchor,
                baseline_time=baseline_time,
                target_time=float(source_row["continuous_age_norm"]),
                target_label_ad=int(source_row["label_ad"]),
            )
            source_mesh = model_helpers.load_mesh(source_row["mesh_path"])
            source_vertices = np.asarray(source_mesh.vertices, dtype=np.float32)
            model_speed = model_helpers.implicit_surface_normal_velocity(
                bundle,
                source_latent,
                float(source_row["continuous_age_norm"]),
                int(source_row["label_ad"]),
                source_vertices,
                yearly=True,
                method=velocity_method,
            )
            rows.append(
                {
                    "subject_id": str(subject_id),
                    "diagnosis": str(source_row["diagnosis"]),
                    "source_scan_id": str(source_row["scan_id"]),
                    "target_scan_id": str(target_row["scan_id"]),
                    "source_age_years": float(source_row["continuous_age_years"]),
                    "target_age_years": float(target_row["continuous_age_years"]),
                    "delta_years": float(delta_years),
                    "observed_area_weighted_rms_speed_per_year": float(
                        observed["area_weighted_rms_speed_per_year"]
                    ),
                    "observed_mean_absolute_speed_per_year": float(
                        observed["mean_absolute_speed_per_year"]
                    ),
                    "observed_net_volume_rate_per_year": float(
                        observed["net_volume_rate_per_year"]
                    ),
                    "model_area_weighted_rms_speed_per_year": float(
                        model_helpers.summarize_speed_map(source_mesh, model_speed)[
                            "area_weighted_rms_speed_per_year"
                        ]
                    ),
                    "model_mean_absolute_speed_per_year": float(
                        model_helpers.summarize_speed_map(source_mesh, model_speed)[
                            "mean_absolute_speed_per_year"
                        ]
                    ),
                    "model_net_volume_rate_per_year": float(
                        model_helpers.summarize_speed_map(source_mesh, model_speed)[
                            "net_volume_rate_per_year"
                        ]
                    ),
                    "local_speed_corr": float(np.corrcoef(observed["speed"], model_speed)[0, 1]),
                    "local_speed_rmse": float(
                        np.sqrt(np.mean((np.asarray(observed["speed"]) - np.asarray(model_speed)) ** 2))
                    ),
                    "local_speed_bias": float(
                        np.mean(np.asarray(model_speed) - np.asarray(observed["speed"]))
                    ),
                    "local_speed_sign_agreement": float(
                        np.mean(
                            np.sign(np.asarray(model_speed))
                            == np.sign(np.asarray(observed["speed"]))
                        )
                    ),
                }
            )
    compare_df = pd.DataFrame(rows).sort_values(
        ["diagnosis", "source_age_years", "subject_id"]
    ).reset_index(drop=True)
    summary_df = model_helpers.compute_group_summary(
        compare_df,
        metric_cols=[
            "local_speed_corr",
            "local_speed_rmse",
            "observed_area_weighted_rms_speed_per_year",
            "model_area_weighted_rms_speed_per_year",
            "observed_net_volume_rate_per_year",
            "model_net_volume_rate_per_year",
        ],
        diagnosis_col="diagnosis",
        subject_col="subject_id",
        bootstrap_iterations=1000,
        seed=21,
    )
    return {
        "compare_df": compare_df,
        "summary_df": summary_df,
    }


def train_adjacent_comparison_figures(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    velocity_method: str = "finite_difference",
) -> List[go.Figure]:
    payload = evaluate_selected_train_adjacent_pairs(
        checkpoint=checkpoint,
        device=device,
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
        velocity_method=velocity_method,
    )
    compare_df = payload["compare_df"].copy()
    figures: List[go.Figure] = []

    rms_fig = go.Figure()
    for diagnosis, group in compare_df.groupby("diagnosis", sort=True):
        rms_fig.add_trace(
            go.Scatter(
                x=group["observed_area_weighted_rms_speed_per_year"],
                y=group["model_area_weighted_rms_speed_per_year"],
                mode="markers",
                name=str(diagnosis),
                text=group["subject_id"],
            )
        )
    if not compare_df.empty:
        lo = float(
            min(
                compare_df["observed_area_weighted_rms_speed_per_year"].min(),
                compare_df["model_area_weighted_rms_speed_per_year"].min(),
            )
        )
        hi = float(
            max(
                compare_df["observed_area_weighted_rms_speed_per_year"].max(),
                compare_df["model_area_weighted_rms_speed_per_year"].max(),
            )
        )
        rms_fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="y=x",
                line=dict(color="black", dash="dash"),
            )
        )
    rms_fig.update_layout(
        title="Training-set observed versus model RMS speed on adjacent pairs",
        xaxis_title="Observed RMS speed / year",
        yaxis_title="Model RMS speed / year",
        template="plotly_white",
    )
    figures.append(rms_fig)

    corr_fig = go.Figure()
    for diagnosis, group in compare_df.groupby("diagnosis", sort=True):
        corr_fig.add_trace(
            go.Box(
                y=group["local_speed_corr"],
                name=str(diagnosis),
                boxmean=True,
            )
        )
    corr_fig.update_layout(
        title="Training-set local speed correlation on adjacent pairs",
        yaxis_title="Observed vs model local-speed correlation",
        template="plotly_white",
    )
    figures.append(corr_fig)

    nvr_fig = go.Figure()
    for diagnosis, group in compare_df.groupby("diagnosis", sort=True):
        nvr_fig.add_trace(
            go.Scatter(
                x=group["observed_net_volume_rate_per_year"],
                y=group["model_net_volume_rate_per_year"],
                mode="markers",
                name=str(diagnosis),
                text=group["subject_id"],
            )
        )
    if not compare_df.empty:
        lo = float(
            min(
                compare_df["observed_net_volume_rate_per_year"].min(),
                compare_df["model_net_volume_rate_per_year"].min(),
            )
        )
        hi = float(
            max(
                compare_df["observed_net_volume_rate_per_year"].max(),
                compare_df["model_net_volume_rate_per_year"].max(),
            )
        )
        nvr_fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="y=x",
                line=dict(color="black", dash="dash"),
            )
        )
    nvr_fig.update_layout(
        title="Training-set observed versus model net volume rate on adjacent pairs",
        xaxis_title="Observed net volume rate / year",
        yaxis_title="Model net volume rate / year",
        template="plotly_white",
    )
    figures.append(nvr_fig)

    return figures


def representative_train_local_map_figures(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    velocity_method: str = "finite_difference",
) -> List[go.Figure]:
    payload = evaluate_selected_train_adjacent_pairs(
        checkpoint=checkpoint,
        device=device,
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
        velocity_method=velocity_method,
    )
    compare_df = payload["compare_df"].copy()
    bundle = _cached_bundle(checkpoint, device)
    figures: List[go.Figure] = []

    for diagnosis in ("CN", "AD"):
        group = compare_df.loc[compare_df["diagnosis"] == diagnosis].copy()
        if group.empty:
            continue
        row = group.sort_values(["local_speed_corr", "source_age_years"], ascending=[False, True]).iloc[0]
        source_meta = _train_records_df().loc[
            _train_records_df()["scan_id"] == str(row["source_scan_id"])
        ].iloc[0]
        target_meta = _train_records_df().loc[
            _train_records_df()["scan_id"] == str(row["target_scan_id"])
        ].iloc[0]
        delta_years = float(target_meta["continuous_age_years"]) - float(
            source_meta["continuous_age_years"]
        )
        observed = model_helpers.observed_correspondence_speed(
            source_meta["mesh_path"],
            target_meta["mesh_path"],
            delta_years=delta_years,
        )
        subject_group = _train_subject_groups()[str(row["subject_id"])]
        baseline_time = float(subject_group["continuous_age_norm"].min())
        anchor = load_train_subject_anchor(
            str(row["subject_id"]),
            checkpoint=checkpoint,
            device=device,
        )
        latent = model_helpers.transport_direct(
            bundle,
            anchor,
            baseline_time=baseline_time,
            target_time=float(source_meta["continuous_age_norm"]),
            target_label_ad=int(source_meta["label_ad"]),
        )
        mesh_obj = model_helpers.load_mesh(source_meta["mesh_path"])
        vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
        faces = np.asarray(mesh_obj.faces, dtype=int)
        model_speed = model_helpers.implicit_surface_normal_velocity(
            bundle,
            latent,
            float(source_meta["continuous_age_norm"]),
            int(source_meta["label_ad"]),
            vertices,
            yearly=True,
            method=velocity_method,
        )
        figures.extend(
            [
                base_helpers.vertex_value_figure(
                    vertices,
                    faces,
                    observed["speed"],
                    title=(
                        f"Observed local change: train {diagnosis} subject {row['subject_id']} "
                        f"({float(source_meta['continuous_age_years']):.1f} -> {float(target_meta['continuous_age_years']):.1f})"
                    ),
                    symmetric=True,
                    colorbar_title="observed normal speed / year",
                ),
                base_helpers.vertex_value_figure(
                    vertices,
                    faces,
                    model_speed,
                    title=(
                        f"Model local change: train {diagnosis} subject {row['subject_id']} "
                        f"(source age {float(source_meta['continuous_age_years']):.1f})"
                    ),
                    symmetric=True,
                    colorbar_title="model normal speed / year",
                ),
                base_helpers.vertex_value_figure(
                    vertices,
                    faces,
                    model_speed - np.asarray(observed["speed"]),
                    title=(
                        f"Model minus observed local change: train {diagnosis} subject {row['subject_id']}"
                    ),
                    symmetric=True,
                    colorbar_title="model - observed / year",
                ),
            ]
        )
    return figures


def _representative_sweep_rows(
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    reference_age_years: float = 70.0,
) -> pd.DataFrame:
    selected_scans = selected_train_scan_frame(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )
    selected_subject_df = selected_train_subjects(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
    )
    chosen_rows: List[pd.Series] = []
    for diagnosis in ("CN", "AD"):
        subject_candidates = selected_subject_df.loc[
            selected_subject_df["last_diagnosis"] == diagnosis
        ]
        if subject_candidates.empty:
            subject_candidates = selected_subject_df.loc[
                selected_subject_df["first_diagnosis"] == diagnosis
            ]
        if subject_candidates.empty:
            continue
        best_row = None
        best_dist = float("inf")
        for subject_id in subject_candidates["subject_id"]:
            scan_group = selected_scans.loc[selected_scans["subject_id"] == subject_id].copy()
            if scan_group.empty:
                continue
            candidate = scan_group.iloc[
                np.argmin(np.abs(scan_group["continuous_age_years"].to_numpy() - float(reference_age_years)))
            ]
            dist = abs(float(candidate["continuous_age_years"]) - float(reference_age_years))
            if dist < best_dist:
                best_dist = dist
                best_row = candidate
        if best_row is not None:
            chosen_rows.append(best_row)
    return pd.DataFrame(chosen_rows).reset_index(drop=True)


@lru_cache(maxsize=16)
def evaluate_train_age_sweeps(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    mesh_resolution: int = DEFAULT_MESH_RESOLUTION,
    end_age_years: float = 110.0,
    step_years: float = DEFAULT_SWEEP_STEP_YEARS,
    reference_age_years: float = 70.0,
) -> Dict[str, object]:
    bundle = _cached_bundle(checkpoint, device)
    rep_rows = _representative_sweep_rows(
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
        reference_age_years=reference_age_years,
    )
    sweep_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for _, start_row in rep_rows.iterrows():
        subject_id = str(start_row["subject_id"])
        diagnosis = str(start_row["diagnosis"])
        start_age = float(start_row["continuous_age_years"])
        start_time = float(start_row["continuous_age_norm"])
        start_label = int(start_row["label_ad"])
        start_scan_id = str(start_row["scan_id"])
        subject_group = _train_subject_groups()[subject_id]
        baseline_time = float(subject_group["continuous_age_norm"].min())
        subject_anchor = load_train_subject_anchor(
            subject_id,
            checkpoint=checkpoint,
            device=device,
        )
        start_latent = model_helpers.transport_direct(
            bundle,
            subject_anchor,
            baseline_time=baseline_time,
            target_time=start_time,
            target_label_ad=start_label,
        )

        for method in ("direct", "composed"):
            last_valid_age = start_age
            first_invalid_age = float("nan")
            current_latent = start_latent
            for age_years in np.arange(start_age, float(end_age_years) + 1e-8, float(step_years)):
                if method == "direct":
                    pred_latent = model_helpers.transport_direct(
                        bundle,
                        start_latent,
                        baseline_time=start_time,
                        target_time=(float(age_years) - model_helpers.TRAINING_AGE_MIN)
                        / model_helpers.TRAINING_AGE_RANGE_YEARS,
                        target_label_ad=start_label,
                    )
                else:
                    pred_latent = (
                        start_latent
                        if abs(float(age_years) - start_age) <= 1e-12
                        else _compose_latent_uniform(
                            bundle,
                            start_latent,
                            start_age_years=start_age,
                            final_age_years=float(age_years),
                            label_ad=start_label,
                            step_years=step_years,
                        )
                    )

                decode_info = (
                    {
                        "mesh_valid": True,
                        "mesh": model_helpers.load_mesh(start_row["mesh_path"]),
                        "mesh_error": "",
                        "pred_volume": model_helpers.mesh_volume(start_row["mesh_path"]),
                        **_probe_decoder_range(bundle, pred_latent),
                    }
                    if abs(float(age_years) - start_age) <= 1e-12
                    else _safe_decode_mesh(
                        bundle,
                        pred_latent,
                        resolution=mesh_resolution,
                        max_batch=DEFAULT_MAX_BATCH,
                    )
                )
                if decode_info["mesh_valid"]:
                    last_valid_age = float(age_years)
                elif math.isnan(first_invalid_age):
                    first_invalid_age = float(age_years)
                sweep_rows.append(
                    {
                        "subject_id": subject_id,
                        "diagnosis": diagnosis,
                        "start_scan_id": start_scan_id,
                        "start_age_years": start_age,
                        "age_years": float(age_years),
                        "years_from_start": float(age_years - start_age),
                        "method": method,
                        "in_distribution": bool(float(age_years) <= TRAINING_AGE_MAX_IN_DISTRIBUTION),
                        "mesh_valid": bool(decode_info["mesh_valid"]),
                        "pred_volume": float(decode_info["pred_volume"])
                        if decode_info["mesh_valid"]
                        else float("nan"),
                        "probe_sdf_min": float(decode_info["probe_sdf_min"]),
                        "probe_sdf_max": float(decode_info["probe_sdf_max"]),
                        "probe_zero_cross": bool(decode_info["probe_zero_cross"]),
                        "latent_l2_norm": float(decode_info["latent_l2_norm"]),
                    }
                )
            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "diagnosis": diagnosis,
                    "start_scan_id": start_scan_id,
                    "start_age_years": start_age,
                    "method": method,
                    "last_valid_age_years": float(last_valid_age),
                    "first_invalid_age_years": float(first_invalid_age)
                    if not math.isnan(first_invalid_age)
                    else float("nan"),
                }
            )

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ["diagnosis", "subject_id", "method", "age_years"]
    ).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["diagnosis", "subject_id", "method"]
    ).reset_index(drop=True)
    return {
        "sweep_df": sweep_df,
        "summary_df": summary_df,
        "representative_start_rows": rep_rows,
    }


def train_age_sweep_figures(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    n_subjects: int = DEFAULT_SELECTION_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    bin_width_years: float = DEFAULT_AGE_BIN_WIDTH,
    mesh_resolution: int = DEFAULT_MESH_RESOLUTION,
    end_age_years: float = 110.0,
    step_years: float = DEFAULT_SWEEP_STEP_YEARS,
    reference_age_years: float = 70.0,
) -> List[go.Figure]:
    payload = evaluate_train_age_sweeps(
        checkpoint=checkpoint,
        device=device,
        n_subjects=n_subjects,
        seed=seed,
        bin_width_years=bin_width_years,
        mesh_resolution=mesh_resolution,
        end_age_years=end_age_years,
        step_years=step_years,
        reference_age_years=reference_age_years,
    )
    sweep_df = payload["sweep_df"].copy()
    figures: List[go.Figure] = []

    volume_fig = go.Figure()
    for (diagnosis, method), group in sweep_df.groupby(["diagnosis", "method"], sort=True):
        valid = group.loc[group["mesh_valid"]].copy()
        if valid.empty:
            continue
        volume_fig.add_trace(
            go.Scatter(
                x=valid["age_years"],
                y=valid["pred_volume"],
                mode="lines+markers",
                name=f"{diagnosis} | {method}",
                line=dict(dash="solid" if method == "direct" else "dot"),
            )
        )
    volume_fig.add_vline(
        x=TRAINING_AGE_MAX_IN_DISTRIBUTION,
        line=dict(color="black", dash="dash"),
        annotation_text="train max age",
    )
    volume_fig.update_layout(
        title="Train-latent in-distribution and OOD age sweep volume trend",
        xaxis_title="Target age (years)",
        yaxis_title="Predicted volume (valid meshes only)",
        template="plotly_white",
    )
    figures.append(volume_fig)

    validity_fig = go.Figure()
    for (diagnosis, method), group in sweep_df.groupby(["diagnosis", "method"], sort=True):
        validity_fig.add_trace(
            go.Scatter(
                x=group["age_years"],
                y=group["mesh_valid"].astype(int),
                mode="lines+markers",
                name=f"{diagnosis} | {method}",
                line=dict(dash="solid" if method == "direct" else "dot"),
            )
        )
    validity_fig.add_vline(
        x=TRAINING_AGE_MAX_IN_DISTRIBUTION,
        line=dict(color="black", dash="dash"),
        annotation_text="train max age",
    )
    validity_fig.update_layout(
        title="Train-latent mesh validity across in-distribution and OOD ages",
        xaxis_title="Target age (years)",
        yaxis_title="Mesh valid (1=yes, 0=no)",
        template="plotly_white",
        yaxis=dict(range=[-0.05, 1.05]),
    )
    figures.append(validity_fig)

    norm_fig = go.Figure()
    for (diagnosis, method), group in sweep_df.groupby(["diagnosis", "method"], sort=True):
        norm_fig.add_trace(
            go.Scatter(
                x=group["age_years"],
                y=group["latent_l2_norm"],
                mode="lines+markers",
                name=f"{diagnosis} | {method}",
                line=dict(dash="solid" if method == "direct" else "dot"),
            )
        )
    norm_fig.add_vline(
        x=TRAINING_AGE_MAX_IN_DISTRIBUTION,
        line=dict(color="black", dash="dash"),
        annotation_text="train max age",
    )
    norm_fig.update_layout(
        title="Train-latent norm drift across in-distribution and OOD ages",
        xaxis_title="Target age (years)",
        yaxis_title="Latent L2 norm",
        template="plotly_white",
    )
    figures.append(norm_fig)

    return figures
