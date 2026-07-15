from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_deep_sdf_longitudinal as longitudinal
from adni_no_mci_longitudinal_model_helpers import (
    DEFAULT_ANCHOR_FIT_LR,
    DEFAULT_ANCHOR_FIT_SAMPLES,
    DEFAULT_ANCHOR_FIT_STEPS,
    DEFAULT_ANCHOR_INIT_STD,
    DEFAULT_VELOCITY_EPS,
    anchor_baseline_time,
    compute_group_summary,
    ensure_dir,
    ensure_file,
    finite_step_surface_change,
    fit_subject_anchor,
    forecast_time_delta_to_years,
    grouped_subject_rows,
    implicit_surface_normal_velocity,
    latent_velocity_vector,
    load_mesh,
    load_metadata,
    load_model_bundle,
    observed_correspondence_speed,
    summarize_speed_map,
    seed_everything,
    split_frame,
    transport_composed,
    transport_direct,
)


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_AGE_BIN_CENTERS = (60.0, 70.0, 80.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cocycle and velocity diagnostics for ADNI no-MCI longitudinal modeling."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory with metadata, specs, and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        default="selected",
        help="Checkpoint to evaluate. Use 'selected' to read validation selection output.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for loading the decoder and flow.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=("train", "val", "test"),
        help="Metadata splits to analyze.",
    )
    parser.add_argument(
        "--anchor-fit-steps",
        type=int,
        default=DEFAULT_ANCHOR_FIT_STEPS,
        help="Anchor fitting optimization iterations.",
    )
    parser.add_argument(
        "--anchor-fit-samples",
        type=int,
        default=DEFAULT_ANCHOR_FIT_SAMPLES,
        help="SDF samples per anchor optimization step.",
    )
    parser.add_argument(
        "--anchor-fit-lr",
        type=float,
        default=DEFAULT_ANCHOR_FIT_LR,
        help="Anchor fitting learning rate.",
    )
    parser.add_argument(
        "--anchor-init-std",
        type=float,
        default=DEFAULT_ANCHOR_INIT_STD,
        help="Anchor initialization stddev.",
    )
    parser.add_argument(
        "--velocity-eps",
        type=float,
        default=DEFAULT_VELOCITY_EPS,
        help="Finite-difference epsilon in normalized age units for dz/dt.",
    )
    parser.add_argument(
        "--age-bin-half-width",
        type=float,
        default=2.5,
        help="Half-width in years for the fixed age windows centered at 60, 70, and 80.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for deterministic evaluation.",
    )
    return parser.parse_args()


def load_validation_report(experiment_dir: Path) -> Dict[str, object]:
    report_path = ensure_file(experiment_dir / "metadata" / "input_validation_report.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if str(payload.get("status", "")).lower() != "pass":
        raise RuntimeError(
            f"Input validation status is not pass at {report_path}. Refusing to evaluate."
        )
    return payload


def analysis_dir(experiment_dir: Path) -> Path:
    return ensure_dir(experiment_dir / "analysis" / "velocity")


def maps_dir(experiment_dir: Path) -> Path:
    return ensure_dir(analysis_dir(experiment_dir) / "maps")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_selected_checkpoint(experiment_dir: Path) -> str:
    selected_path = ensure_file(experiment_dir / "analysis" / "validation" / "selected_checkpoint.json")
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    checkpoint = payload.get("selected_checkpoint", None)
    if checkpoint is None:
        raise RuntimeError(f"selected_checkpoint.json has no selected_checkpoint: {selected_path}")
    return str(checkpoint)


def resolve_checkpoint(experiment_dir: Path, checkpoint: str) -> str:
    return read_selected_checkpoint(experiment_dir) if str(checkpoint) == "selected" else str(checkpoint)


def evaluation_frame(bundle, splits: Sequence[str]) -> pd.DataFrame:
    frames = [split_frame(bundle.metadata, split_name) for split_name in splits]
    if not frames:
        return pd.DataFrame(columns=bundle.metadata.columns)
    return (
        pd.concat(frames, axis=0, ignore_index=True)
        .sort_values(["split", "subject_id", "visit_order", "scan_id"])
        .reset_index(drop=True)
    )


def observation_seed(base_seed: int, subject_index: int) -> int:
    return int(base_seed + subject_index * 1000)


def pair_seed(base_seed: int, subject_index: int, pair_index: int) -> int:
    return int(base_seed + subject_index * 10000 + pair_index * 100)


def age_window_center(age_years: float, half_width: float) -> Optional[float]:
    for center in DEFAULT_AGE_BIN_CENTERS:
        if abs(float(age_years) - float(center)) <= float(half_width):
            return float(center)
    return None


def apply_flow_interval(bundle, latent, source_time: float, target_time: float, label_ad: int) -> torch.Tensor:
    source_tensor = torch.tensor([[float(source_time)]], device=bundle.device, dtype=latent.dtype)
    target_tensor = torch.tensor([[float(target_time)]], device=bundle.device, dtype=latent.dtype)
    cond = torch.tensor([[float(label_ad)]], device=bundle.device, dtype=latent.dtype)
    with torch.no_grad():
        return longitudinal.apply_temporal_flow(
            bundle.flow, latent, source_tensor, target_tensor, age_cond=cond
        )


def direct_vs_composed_diagnostics(
    bundle,
    subject_id: str,
    diagnosis: str,
    anchor,
    baseline_time: float,
    subject_rows: pd.DataFrame,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ordered = subject_rows.sort_values("visit_order").reset_index(drop=True)
    if len(ordered) < 3:
        return rows
    first_time = float(ordered.iloc[0]["continuous_age_norm"])
    for target_idx in range(2, len(ordered)):
        target_row = ordered.iloc[target_idx]
        intermediate_times = [
            float(v)
            for v in ordered.iloc[1:target_idx]["continuous_age_norm"].tolist()
        ]
        label_ad = int(target_row["label_ad"])
        direct = transport_direct(
            bundle,
            anchor,
            baseline_time=baseline_time,
            target_time=float(target_row["continuous_age_norm"]),
            target_label_ad=label_ad,
        )
        composed = transport_composed(
            bundle,
            anchor,
            baseline_time=baseline_time,
            intermediate_times=intermediate_times,
            target_time=float(target_row["continuous_age_norm"]),
            target_label_ad=label_ad,
        )
        diff = direct - composed
        rows.append(
            {
                "diagnostic_type": "direct_vs_composed",
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "source_scan_id": str(ordered.iloc[0]["scan_id"]),
                "intermediate_scan_id": "|".join(
                    str(v) for v in ordered.iloc[1:target_idx]["scan_id"].tolist()
                ),
                "target_scan_id": str(target_row["scan_id"]),
                "source_visit_order": int(ordered.iloc[0]["visit_order"]),
                "target_visit_order": int(target_row["visit_order"]),
                "source_time_norm": first_time,
                "target_time_norm": float(target_row["continuous_age_norm"]),
                "delta_years": forecast_time_delta_to_years(
                    float(target_row["continuous_age_norm"]) - first_time
                ),
                "latent_l2_error": float(diff.norm().item()),
                "latent_mean_abs_error": float(diff.abs().mean().item()),
            }
        )
    return rows


def forward_backward_closure_row(
    bundle,
    subject_id: str,
    diagnosis: str,
    source_row: pd.Series,
    target_row: pd.Series,
    latent_source: torch.Tensor,
    label_ad: int,
) -> Dict[str, object]:
    z_forward = apply_flow_interval(
        bundle,
        latent_source,
        source_time=float(source_row["continuous_age_norm"]),
        target_time=float(target_row["continuous_age_norm"]),
        label_ad=label_ad,
    )
    z_back = apply_flow_interval(
        bundle,
        z_forward,
        source_time=float(target_row["continuous_age_norm"]),
        target_time=float(source_row["continuous_age_norm"]),
        label_ad=label_ad,
    )
    diff = z_back - latent_source
    return {
        "diagnostic_type": "forward_backward_closure",
        "subject_id": subject_id,
        "diagnosis": diagnosis,
        "source_scan_id": str(source_row["scan_id"]),
        "intermediate_scan_id": "",
        "target_scan_id": str(target_row["scan_id"]),
        "source_visit_order": int(source_row["visit_order"]),
        "target_visit_order": int(target_row["visit_order"]),
        "source_time_norm": float(source_row["continuous_age_norm"]),
        "target_time_norm": float(target_row["continuous_age_norm"]),
        "delta_years": float(target_row["elapsed_years"]) - float(source_row["elapsed_years"]),
        "latent_l2_error": float(diff.norm().item()),
        "latent_mean_abs_error": float(diff.abs().mean().item()),
    }


def pair_comparison_metrics(observed_speed: np.ndarray, model_speed: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(observed_speed, dtype=np.float64).reshape(-1)
    model = np.asarray(model_speed, dtype=np.float64).reshape(-1)
    if obs.shape != model.shape:
        raise ValueError(f"Observed and model local-speed arrays must match, got {obs.shape} vs {model.shape}")
    diff = model - obs
    corr = float("nan")
    if obs.size >= 2 and np.std(obs) > 1e-12 and np.std(model) > 1e-12:
        corr = float(np.corrcoef(obs, model)[0, 1])
    sign_agreement = float(np.mean(np.sign(obs) == np.sign(model))) if obs.size else float("nan")
    return {
        "local_speed_corr": corr,
        "local_speed_mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
        "local_speed_rmse": float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan"),
        "local_speed_bias": float(np.mean(diff)) if diff.size else float("nan"),
        "local_speed_sign_agreement": sign_agreement,
    }


def export_scan_map(
    out_dir: Path,
    subject_id: str,
    scan_id: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    speed_factual: np.ndarray,
    speed_cn: np.ndarray,
    speed_ad: np.ndarray,
) -> str:
    out_path = out_dir / f"{subject_id}__{scan_id}__speed_maps.npz"
    np.savez_compressed(
        out_path,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        yearly_speed_factual=np.asarray(speed_factual, dtype=np.float32),
        yearly_speed_cn=np.asarray(speed_cn, dtype=np.float32),
        yearly_speed_ad=np.asarray(speed_ad, dtype=np.float32),
        yearly_speed_ad_minus_cn=np.asarray(speed_ad - speed_cn, dtype=np.float32),
    )
    return str(out_path.resolve())


def export_pair_map(
    out_dir: Path,
    subject_id: str,
    source_scan_id: str,
    target_scan_id: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    observed_speed: np.ndarray,
    model_speed: np.ndarray,
) -> str:
    out_path = out_dir / f"{subject_id}__{source_scan_id}__to__{target_scan_id}__pair_speed_maps.npz"
    np.savez_compressed(
        out_path,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        observed_local_speed=np.asarray(observed_speed, dtype=np.float32),
        model_local_speed=np.asarray(model_speed, dtype=np.float32),
        local_speed_difference=np.asarray(model_speed - observed_speed, dtype=np.float32),
    )
    return str(out_path.resolve())


def build_age_bin_summary(
    scan_df: pd.DataFrame,
    half_width: float,
) -> pd.DataFrame:
    frame = scan_df.copy()
    frame["age_bin_center"] = frame["continuous_age_years"].apply(
        lambda value: age_window_center(value, half_width=half_width)
    )
    frame = frame.dropna(subset=["age_bin_center"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    metric_cols = [
        "area_weighted_rms_speed_per_year",
        "mean_absolute_speed_per_year",
        "net_volume_rate_per_year",
        "counterfactual_ad_minus_cn_rms_speed_per_year",
        "latent_velocity_l2_per_year",
    ]
    for (age_center, diagnosis), group in frame.groupby(["age_bin_center", "diagnosis"], sort=True):
        row = {
            "age_bin_center": float(age_center),
            "diagnosis": str(diagnosis),
            "age_bin_half_width_years": float(half_width),
            "num_rows": int(len(group)),
            "num_subjects": int(group["subject_id"].nunique()),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
        rows.append(row)
    all_diag = (
        frame.groupby("age_bin_center", sort=True)
        .agg(
            num_rows=("subject_id", "count"),
            num_subjects=("subject_id", "nunique"),
            area_weighted_rms_speed_per_year_mean=("area_weighted_rms_speed_per_year", "mean"),
            mean_absolute_speed_per_year_mean=("mean_absolute_speed_per_year", "mean"),
            net_volume_rate_per_year_mean=("net_volume_rate_per_year", "mean"),
            counterfactual_ad_minus_cn_rms_speed_per_year_mean=(
                "counterfactual_ad_minus_cn_rms_speed_per_year",
                "mean",
            ),
            latent_velocity_l2_per_year_mean=("latent_velocity_l2_per_year", "mean"),
        )
        .reset_index()
    )
    if not all_diag.empty:
        all_diag.insert(1, "diagnosis", "all")
        all_diag.insert(2, "age_bin_half_width_years", float(half_width))
        rows.extend(all_diag.to_dict(orient="records"))
    return pd.DataFrame(rows).sort_values(["age_bin_center", "diagnosis"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.experiment = args.experiment.resolve()
    seed_everything(args.seed)
    load_validation_report(args.experiment)

    checkpoint = resolve_checkpoint(args.experiment, args.checkpoint)
    bundle = load_model_bundle(args.experiment, checkpoint, device=args.device)
    frame = evaluation_frame(bundle, args.splits)
    subjects = grouped_subject_rows(frame)

    out_dir = analysis_dir(args.experiment)
    map_dir = maps_dir(args.experiment)

    subject_summary_rows: List[Dict[str, object]] = []
    scan_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    for subject_index, (subject_id, subject_rows) in enumerate(subjects.items()):
        ordered = subject_rows.sort_values("visit_order").reset_index(drop=True)
        diagnosis = str(ordered.iloc[0]["diagnosis"])
        anchor, loss_hist, observations = fit_subject_anchor(
            bundle,
            ordered["scan_id"].tolist(),
            seed=observation_seed(args.seed, subject_index),
            num_iterations=args.anchor_fit_steps,
            num_samples=args.anchor_fit_samples,
            lr=args.anchor_fit_lr,
            init_std=args.anchor_init_std,
        )
        baseline_time = float(anchor_baseline_time(observations))
        latent_by_scan: Dict[str, torch.Tensor] = {}
        scan_rows_before = len(scan_rows)
        pair_rows_before = len(pair_rows)
        diag_rows_before = len(diagnostic_rows)

        diagnostic_rows.extend(
            direct_vs_composed_diagnostics(
                bundle=bundle,
                subject_id=subject_id,
                diagnosis=diagnosis,
                anchor=anchor,
                baseline_time=baseline_time,
                subject_rows=ordered,
            )
        )

        for _, row in ordered.iterrows():
            mesh = load_mesh(row["mesh_path"])
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            current_time = float(row["continuous_age_norm"])
            current_age_years = float(row["continuous_age_years"])
            label_ad = int(row["label_ad"])

            latent = transport_direct(
                bundle,
                anchor,
                baseline_time=baseline_time,
                target_time=current_time,
                target_label_ad=label_ad,
            )
            latent_by_scan[str(row["scan_id"])] = latent

            factual_speed = implicit_surface_normal_velocity(
                bundle,
                latent,
                current_time=current_time,
                label_ad=label_ad,
                query_points=vertices,
                eps=args.velocity_eps,
                yearly=True,
            )
            cn_speed = implicit_surface_normal_velocity(
                bundle,
                latent,
                current_time=current_time,
                label_ad=0,
                query_points=vertices,
                eps=args.velocity_eps,
                yearly=True,
            )
            ad_speed = implicit_surface_normal_velocity(
                bundle,
                latent,
                current_time=current_time,
                label_ad=1,
                query_points=vertices,
                eps=args.velocity_eps,
                yearly=True,
            )
            contrast_speed = ad_speed - cn_speed
            factual_summary = summarize_speed_map(mesh, factual_speed)
            cn_summary = summarize_speed_map(mesh, cn_speed)
            ad_summary = summarize_speed_map(mesh, ad_speed)
            contrast_summary = summarize_speed_map(mesh, contrast_speed)
            dz_dt = latent_velocity_vector(
                bundle,
                latent,
                current_time=current_time,
                label_ad=label_ad,
                eps=args.velocity_eps,
            )
            latent_velocity_norm = float(dz_dt.norm().item())
            latent_velocity_year = float(latent_velocity_norm / 34.0)
            map_path = export_scan_map(
                map_dir,
                subject_id=subject_id,
                scan_id=str(row["scan_id"]),
                vertices=vertices,
                faces=faces,
                speed_factual=factual_speed,
                speed_cn=cn_speed,
                speed_ad=ad_speed,
            )

            scan_record = {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "split": str(row["split"]),
                "scan_id": str(row["scan_id"]),
                "visit_order": int(row["visit_order"]),
                "months_from_baseline": float(row["months_from_baseline"]),
                "elapsed_years": float(row["elapsed_years"]),
                "baseline_age_years": float(row["baseline_age_years"]),
                "continuous_age_years": current_age_years,
                "continuous_age_norm": current_time,
                "label_ad": label_ad,
                "map_path": map_path,
                "latent_velocity_l2_per_norm_time": latent_velocity_norm,
                "latent_velocity_l2_per_year": latent_velocity_year,
            }
            scan_record.update(factual_summary)
            scan_record.update(
                {
                    "cn_area_weighted_rms_speed_per_year": cn_summary["area_weighted_rms_speed_per_year"],
                    "cn_mean_absolute_speed_per_year": cn_summary["mean_absolute_speed_per_year"],
                    "cn_net_volume_rate_per_year": cn_summary["net_volume_rate_per_year"],
                    "ad_area_weighted_rms_speed_per_year": ad_summary["area_weighted_rms_speed_per_year"],
                    "ad_mean_absolute_speed_per_year": ad_summary["mean_absolute_speed_per_year"],
                    "ad_net_volume_rate_per_year": ad_summary["net_volume_rate_per_year"],
                    "counterfactual_ad_minus_cn_rms_speed_per_year": contrast_summary["area_weighted_rms_speed_per_year"],
                    "counterfactual_ad_minus_cn_mean_absolute_speed_per_year": contrast_summary["mean_absolute_speed_per_year"],
                    "counterfactual_ad_minus_cn_net_volume_rate_per_year": contrast_summary["net_volume_rate_per_year"],
                }
            )
            scan_rows.append(scan_record)

        for pair_index in range(len(ordered) - 1):
            source_row = ordered.iloc[pair_index]
            target_row = ordered.iloc[pair_index + 1]
            delta_time_norm = float(target_row["continuous_age_norm"]) - float(source_row["continuous_age_norm"])
            delta_years = forecast_time_delta_to_years(delta_time_norm)
            source_mesh = load_mesh(source_row["mesh_path"])
            vertices = np.asarray(source_mesh.vertices, dtype=np.float32)
            faces = np.asarray(source_mesh.faces, dtype=np.int64)
            label_ad = int(source_row["label_ad"])
            latent_source = latent_by_scan[str(source_row["scan_id"])]
            latent_target = latent_by_scan[str(target_row["scan_id"])]

            observed = observed_correspondence_speed(
                source_row["mesh_path"],
                target_row["mesh_path"],
                delta_years=delta_years,
            )
            model_speed = finite_step_surface_change(
                bundle,
                latent_current=latent_source,
                latent_future=latent_target,
                query_points=vertices,
            ) / max(delta_years, 1e-8)
            model_summary = summarize_speed_map(source_mesh, model_speed)
            compare = pair_comparison_metrics(observed["speed"], model_speed)
            pair_map_path = export_pair_map(
                map_dir,
                subject_id=subject_id,
                source_scan_id=str(source_row["scan_id"]),
                target_scan_id=str(target_row["scan_id"]),
                vertices=vertices,
                faces=faces,
                observed_speed=observed["speed"],
                model_speed=model_speed,
            )
            pair_row = {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "split": str(source_row["split"]),
                "source_scan_id": str(source_row["scan_id"]),
                "target_scan_id": str(target_row["scan_id"]),
                "source_visit_order": int(source_row["visit_order"]),
                "target_visit_order": int(target_row["visit_order"]),
                "source_age_years": float(source_row["continuous_age_years"]),
                "target_age_years": float(target_row["continuous_age_years"]),
                "midpoint_age_years": 0.5
                * (
                    float(source_row["continuous_age_years"])
                    + float(target_row["continuous_age_years"])
                ),
                "delta_years": delta_years,
                "pair_map_path": pair_map_path,
                "observed_area_weighted_rms_speed_per_year": observed["area_weighted_rms_speed_per_year"],
                "observed_mean_absolute_speed_per_year": observed["mean_absolute_speed_per_year"],
                "observed_net_volume_rate_per_year": observed["net_volume_rate_per_year"],
                "model_area_weighted_rms_speed_per_year": model_summary["area_weighted_rms_speed_per_year"],
                "model_mean_absolute_speed_per_year": model_summary["mean_absolute_speed_per_year"],
                "model_net_volume_rate_per_year": model_summary["net_volume_rate_per_year"],
            }
            pair_row.update(compare)
            pair_rows.append(pair_row)
            diagnostic_rows.append(
                forward_backward_closure_row(
                    bundle=bundle,
                    subject_id=subject_id,
                    diagnosis=diagnosis,
                    source_row=source_row,
                    target_row=target_row,
                    latent_source=latent_source,
                    label_ad=label_ad,
                )
            )

        subject_scan_df = pd.DataFrame(scan_rows[scan_rows_before:])
        subject_pair_df = pd.DataFrame(pair_rows[pair_rows_before:])
        subject_diag_df = pd.DataFrame(diagnostic_rows[diag_rows_before:])

        subject_summary_rows.append(
            {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "split": str(ordered.iloc[0]["split"]),
                "num_scans": int(len(ordered)),
                "num_pairs": int(max(0, len(ordered) - 1)),
                "baseline_age_years": float(ordered.iloc[0]["baseline_age_years"]),
                "mean_continuous_age_years": float(ordered["continuous_age_years"].mean()),
                "anchor_fit_loss_start": float(loss_hist[0]) if loss_hist else float("nan"),
                "anchor_fit_loss_end": float(loss_hist[-1]) if loss_hist else float("nan"),
                "mean_latent_velocity_l2_per_year": float(subject_scan_df["latent_velocity_l2_per_year"].mean()),
                "mean_area_weighted_rms_speed_per_year": float(subject_scan_df["area_weighted_rms_speed_per_year"].mean()),
                "mean_mean_absolute_speed_per_year": float(subject_scan_df["mean_absolute_speed_per_year"].mean()),
                "mean_net_volume_rate_per_year": float(subject_scan_df["net_volume_rate_per_year"].mean()),
                "mean_contracting_area_fraction": float(subject_scan_df["contracting_area_fraction"].mean()),
                "mean_expanding_area_fraction": float(subject_scan_df["expanding_area_fraction"].mean()),
                "mean_counterfactual_ad_minus_cn_rms_speed_per_year": float(
                    subject_scan_df["counterfactual_ad_minus_cn_rms_speed_per_year"].mean()
                ),
                "mean_counterfactual_ad_minus_cn_net_volume_rate_per_year": float(
                    subject_scan_df["counterfactual_ad_minus_cn_net_volume_rate_per_year"].mean()
                ),
                "mean_model_observed_local_speed_corr": float(
                    subject_pair_df["local_speed_corr"].mean()
                )
                if not subject_pair_df.empty
                else float("nan"),
                "mean_model_observed_local_speed_rmse": float(
                    subject_pair_df["local_speed_rmse"].mean()
                )
                if not subject_pair_df.empty
                else float("nan"),
                "mean_forward_backward_closure_l2_error": float(
                    subject_diag_df.loc[
                        subject_diag_df["diagnostic_type"] == "forward_backward_closure",
                        "latent_l2_error",
                    ].mean()
                )
                if not subject_diag_df.empty
                else float("nan"),
                "mean_direct_vs_composed_l2_error": float(
                    subject_diag_df.loc[
                        subject_diag_df["diagnostic_type"] == "direct_vs_composed",
                        "latent_l2_error",
                    ].mean()
                )
                if not subject_diag_df.empty
                else float("nan"),
            }
        )

    subject_df = pd.DataFrame(subject_summary_rows).sort_values(["diagnosis", "subject_id"]).reset_index(drop=True)
    scan_df = pd.DataFrame(scan_rows).sort_values(["subject_id", "visit_order"]).reset_index(drop=True)
    pair_df = pd.DataFrame(pair_rows).sort_values(["subject_id", "source_visit_order"]).reset_index(drop=True)
    diag_df = pd.DataFrame(diagnostic_rows).sort_values(
        ["subject_id", "diagnostic_type", "source_visit_order", "target_visit_order"]
    ).reset_index(drop=True)
    age_bins_df = build_age_bin_summary(scan_df, half_width=args.age_bin_half_width)

    subject_path = out_dir / "velocity_per_subject.csv"
    age_bins_path = out_dir / "velocity_age_bins.csv"
    pair_path = out_dir / "velocity_model_vs_observed.csv"
    diag_path = out_dir / "cocycle_diagnostics.csv"
    subject_df.to_csv(subject_path, index=False)
    age_bins_df.to_csv(age_bins_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    subject_summary = compute_group_summary(
        subject_df,
        metric_cols=[
            "mean_latent_velocity_l2_per_year",
            "mean_area_weighted_rms_speed_per_year",
            "mean_mean_absolute_speed_per_year",
            "mean_net_volume_rate_per_year",
            "mean_counterfactual_ad_minus_cn_rms_speed_per_year",
            "mean_model_observed_local_speed_rmse",
        ],
        seed=101,
    )
    pair_summary = compute_group_summary(
        pair_df,
        metric_cols=[
            "local_speed_corr",
            "local_speed_rmse",
            "model_area_weighted_rms_speed_per_year",
            "observed_area_weighted_rms_speed_per_year",
        ],
        seed=303,
    )

    summary_payload = {
        "experiment_dir": str(args.experiment),
        "checkpoint": bundle.checkpoint,
        "checkpoint_epoch": int(bundle.checkpoint_epoch),
        "splits": list(args.splits),
        "num_subjects": int(subject_df["subject_id"].nunique()) if not subject_df.empty else 0,
        "num_scans": int(len(scan_df)),
        "num_pairs": int(len(pair_df)),
        "num_diagnostics": int(len(diag_df)),
        "age_bin_centers_years": list(DEFAULT_AGE_BIN_CENTERS),
        "age_bin_half_width_years": float(args.age_bin_half_width),
        "files": {
            "velocity_per_subject": str(subject_path.resolve()),
            "velocity_age_bins": str(age_bins_path.resolve()),
            "velocity_model_vs_observed": str(pair_path.resolve()),
            "cocycle_diagnostics": str(diag_path.resolve()),
            "maps_dir": str(map_dir.resolve()),
        },
        "subject_summary": subject_summary.to_dict(orient="records"),
        "pair_summary": pair_summary.to_dict(orient="records"),
    }
    save_json(out_dir / "velocity_summary.json", summary_payload)


if __name__ == "__main__":
    main()
