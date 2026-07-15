from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import trimesh

from adni_no_mci_longitudinal_model_helpers import (
    DEFAULT_ANCHOR_FIT_LR,
    DEFAULT_ANCHOR_FIT_SAMPLES,
    DEFAULT_ANCHOR_FIT_STEPS,
    DEFAULT_ANCHOR_INIT_STD,
    DEFAULT_SURFACE_SAMPLES,
    anchor_baseline_time,
    canonical_checkpoint_name,
    compute_group_summary,
    decode_mesh,
    deterministic_mesh_metrics,
    ensure_dir,
    ensure_file,
    fit_subject_anchor,
    grouped_subject_rows,
    list_model_checkpoints,
    load_mesh,
    load_metadata,
    load_model_bundle,
    load_task2_reference_mesh,
    seed_everything,
    select_subjects_by_diagnosis,
    split_frame,
    task2_reference_root,
    transport_composed,
    transport_direct,
)


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent
FORECAST_METHODS = ("direct", "composed", "no_change", "task2_upper_bound")


def log_progress(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ADNI no-MCI longitudinal one-shot forecasting."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory with specs.json, metadata, and checkpoints.",
    )
    parser.add_argument(
        "--mode",
        choices=("validation", "test", "both"),
        default="both",
        help="Run validation checkpoint selection, locked test evaluation, or both.",
    )
    parser.add_argument(
        "--checkpoint",
        default="selected",
        help="Checkpoint for locked test mode. Accepts values like 'selected', '1000', or '1000.pth'.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Forecast mesh extraction requires CUDA.",
    )
    parser.add_argument(
        "--mesh-resolution",
        type=int,
        default=256,
        help="Marching-cubes grid resolution for predicted meshes.",
    )
    parser.add_argument(
        "--mesh-max-batch",
        type=int,
        default=2 ** 18,
        help="Max decoder query batch for mesh extraction.",
    )
    parser.add_argument(
        "--metric-samples",
        type=int,
        default=DEFAULT_SURFACE_SAMPLES,
        help="Deterministic surface samples per mesh for Chamfer/ASSD/HD95.",
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
        help="SDF samples per optimization step when fitting subject anchors.",
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
        "--seed",
        type=int,
        default=0,
        help="Global random seed for deterministic evaluation.",
    )
    parser.add_argument(
        "--selected-mesh-subjects-per-group",
        type=int,
        default=2,
        help="How many CN and AD test subjects to export predicted meshes for.",
    )
    return parser.parse_args()


def load_validation_report(experiment_dir: Path) -> Dict[str, object]:
    report_path = ensure_file(experiment_dir / "metadata" / "input_validation_report.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status = str(payload.get("status", "")).lower()
    if status != "pass":
        raise RuntimeError(
            f"Input validation status is not pass at {report_path}. Refusing to evaluate."
        )
    return payload


def analysis_dir(experiment_dir: Path) -> Path:
    return ensure_dir(experiment_dir / "analysis")


def validation_dir(experiment_dir: Path) -> Path:
    return ensure_dir(analysis_dir(experiment_dir) / "validation")


def forecast_mesh_dir(experiment_dir: Path) -> Path:
    return ensure_dir(analysis_dir(experiment_dir) / "test_forecast_meshes")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_selected_checkpoint(experiment_dir: Path) -> str:
    selected_path = ensure_file(validation_dir(experiment_dir) / "selected_checkpoint.json")
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    checkpoint = payload.get("selected_checkpoint", None)
    if checkpoint is None:
        raise RuntimeError(f"selected_checkpoint.json has no selected_checkpoint: {selected_path}")
    return str(checkpoint)


def ensure_cuda_bundle_device(device_name: str) -> None:
    if str(device_name).split(":")[0] != "cuda":
        raise RuntimeError(
            "Forecast evaluation requires CUDA because DeepSDF mesh extraction uses CUDA tensors."
        )


def build_subject_tasks(subject_rows: pd.DataFrame) -> List[Dict[str, object]]:
    ordered = subject_rows.sort_values("visit_order").reset_index(drop=True)
    visit_orders = set(int(v) for v in ordered["visit_order"].tolist())
    tasks: List[Dict[str, object]] = []
    if {0, 1}.issubset(visit_orders):
        tasks.append(
            {
                "observation_mode": "baseline_only",
                "observation_visits": [0],
                "target_visit_order": 1,
                "target_label": "m06",
                "allows_composed": False,
            }
        )
    if {0, 2}.issubset(visit_orders):
        tasks.append(
            {
                "observation_mode": "baseline_only",
                "observation_visits": [0],
                "target_visit_order": 2,
                "target_label": "m12",
                "allows_composed": True,
            }
        )
    if {0, 1, 2}.issubset(visit_orders):
        tasks.append(
            {
                "observation_mode": "baseline_plus_m06",
                "observation_visits": [0, 1],
                "target_visit_order": 2,
                "target_label": "m12",
                "allows_composed": True,
            }
        )
    return tasks


def observation_seed(base_seed: int, subject_index: int, observation_visits: Sequence[int]) -> int:
    return int(base_seed + subject_index * 1000 + sum((idx + 1) * int(v) for idx, v in enumerate(observation_visits)))


def select_composed_intermediate_times(
    subject_rows: pd.DataFrame,
    observation_visits: Sequence[int],
    target_visit_order: int,
) -> List[float]:
    observed = set(int(v) for v in observation_visits)
    between = subject_rows.loc[
        (subject_rows["visit_order"] > min(observed))
        & (subject_rows["visit_order"] < int(target_visit_order))
    ].sort_values("visit_order")
    return [float(v) for v in between["continuous_age_norm"].tolist()]


def task_metrics_seed(subject_index: int, task_index: int, method_index: int, base_seed: int) -> int:
    return int(base_seed + subject_index * 10000 + task_index * 100 + method_index)


def export_mesh(mesh_obj: trimesh.Trimesh, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_obj.export(out_path)
    return str(out_path.resolve())


def export_selected_subject_manifest(
    experiment_dir: Path,
    subject_id: str,
    payload: Dict[str, object],
) -> None:
    subject_dir = ensure_dir(forecast_mesh_dir(experiment_dir) / subject_id)
    save_json(subject_dir / "manifest.json", payload)


def make_prediction_mesh(
    bundle,
    method: str,
    anchor,
    baseline_time: float,
    target_row: pd.Series,
    latest_observed_row: pd.Series,
    subject_rows: pd.DataFrame,
    observation_visits: Sequence[int],
    mesh_resolution: int,
    mesh_max_batch: int,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    label_ad = int(target_row["label_ad"])
    target_time = float(target_row["continuous_age_norm"])
    prediction_info: Dict[str, object] = {
        "method": method,
        "target_scan_id": str(target_row["scan_id"]),
        "target_visit_order": int(target_row["visit_order"]),
    }
    if method == "direct":
        latent = transport_direct(bundle, anchor, baseline_time, target_time, label_ad)
        prediction_info["transport"] = "direct"
        prediction_info["latent_l2_norm"] = float(np.linalg.norm(latent.detach().cpu().numpy()))
        return (
            decode_mesh(
                bundle,
                latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            ),
            prediction_info,
        )
    if method == "composed":
        intermediate_times = select_composed_intermediate_times(
            subject_rows, observation_visits=observation_visits, target_visit_order=int(target_row["visit_order"])
        )
        latent = transport_composed(
            bundle,
            anchor,
            baseline_time,
            intermediate_times=intermediate_times,
            target_time=target_time,
            target_label_ad=label_ad,
        )
        prediction_info["transport"] = "composed"
        prediction_info["intermediate_times"] = intermediate_times
        prediction_info["latent_l2_norm"] = float(np.linalg.norm(latent.detach().cpu().numpy()))
        return (
            decode_mesh(
                bundle,
                latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            ),
            prediction_info,
        )
    if method == "no_change":
        prediction_info["transport"] = "identity_latest_observed_mesh"
        prediction_info["latest_observed_scan_id"] = str(latest_observed_row["scan_id"])
        return load_mesh(latest_observed_row["mesh_path"]).copy(), prediction_info
    if method == "task2_upper_bound":
        prediction_info["transport"] = "task2_target_fit_reference_only"
        return load_task2_reference_mesh(bundle.experiment_dir, str(target_row["scan_id"])), prediction_info
    raise ValueError(f"Unsupported forecast method: {method}")


def evaluate_subject_tasks(
    bundle,
    subject_id: str,
    subject_rows: pd.DataFrame,
    subject_index: int,
    methods: Sequence[str],
    seed: int,
    anchor_fit_steps: int,
    anchor_fit_samples: int,
    anchor_fit_lr: float,
    anchor_init_std: float,
    mesh_resolution: int,
    mesh_max_batch: int,
    metric_samples: int,
    selected_subjects_for_mesh_export: Optional[Sequence[str]] = None,
    progress_prefix: Optional[str] = None,
) -> List[Dict[str, object]]:
    tasks = build_subject_tasks(subject_rows)
    if not tasks:
        if progress_prefix:
            log_progress(f"{progress_prefix}: skipped because no valid forecast tasks are available")
        return []

    anchor_cache: Dict[Tuple[int, ...], Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    subject_rows = subject_rows.sort_values("visit_order").reset_index(drop=True)
    selected_subjects = set(str(s) for s in (selected_subjects_for_mesh_export or []))
    mesh_manifest: Dict[str, object] = {
        "subject_id": subject_id,
        "diagnosis": str(subject_rows.iloc[0]["diagnosis"]),
        "exports": [],
    }

    for task_index, task in enumerate(tasks):
        observation_visits = tuple(int(v) for v in task["observation_visits"])
        cache_key = observation_visits
        if cache_key not in anchor_cache:
            observation_rows = subject_rows.loc[
                subject_rows["visit_order"].isin(observation_visits)
            ].sort_values("visit_order")
            obs_seed = observation_seed(seed, subject_index, observation_visits)
            if progress_prefix:
                log_progress(
                    f"{progress_prefix}: fitting anchor for visits [{','.join(str(v) for v in observation_visits)}]"
                )
            anchor, loss_hist, observations = fit_subject_anchor(
                bundle,
                observation_rows["scan_id"].tolist(),
                seed=obs_seed,
                num_iterations=anchor_fit_steps,
                num_samples=anchor_fit_samples,
                lr=anchor_fit_lr,
                init_std=anchor_init_std,
            )
            anchor_cache[cache_key] = {
                "anchor": anchor,
                "loss_hist": [float(v) for v in loss_hist],
                "observations": observations,
                "baseline_time": float(anchor_baseline_time(observations)),
                "observation_rows": observation_rows.reset_index(drop=True),
            }

        cache = anchor_cache[cache_key]
        observation_rows = cache["observation_rows"]
        latest_observed_row = observation_rows.sort_values("visit_order").iloc[-1]
        target_row = subject_rows.loc[
            subject_rows["visit_order"] == int(task["target_visit_order"])
        ].iloc[0]
        gt_mesh = load_mesh(target_row["mesh_path"])
        target_months = float(target_row["months_from_baseline"])
        latest_months = float(latest_observed_row["months_from_baseline"])
        base_row = {
            "checkpoint": bundle.checkpoint,
            "checkpoint_epoch": int(bundle.checkpoint_epoch),
            "subject_id": str(subject_id),
            "diagnosis": str(target_row["diagnosis"]),
            "split": str(target_row["split"]),
            "observation_mode": str(task["observation_mode"]),
            "observation_visits": ",".join(str(v) for v in observation_visits),
            "observation_scan_ids": "|".join(str(v) for v in observation_rows["scan_id"].tolist()),
            "num_observed_scans": int(len(observation_rows)),
            "latest_observed_scan_id": str(latest_observed_row["scan_id"]),
            "latest_observed_visit_order": int(latest_observed_row["visit_order"]),
            "latest_observed_months_from_baseline": latest_months,
            "target_scan_id": str(target_row["scan_id"]),
            "target_visit_order": int(target_row["visit_order"]),
            "target_label": str(task["target_label"]),
            "target_months_from_baseline": target_months,
            "horizon_months_from_latest_observed": target_months - latest_months,
            "horizon_months_from_baseline": target_months - float(subject_rows.iloc[0]["months_from_baseline"]),
            "anchor_fit_loss_start": float(cache["loss_hist"][0]) if cache["loss_hist"] else float("nan"),
            "anchor_fit_loss_end": float(cache["loss_hist"][-1]) if cache["loss_hist"] else float("nan"),
            "anchor_fit_steps": int(anchor_fit_steps),
            "anchor_fit_seed": observation_seed(seed, subject_index, observation_visits),
        }

        task_methods = [m for m in methods if task["allows_composed"] or m != "composed"]
        for method_index, method in enumerate(task_methods):
            if progress_prefix:
                log_progress(
                    f"{progress_prefix}: {method} forecast for {task['target_label']} "
                    f"(target visit {int(task['target_visit_order'])})"
                )
            pred_mesh, prediction_info = make_prediction_mesh(
                bundle=bundle,
                method=method,
                anchor=cache["anchor"],
                baseline_time=float(cache["baseline_time"]),
                target_row=target_row,
                latest_observed_row=latest_observed_row,
                subject_rows=subject_rows,
                observation_visits=observation_visits,
                mesh_resolution=mesh_resolution,
                mesh_max_batch=mesh_max_batch,
            )
            metrics = deterministic_mesh_metrics(
                gt_mesh,
                pred_mesh,
                num_samples=metric_samples,
                seed=task_metrics_seed(subject_index, task_index, method_index, seed),
                align_mode=bundle.align_mode,
                align_iters=bundle.align_iters,
                align_trim_quantile=bundle.align_trim_quantile,
            )
            export_path = ""
            if subject_id in selected_subjects and method in ("direct", "composed"):
                subject_mesh_dir = ensure_dir(forecast_mesh_dir(bundle.experiment_dir) / subject_id)
                mesh_name = (
                    f"{task['observation_mode']}_{task['target_label']}_{method}_{target_row['scan_id']}.ply"
                )
                export_path = export_mesh(pred_mesh, subject_mesh_dir / mesh_name)
                mesh_manifest["exports"].append(
                    {
                        "observation_mode": str(task["observation_mode"]),
                        "target_label": str(task["target_label"]),
                        "method": method,
                        "path": export_path,
                        "target_scan_id": str(target_row["scan_id"]),
                        "latest_observed_scan_id": str(latest_observed_row["scan_id"]),
                    }
                )

            row = dict(base_row)
            row.update(
                {
                    "method": method,
                    "is_forecast_method": method in ("direct", "composed", "no_change"),
                    "prediction_mesh_path": export_path,
                    "target_mesh_path": str(Path(target_row["mesh_path"]).resolve()),
                    "reference_task2_mesh_path": str(
                        (
                            task2_reference_root(bundle.experiment_dir)
                            / "reconstructed_meshes"
                            / f"{target_row['scan_id']}.ply"
                        ).resolve()
                    )
                    if method == "task2_upper_bound"
                    else "",
                }
            )
            row.update(prediction_info)
            row.update(metrics)
            row["pred_minus_target_volume"] = float(
                row["pred_volume"] - row["target_volume"]
            )
            rows.append(row)
            if progress_prefix:
                log_progress(
                    f"{progress_prefix}: finished {method} {task['target_label']} "
                    f"with aligned Chamfer {float(row['chamfer_aligned']):.6f}"
                )

    if mesh_manifest["exports"]:
        export_selected_subject_manifest(bundle.experiment_dir, subject_id, mesh_manifest)
    return rows


def summarize_validation_rows(frame: pd.DataFrame) -> Dict[str, float]:
    direct = frame.loc[frame["method"] == "direct"].copy()
    if direct.empty:
        return {
            "num_rows": 0,
            "num_subjects": 0,
            "future_chamfer_aligned_mean": float("nan"),
            "future_assd_aligned_mean": float("nan"),
            "future_hd95_aligned_mean": float("nan"),
        }
    return {
        "num_rows": int(len(direct)),
        "num_subjects": int(direct["subject_id"].nunique()),
        "future_chamfer_aligned_mean": float(direct["chamfer_aligned"].mean()),
        "future_assd_aligned_mean": float(direct["assd_aligned"].mean()),
        "future_hd95_aligned_mean": float(direct["hd95_aligned"].mean()),
    }


def run_validation(args: argparse.Namespace) -> Dict[str, object]:
    experiment_dir = args.experiment.resolve()
    out_dir = validation_dir(experiment_dir)
    checkpoints = list_model_checkpoints(experiment_dir)
    if not checkpoints:
        raise RuntimeError(
            f"No saved checkpoints found under {experiment_dir / 'ModelParameters'}. "
            "Run training first, then re-run validation selection."
        )
    log_progress(
        f"Validation mode: evaluating {len(checkpoints)} checkpoint(s) under {experiment_dir / 'ModelParameters'}"
    )
    log_progress(f"Validation outputs will be written under {out_dir}")

    validation_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    selected_payload: Optional[Dict[str, object]] = None

    total_checkpoints = len(checkpoints)
    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        log_progress(f"Validation checkpoint {checkpoint_index}/{total_checkpoints}: loading {checkpoint}")
        bundle = load_model_bundle(experiment_dir, checkpoint, device=args.device)
        ensure_cuda_bundle_device(bundle.device.type)
        val_frame = split_frame(bundle.metadata, "val")
        val_subjects = grouped_subject_rows(val_frame)
        rows: List[Dict[str, object]] = []
        total_subjects = len(val_subjects)
        for subject_index, (subject_id, subject_rows) in enumerate(val_subjects.items()):
            diagnosis = str(subject_rows.iloc[0]["diagnosis"])
            rows.extend(
                evaluate_subject_tasks(
                    bundle=bundle,
                    subject_id=subject_id,
                    subject_rows=subject_rows,
                    subject_index=subject_index,
                    methods=("direct",),
                    seed=args.seed,
                    anchor_fit_steps=args.anchor_fit_steps,
                    anchor_fit_samples=args.anchor_fit_samples,
                    anchor_fit_lr=args.anchor_fit_lr,
                    anchor_init_std=args.anchor_init_std,
                    mesh_resolution=args.mesh_resolution,
                    mesh_max_batch=args.mesh_max_batch,
                    metric_samples=args.metric_samples,
                    progress_prefix=(
                        f"validation checkpoint {checkpoint} "
                        f"subject {subject_index + 1}/{total_subjects} {subject_id} ({diagnosis})"
                    ),
                )
            )
        frame = pd.DataFrame(rows)
        validation_rows.extend(rows)
        score = summarize_validation_rows(frame)
        score.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": int(bundle.checkpoint_epoch),
            }
        )
        score_rows.append(score)
        log_progress(
            f"Validation checkpoint {checkpoint} complete: "
            f"mean future Chamfer {score['future_chamfer_aligned_mean']:.6f}, "
            f"mean ASSD {score['future_assd_aligned_mean']:.6f}"
        )

    score_df = pd.DataFrame(score_rows).sort_values(
        ["future_chamfer_aligned_mean", "future_assd_aligned_mean", "checkpoint_epoch"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best = score_df.iloc[0].to_dict()
    selected_payload = {
        "selected_checkpoint": str(best["checkpoint"]),
        "selected_epoch": int(best["checkpoint_epoch"]),
        "selection_metric": "future_chamfer_aligned_mean",
        "tie_breaker": "future_assd_aligned_mean",
        "num_validation_subjects": int(
            split_frame(load_metadata(experiment_dir)[0], "val")["subject_id"].nunique()
        ),
        "checkpoint_scores": score_df.to_dict(orient="records"),
    }
    score_df.to_csv(out_dir / "checkpoint_scores.csv", index=False)
    save_json(out_dir / "selected_checkpoint.json", selected_payload)
    if validation_rows:
        pd.DataFrame(validation_rows).to_csv(out_dir / "validation_direct_forecasts.csv", index=False)
    log_progress(
        f"Validation selection complete: chose checkpoint {selected_payload['selected_checkpoint']} "
        f"(epoch {selected_payload['selected_epoch']})"
    )
    return selected_payload


def summarize_test_results(frame: pd.DataFrame) -> pd.DataFrame:
    summary_frames: List[pd.DataFrame] = []
    metric_cols = ["chamfer_aligned", "assd_aligned", "hd95_aligned", "pred_minus_target_volume"]
    for (observation_mode, target_label, method), group in frame.groupby(
        ["observation_mode", "target_label", "method"], sort=True
    ):
        cohort_summary = compute_group_summary(group, metric_cols=metric_cols, seed=17)
        cohort_summary.insert(0, "method", method)
        cohort_summary.insert(0, "target_label", target_label)
        cohort_summary.insert(0, "observation_mode", observation_mode)
        summary_frames.append(cohort_summary)
    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)


def run_locked_test(args: argparse.Namespace, checkpoint: Optional[str] = None) -> Dict[str, object]:
    experiment_dir = args.experiment.resolve()
    out_dir = analysis_dir(experiment_dir)
    mesh_dir = forecast_mesh_dir(experiment_dir)
    requested_checkpoint = str(checkpoint or args.checkpoint)
    checkpoint_name = canonical_checkpoint_name(experiment_dir, requested_checkpoint)
    log_progress(
        f"Locked test mode: requested checkpoint '{requested_checkpoint}' resolved to '{checkpoint_name}'"
    )
    log_progress(f"Forecast outputs will be written under {out_dir}")
    log_progress(f"Selected forecast meshes will be written under {mesh_dir}")

    bundle = load_model_bundle(experiment_dir, checkpoint_name, device=args.device)
    ensure_cuda_bundle_device(bundle.device.type)
    test_frame = split_frame(bundle.metadata, "test")
    test_subjects = grouped_subject_rows(test_frame)
    selected_mesh_subjects = select_subjects_by_diagnosis(
        test_frame, per_group=args.selected_mesh_subjects_per_group
    )
    log_progress(
        f"Locked test will evaluate {len(test_subjects)} subject(s); mesh export subjects: "
        f"{', '.join(selected_mesh_subjects) if selected_mesh_subjects else 'none'}"
    )

    rows: List[Dict[str, object]] = []
    total_subjects = len(test_subjects)
    for subject_index, (subject_id, subject_rows) in enumerate(test_subjects.items()):
        diagnosis = str(subject_rows.iloc[0]["diagnosis"])
        rows.extend(
            evaluate_subject_tasks(
                bundle=bundle,
                subject_id=subject_id,
                subject_rows=subject_rows,
                subject_index=subject_index,
                methods=FORECAST_METHODS,
                seed=args.seed,
                anchor_fit_steps=args.anchor_fit_steps,
                anchor_fit_samples=args.anchor_fit_samples,
                anchor_fit_lr=args.anchor_fit_lr,
                anchor_init_std=args.anchor_init_std,
                mesh_resolution=args.mesh_resolution,
                mesh_max_batch=args.mesh_max_batch,
                metric_samples=args.metric_samples,
                selected_subjects_for_mesh_export=selected_mesh_subjects,
                progress_prefix=(
                    f"locked test checkpoint {checkpoint_name} "
                    f"subject {subject_index + 1}/{total_subjects} {subject_id} ({diagnosis})"
                ),
            )
        )

    per_scan_df = pd.DataFrame(rows).sort_values(
        ["subject_id", "observation_mode", "target_visit_order", "method"]
    ).reset_index(drop=True)
    per_subject_df = (
        per_scan_df.groupby(
            ["subject_id", "diagnosis", "split", "observation_mode", "method"], sort=True
        )
        .agg(
            num_targets=("target_scan_id", "count"),
            mean_chamfer_aligned=("chamfer_aligned", "mean"),
            mean_assd_aligned=("assd_aligned", "mean"),
            mean_hd95_aligned=("hd95_aligned", "mean"),
            mean_pred_minus_target_volume=("pred_minus_target_volume", "mean"),
        )
        .reset_index()
    )
    summary_df = summarize_test_results(per_scan_df)

    per_scan_path = out_dir / "test_forecast_per_scan.csv"
    per_subject_path = out_dir / "test_forecast_per_subject.csv"
    summary_path = out_dir / "test_forecast_summary.csv"
    per_scan_df.to_csv(per_scan_path, index=False)
    per_subject_df.to_csv(per_subject_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": bundle.checkpoint,
        "checkpoint_epoch": int(bundle.checkpoint_epoch),
        "num_rows": int(len(per_scan_df)),
        "num_subjects": int(per_scan_df["subject_id"].nunique()),
        "selected_mesh_subjects": selected_mesh_subjects,
        "files": {
            "per_scan": str(per_scan_path.resolve()),
            "per_subject": str(per_subject_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "selected_mesh_dir": str(forecast_mesh_dir(experiment_dir).resolve()),
        },
        "summary_rows": summary_df.to_dict(orient="records"),
    }
    save_json(out_dir / "test_forecast_summary.json", payload)
    log_progress(
        f"Locked test complete: wrote per-scan, per-subject, summary CSV, and summary JSON under {out_dir}"
    )
    return payload


def main() -> None:
    args = parse_args()
    args.experiment = args.experiment.resolve()
    log_progress(
        f"Starting forecast evaluation with mode={args.mode}, checkpoint={args.checkpoint}, "
        f"device={args.device}, experiment={args.experiment}"
    )
    seed_everything(args.seed)
    validation_report = load_validation_report(args.experiment)
    log_progress(
        f"Input validation status is {validation_report.get('status', 'unknown')} for {args.experiment}"
    )

    validation_payload: Optional[Dict[str, object]] = None
    if args.mode in ("validation", "both"):
        validation_payload = run_validation(args)

    if args.mode in ("test", "both"):
        checkpoint = None
        if args.mode == "both":
            checkpoint = str(validation_payload["selected_checkpoint"]) if validation_payload else None
        run_locked_test(args, checkpoint=checkpoint)
    log_progress("Forecast evaluation finished")


if __name__ == "__main__":
    main()
