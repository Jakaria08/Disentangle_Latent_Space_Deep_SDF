#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
OLD_HELPER_DIR = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI"
PCA_SCRIPT_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI_large_strict_left"
    / "task3_longitudinal_prediction"
    / "brainode_pca150_qc_stable"
    / "scripts"
)
for import_path in (SCRIPT_DIR, REPO_ROOT, OLD_HELPER_DIR, PCA_SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import direct_flow_rich_notebook_helpers as flow_helpers  # noqa: E402
import evaluate_pca_cocycle_flow as pca_eval  # noqa: E402
from core_brainode_common import TASK_DIR, load_config, resolve_repo_path  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/unified_longitudinal_visual_report/generated_artifacts"
)
DEFAULT_PCA_FLOW_DIR = (
    "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
    "pca150_direct_cocycle_flow_qc_v1"
)
DEFAULT_PCA_CHECKPOINT = "best_val_endpoint_vertex_mae"

SIREN_MODELS = {
    "qc_siren_drop_bad_min2": {
        "dataset": "qc_large",
        "family": "SIREN flow",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2"
        ),
    },
    "qc_siren_latent_ode": {
        "dataset": "qc_large",
        "family": "SIREN latent ODE",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1"
        ),
    },
    "qc_siren_local_decomp_volume": {
        "dataset": "qc_large",
        "family": "SIREN local decomposed volume",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_local_decomposed_flow_qc_volume_v1"
        ),
    },
    "qc_siren_seq_rollout": {
        "dataset": "qc_large",
        "family": "SIREN sequence rollout",
        "experiment_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing artifacts for the unified longitudinal report. "
            "The default path computes instantaneous diagonal velocity only."
        )
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pca150_flow", "qc_siren_drop_bad_min2", "qc_siren_latent_ode", "qc_siren_local_decomp_volume"],
        help="Models for instantaneous velocity. Use all for every supported model.",
    )
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--max-pairs-per-split", type=int, default=0)
    parser.add_argument("--finite-diff-eps-years", type=float, default=0.05)
    parser.add_argument("--skip-instantaneous-velocity", action="store_true")
    parser.add_argument("--pca-flow-dir", default=DEFAULT_PCA_FLOW_DIR)
    parser.add_argument("--pca-checkpoint", default=DEFAULT_PCA_CHECKPOINT)
    parser.add_argument("--pca-components", type=int, default=150)
    parser.add_argument("--pca-ood", action="store_true")
    parser.add_argument("--pca-ood-horizons-years", type=float, nargs="+", default=[1.0, 2.0, 4.0, 6.0, 10.0, 20.0])
    parser.add_argument("--pca-ood-composed-step-years", type=float, default=0.5)
    parser.add_argument("--max-ood-subjects-per-split", type=int, default=3)
    parser.add_argument("--subjects-per-diagnosis", type=int, default=25)
    parser.add_argument(
        "--selected-pair-source-mode",
        choices=("baseline_only", "all_sources"),
        default="baseline_only",
        help=(
            "For selected trend/surface cases, keep baseline-to-future pairs only "
            "or every source-to-future pair for the selected subjects."
        ),
    )
    parser.add_argument(
        "--surface-placeholders",
        action="store_true",
        help="Write selected surface-change case manifest only; does not decode meshes.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_device(value: str) -> torch.device:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    else:
        keys = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dataframe_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def string_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def old_adni_trend_pairs(path: Path) -> pd.DataFrame:
    frame = dataframe_csv(path)
    if frame.empty:
        return pd.DataFrame()
    required = {"split", "diagnosis", "subject_id", "base_scan_id", "scan_id", "visit_order"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.loc[frame["transport_method"].astype(str) == "real_observed"].copy()
    frame["visit_order"] = pd.to_numeric(frame["visit_order"], errors="coerce")
    frame = frame.loc[frame["visit_order"] > 0].copy()
    if frame.empty:
        return pd.DataFrame()
    output = pd.DataFrame(
        {
            "dataset": "old_adni",
            "split": frame["split"].astype(str),
            "diagnosis": frame["diagnosis"].astype(str),
            "label_ad": frame.get("label_ad", ""),
            "subject_id": frame["subject_id"].astype(str),
            "source_scan_id": frame["base_scan_id"].astype(str),
            "target_scan_id": frame["scan_id"].astype(str),
            "source_visit_order": 0,
            "target_visit_order": frame["visit_order"],
            "source_age_years": pd.to_numeric(frame.get("base_age_years", np.nan), errors="coerce"),
            "target_age_years": pd.to_numeric(frame.get("age_years", np.nan), errors="coerce"),
            "gap_years": pd.to_numeric(frame.get("years_from_baseline", np.nan), errors="coerce"),
            "pair_type": np.where(frame["visit_order"].astype(float) == 1.0, "adjacent", "nonadjacent"),
            "source_table": str(path),
        }
    )
    return output


def vector_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.reshape(1, -1), dim=1)[0].item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    denom = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)
    if float(denom.item()) <= 1.0e-12:
        return float("nan")
    return float((a * b).sum(dim=1).div(denom).item())


@torch.no_grad()
def flow_diag_velocity_per_year(
    flow: torch.nn.Module,
    latent: torch.Tensor,
    time: torch.Tensor,
    condition: torch.Tensor,
    age_range_years: float,
) -> torch.Tensor:
    if hasattr(flow, "instantaneous_velocity_per_year"):
        return flow.instantaneous_velocity_per_year(
            latent,
            time,
            condition,
            float(age_range_years),
        )
    if hasattr(flow, "average_velocity"):
        return flow.average_velocity(latent, time, time, condition) / float(age_range_years)
    raise TypeError(f"Flow does not expose instantaneous or average velocity: {type(flow)!r}")


@torch.no_grad()
def finite_difference_diag_velocity_per_year(
    flow: torch.nn.Module,
    latent: torch.Tensor,
    time: torch.Tensor,
    condition: torch.Tensor,
    age_range_years: float,
    eps_years: float,
) -> torch.Tensor:
    eps_norm = float(eps_years) / float(age_range_years)
    next_time = time + eps_norm
    predicted = flow.transport(latent, time, next_time, condition)
    return (predicted - latent) / float(eps_years)


def pair_type(source_order: int, target_order: int) -> str:
    return "adjacent" if int(target_order) - int(source_order) == 1 else "nonadjacent"


def selected_model_names(raw: Sequence[str]) -> list[str]:
    names = list(raw)
    if any(name == "all" for name in names):
        return ["pca150_flow", *SIREN_MODELS.keys()]
    return names


def limit_rows_by_split(rows: Iterable[Any], max_rows: int) -> Iterable[Any]:
    if int(max_rows) <= 0:
        yield from rows
        return
    for index, row in enumerate(rows):
        if index >= int(max_rows):
            break
        yield row


def collect_siren_instantaneous_velocity(
    *,
    model_name: str,
    spec: dict[str, str],
    splits: Sequence[str],
    checkpoint: str,
    device_value: str,
    max_pairs_per_split: int,
    eps_years: float,
) -> list[dict[str, Any]]:
    bundle = flow_helpers.load_bundle(
        repo_path(spec["experiment_dir"]),
        checkpoint=checkpoint,
        device=device_value,
    )
    flow = bundle.flow.eval()
    device = bundle.device
    age_range = flow_helpers.age_range_years(bundle)
    rows: list[dict[str, Any]] = []
    for split in splits:
        records = bundle.contract.pair_records.get(split, [])
        for record in limit_rows_by_split(records, max_pairs_per_split):
            source = bundle.contract.latent_maps[split][record.source_scan_id].to(device).view(1, -1)
            target = bundle.contract.latent_maps[split][record.target_scan_id].to(device).view(1, -1)
            source_time = torch.tensor([record.source_time], dtype=source.dtype, device=device)
            target_time = torch.tensor([record.target_time], dtype=source.dtype, device=device)
            condition = torch.tensor([float(record.label_ad)], dtype=source.dtype, device=device)
            gap_years = float(age_range) * float(record.target_time - record.source_time)
            if abs(gap_years) <= 1.0e-8:
                continue
            direct = flow.transport(source, source_time, target_time, condition)
            diag = flow_diag_velocity_per_year(flow, source, source_time, condition, age_range)
            fd_diag = finite_difference_diag_velocity_per_year(
                flow,
                source,
                source_time,
                condition,
                age_range,
                eps_years,
            )
            real_vec = (target - source) / gap_years
            average_pred_vec = (direct - source) / gap_years
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "model": model_name,
                    "family": spec["family"],
                    "split": split,
                    "subject_id": record.subject_id,
                    "diagnosis": record.diagnosis,
                    "label_ad": int(record.label_ad),
                    "condition_name": "observed_condition",
                    "condition_value": float(record.label_ad),
                    "source_scan_id": record.source_scan_id,
                    "target_scan_id": record.target_scan_id,
                    "source_visit_order": int(record.source_visit_order),
                    "target_visit_order": int(record.target_visit_order),
                    "pair_type": pair_type(record.source_visit_order, record.target_visit_order),
                    "source_age_norm": float(record.source_time),
                    "target_age_norm": float(record.target_time),
                    "gap_years": gap_years,
                    "age_range_years": float(age_range),
                    "instantaneous_definition": "diag_G_z_t_t_condition_per_year",
                    "finite_difference_eps_years": float(eps_years),
                    "instantaneous_diag_l2_per_year": vector_norm(diag),
                    "finite_difference_diag_l2_per_year": vector_norm(fd_diag),
                    "real_latent_velocity_l2_per_year": vector_norm(real_vec),
                    "average_predicted_velocity_l2_per_year": vector_norm(average_pred_vec),
                    "instantaneous_diag_cosine_with_real": cosine(diag, real_vec),
                    "finite_difference_diag_cosine_with_real": cosine(fd_diag, real_vec),
                    "average_predicted_cosine_with_real": cosine(average_pred_vec, real_vec),
                    "checkpoint": checkpoint,
                }
            )
    return rows


def age_range_from_archive(archive: dict[str, np.ndarray]) -> float:
    ages = archive["visit_continuous_age_years"].astype(float)
    norms = archive["visit_continuous_age_norm"].astype(float)
    pairs = []
    for i in range(len(ages) - 1):
        dn = norms[i + 1] - norms[i]
        da = ages[i + 1] - ages[i]
        if abs(dn) > 1.0e-8 and abs(da) > 1.0e-8:
            pairs.append(da / dn)
    if pairs:
        return float(np.median(pairs))
    train_archive = pca_eval.load_npz(TASK_DIR / "dataset" / "train_subject_sequences.npz")
    age_min = float(np.min(train_archive["visit_continuous_age_years"]))
    age_max = float(np.max(train_archive["visit_continuous_age_years"]))
    return age_max - age_min


def load_pca_flow(
    *,
    pca_flow_dir: Path,
    checkpoint: str,
    components: int,
    device: torch.device,
):
    checkpoint_path = pca_flow_dir / "checkpoints" / (
        checkpoint if checkpoint.endswith(".pth") else f"{checkpoint}.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing PCA-flow checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    flow = pca_eval.make_flow_from_checkpoint(payload, int(components)).to(device)
    flow.load_state_dict(payload["model_state_dict"])
    flow.eval()
    return flow, checkpoint_path


def collect_pca_instantaneous_velocity(
    *,
    pca_flow_dir: Path,
    checkpoint: str,
    components: int,
    splits: Sequence[str],
    device: torch.device,
    max_pairs_per_split: int,
    eps_years: float,
) -> list[dict[str, Any]]:
    flow, checkpoint_path = load_pca_flow(
        pca_flow_dir=pca_flow_dir,
        checkpoint=checkpoint,
        components=components,
        device=device,
    )
    scan_manifest = {
        row["scan_id"]: row for row in read_csv(TASK_DIR / "metadata" / "core_brainode_scan_manifest.csv")
    }
    rows: list[dict[str, Any]] = []
    for split in splits:
        archive = pca_eval.load_npz(TASK_DIR / "dataset" / f"{split}_subject_sequences.npz")
        age_range = age_range_from_archive(archive)
        for pair, _times_np, latents_np, conditions_np in pca_eval.iter_pairs(
            archive=archive,
            split=split,
            scan_manifest=scan_manifest,
            components=int(components),
            pair_type="all",
            max_pairs=int(max_pairs_per_split),
        ):
            source = torch.from_numpy(latents_np[0].astype(np.float32)).view(1, -1).to(device)
            target = torch.from_numpy(latents_np[-1].astype(np.float32)).view(1, -1).to(device)
            source_time = torch.tensor([pair.source_age_norm], dtype=source.dtype, device=device)
            target_time = torch.tensor([pair.target_age_norm], dtype=source.dtype, device=device)
            condition = torch.tensor([float(conditions_np[0])], dtype=source.dtype, device=device)
            gap_years = float(pair.gap_years)
            if abs(gap_years) <= 1.0e-8:
                continue
            direct = flow.transport(source, source_time, target_time, condition)
            diag = flow_diag_velocity_per_year(flow, source, source_time, condition, age_range)
            fd_diag = finite_difference_diag_velocity_per_year(
                flow,
                source,
                source_time,
                condition,
                age_range,
                eps_years,
            )
            real_vec = (target - source) / gap_years
            average_pred_vec = (direct - source) / gap_years
            rows.append(
                {
                    "dataset": "qc_large",
                    "model": "pca150_direct_cocycle_flow",
                    "family": "PCA150 conditional cocycle flow",
                    "split": split,
                    "subject_id": pair.subject_id,
                    "diagnosis": pair.diagnosis,
                    "label_ad": int(pair.label_ad),
                    "condition_name": "observed_condition",
                    "condition_value": float(conditions_np[0]),
                    "source_scan_id": pair.source_scan_id,
                    "target_scan_id": pair.target_scan_id,
                    "source_visit_order": int(pair.source_visit_order),
                    "target_visit_order": int(pair.target_visit_order),
                    "pair_type": pair.pair_type,
                    "source_age_norm": float(pair.source_age_norm),
                    "target_age_norm": float(pair.target_age_norm),
                    "gap_years": gap_years,
                    "age_range_years": float(age_range),
                    "instantaneous_definition": "diag_G_z_t_t_condition_per_year",
                    "finite_difference_eps_years": float(eps_years),
                    "instantaneous_diag_l2_per_year": vector_norm(diag),
                    "finite_difference_diag_l2_per_year": vector_norm(fd_diag),
                    "real_latent_velocity_l2_per_year": vector_norm(real_vec),
                    "average_predicted_velocity_l2_per_year": vector_norm(average_pred_vec),
                    "instantaneous_diag_cosine_with_real": cosine(diag, real_vec),
                    "finite_difference_diag_cosine_with_real": cosine(fd_diag, real_vec),
                    "average_predicted_cosine_with_real": cosine(average_pred_vec, real_vec),
                    "checkpoint": str(checkpoint_path),
                }
            )
    return rows


def pca_model_arrays(components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = load_config(TASK_DIR / "configs" / "core_brainode.json")
    pca_model_dir = resolve_repo_path(config["task2"]["pca_model_dir"])
    mean_flat = np.load(pca_model_dir / "mean.npy").astype(np.float32)
    pca_components = np.load(pca_model_dir / "components_256.npy").astype(np.float32)[: int(components)]
    faces = np.load(pca_model_dir / "faces.npy").astype(np.int64)
    return mean_flat, pca_components, faces


def collect_pca_ood(
    *,
    pca_flow_dir: Path,
    checkpoint: str,
    components: int,
    splits: Sequence[str],
    device: torch.device,
    horizons_years: Sequence[float],
    composed_step_years: float,
    max_subjects_per_split: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flow, checkpoint_path = load_pca_flow(
        pca_flow_dir=pca_flow_dir,
        checkpoint=checkpoint,
        components=components,
        device=device,
    )
    mean_flat, pca_components, faces = pca_model_arrays(components)
    args = SimpleNamespace(
        splits=list(splits),
        components=int(components),
        ood_horizons_years=[float(value) for value in horizons_years],
        ood_composed_step_years=float(composed_step_years),
        max_ood_subjects_per_split=int(max_subjects_per_split),
    )
    rows, summary = pca_eval.build_ood_forecasts(
        flow=flow,
        args=args,
        device=device,
        mean_flat=mean_flat,
        components=pca_components,
        faces=faces,
    )
    for row in rows:
        row["dataset"] = "qc_large"
        row["model"] = "pca150_direct_cocycle_flow"
        row["family"] = "PCA150 conditional cocycle flow"
        row["checkpoint"] = str(checkpoint_path)
    for row in summary:
        row["dataset"] = "qc_large"
        row["model"] = "pca150_direct_cocycle_flow"
        row["family"] = "PCA150 conditional cocycle flow"
        row["checkpoint"] = str(checkpoint_path)
    return rows, summary


def select_surface_case_manifest(
    *,
    output_dir: Path,
    subjects_per_diagnosis: int = 25,
    splits: Sequence[str] = ("train", "val", "test"),
    pair_source_mode: str = "baseline_only",
) -> list[dict[str, Any]]:
    """Write a balanced manifest for later heavy surface-change decoding.

    This intentionally does not decode or save meshes. It selects up to N AD and
    N CN subjects per dataset/split, then keeps the matching ground-truth future
    pairs for those exact subjects.
    """
    fair_path = (
        REPO_ROOT
        / "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
        / "analysis/future_mesh_forecast_comparison/future_mesh_per_pair.csv"
    )
    pca_path = (
        REPO_ROOT
        / DEFAULT_PCA_FLOW_DIR
        / "analysis/checkpoint_best_val_endpoint_vertex_mae/pca_flow_per_pair.csv"
    )
    old_trend_path = (
        REPO_ROOT
        / "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized/"
        / "analysis/notebook_best/batch_observed_age_volume_trend.csv"
    )
    frames = []
    for path in (fair_path, pca_path):
        if path.is_file():
            frame = dataframe_csv(path)
            frame["source_table"] = str(path)
            frames.append(frame)
    old_pairs = old_adni_trend_pairs(old_trend_path)
    if not old_pairs.empty:
        frames.append(old_pairs)
    if not frames:
        return []
    frame = pd.concat(frames, ignore_index=True, sort=False)
    needed = {"dataset", "split", "diagnosis", "subject_id", "source_scan_id", "target_scan_id"}
    if not needed.issubset(frame.columns):
        return []
    frame = frame.loc[frame["transport_method"].astype(str) != "model_no_change"].copy()
    frame = string_columns(frame, ["dataset", "split", "diagnosis", "subject_id", "source_scan_id", "target_scan_id"])
    for column in ["source_visit_order", "target_visit_order", "source_age_years", "target_age_years", "gap_years"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    key_cols = [
        "dataset",
        "split",
        "diagnosis",
        "label_ad",
        "subject_id",
        "source_scan_id",
        "target_scan_id",
        "source_visit_order",
        "target_visit_order",
        "source_mesh_path",
        "target_mesh_path",
        "source_age_years",
        "target_age_years",
        "gap_years",
        "pair_type",
    ]
    key_cols = [column for column in key_cols if column in frame.columns]
    dedupe_cols = [
        "dataset",
        "split",
        "diagnosis",
        "subject_id",
        "source_scan_id",
        "target_scan_id",
    ]
    dedupe_cols = [column for column in dedupe_cols if column in frame.columns]
    sort_cols = [column for column in ["target_visit_order", "source_visit_order"] if column in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    pair_frame = frame.drop_duplicates(dedupe_cols).copy()

    if pair_source_mode == "baseline_only" and "source_visit_order" in pair_frame.columns:
        min_source = pair_frame.groupby(["dataset", "split", "subject_id"])["source_visit_order"].transform("min")
        pair_frame = pair_frame.loc[pair_frame["source_visit_order"] == min_source].copy()

    stats_rows: list[dict[str, Any]] = []
    subject_group_cols = ["dataset", "split", "diagnosis", "subject_id"]
    for key, group in pair_frame.groupby(subject_group_cols, dropna=False, sort=True):
        dataset, split, diagnosis, subject_id = key
        ages = pd.concat(
            [
                pd.to_numeric(group.get("source_age_years", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(group.get("target_age_years", pd.Series(dtype=float)), errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        stats_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "diagnosis": diagnosis,
                "subject_id": subject_id,
                "candidate_pairs": int(len(group)),
                "candidate_targets": int(group["target_scan_id"].nunique()) if "target_scan_id" in group else 0,
                "max_gap_years": float(pd.to_numeric(group.get("gap_years", pd.Series(dtype=float)), errors="coerce").max())
                if "gap_years" in group
                else float("nan"),
                "first_age_years": float(ages.min()) if len(ages) else float("nan"),
                "last_age_years": float(ages.max()) if len(ages) else float("nan"),
            }
        )

    if not stats_rows:
        write_csv(output_dir / "selected_subject_manifest.csv", [])
        write_csv(output_dir / "surface_change_case_manifest.csv", [])
        write_csv(output_dir / "selected_subject_coverage.csv", [])
        return []

    subject_frame = pd.DataFrame(stats_rows)
    selected_subjects: list[dict[str, Any]] = []
    for dataset in sorted(subject_frame["dataset"].astype(str).unique().tolist()):
        for split in splits:
            for diagnosis in ("AD", "CN"):
                subset = subject_frame.loc[
                    (subject_frame["dataset"].astype(str) == str(dataset))
                    & (subject_frame["split"].astype(str) == str(split))
                    & (subject_frame["diagnosis"].astype(str) == diagnosis)
                ].copy()
                if subset.empty:
                    continue
                subset = subset.sort_values(
                    ["candidate_targets", "candidate_pairs", "max_gap_years", "subject_id"],
                    ascending=[False, False, False, True],
                )
                for rank, (_, row) in enumerate(subset.head(int(subjects_per_diagnosis)).iterrows(), start=1):
                    selected = row.to_dict()
                    selected["selected_rank"] = int(rank)
                    selected["requested_subjects_per_diagnosis"] = int(subjects_per_diagnosis)
                    selected["pair_source_mode"] = pair_source_mode
                    selected_subjects.append(selected)

    selected_subject_frame = pd.DataFrame(selected_subjects)
    if selected_subject_frame.empty:
        write_csv(output_dir / "selected_subject_manifest.csv", [])
        write_csv(output_dir / "surface_change_case_manifest.csv", [])
        return []

    selected_keys = selected_subject_frame[["dataset", "split", "diagnosis", "subject_id"]].copy()
    selected_keys = string_columns(selected_keys, ["dataset", "split", "diagnosis", "subject_id"])
    selected_pairs = pair_frame.merge(selected_keys, on=["dataset", "split", "diagnosis", "subject_id"], how="inner")
    selected_pairs = selected_pairs.sort_values(
        [
            "dataset",
            "split",
            "diagnosis",
            "subject_id",
            "source_visit_order" if "source_visit_order" in selected_pairs.columns else "source_scan_id",
            "target_visit_order" if "target_visit_order" in selected_pairs.columns else "target_scan_id",
        ]
    )
    pair_rows = selected_pairs.loc[:, key_cols].to_dict("records")
    subject_rows = selected_subject_frame.sort_values(
        ["dataset", "split", "diagnosis", "selected_rank"]
    ).to_dict("records")
    write_csv(output_dir / "selected_subject_manifest.csv", subject_rows)
    write_csv(output_dir / "surface_change_case_manifest.csv", pair_rows)
    coverage_rows: list[dict[str, Any]] = []
    for (dataset, split, diagnosis), group in selected_subject_frame.groupby(["dataset", "split", "diagnosis"], sort=True):
        pair_count = int(
            selected_pairs.loc[
                (selected_pairs["dataset"].astype(str) == str(dataset))
                & (selected_pairs["split"].astype(str) == str(split))
                & (selected_pairs["diagnosis"].astype(str) == str(diagnosis))
            ].shape[0]
        )
        coverage_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "diagnosis": diagnosis,
                "selected_subjects": int(len(group)),
                "requested_subjects": int(subjects_per_diagnosis),
                "selected_pairs": pair_count,
                "pair_source_mode": pair_source_mode,
            }
        )
    write_csv(output_dir / "selected_subject_coverage.csv", coverage_rows)
    return pair_rows


def summarize_velocity(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    metrics = [
        "instantaneous_diag_l2_per_year",
        "finite_difference_diag_l2_per_year",
        "real_latent_velocity_l2_per_year",
        "average_predicted_velocity_l2_per_year",
        "instantaneous_diag_cosine_with_real",
        "finite_difference_diag_cosine_with_real",
        "average_predicted_cosine_with_real",
    ]
    group_cols = ["dataset", "model", "family", "split", "diagnosis", "pair_type", "condition_name"]
    output: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        row = {column: value for column, value in zip(group_cols, key)}
        row["rows"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_median"] = float(values.median()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else float("nan")
        output.append(row)
    return output


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model_names = selected_model_names(args.models)

    run_manifest: dict[str, Any] = {
        "output_dir": str(output_dir),
        "device": str(device),
        "splits": list(args.splits),
        "models_requested": model_names,
        "finite_difference_eps_years": float(args.finite_diff_eps_years),
    }

    velocity_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not args.skip_instantaneous_velocity:
        for model_name in model_names:
            try:
                if model_name == "pca150_flow":
                    velocity_rows.extend(
                        collect_pca_instantaneous_velocity(
                            pca_flow_dir=repo_path(args.pca_flow_dir),
                            checkpoint=args.pca_checkpoint,
                            components=int(args.pca_components),
                            splits=args.splits,
                            device=device,
                            max_pairs_per_split=int(args.max_pairs_per_split),
                            eps_years=float(args.finite_diff_eps_years),
                        )
                    )
                elif model_name in SIREN_MODELS:
                    velocity_rows.extend(
                        collect_siren_instantaneous_velocity(
                            model_name=model_name,
                            spec=SIREN_MODELS[model_name],
                            splits=args.splits,
                            checkpoint=args.checkpoint,
                            device_value=args.device,
                            max_pairs_per_split=int(args.max_pairs_per_split),
                            eps_years=float(args.finite_diff_eps_years),
                        )
                    )
                else:
                    failures.append(
                        {
                            "artifact": "instantaneous_velocity",
                            "model": model_name,
                            "error": f"unsupported model {model_name}",
                        }
                    )
            except Exception as exc:
                failures.append(
                    {
                        "artifact": "instantaneous_velocity",
                        "model": model_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        write_csv(output_dir / "instantaneous_velocity.csv", velocity_rows)
        write_csv(output_dir / "instantaneous_velocity_summary.csv", summarize_velocity(velocity_rows))
        run_manifest["instantaneous_velocity_rows"] = len(velocity_rows)

    if args.pca_ood:
        try:
            ood_rows, ood_summary = collect_pca_ood(
                pca_flow_dir=repo_path(args.pca_flow_dir),
                checkpoint=args.pca_checkpoint,
                components=int(args.pca_components),
                splits=args.splits,
                device=device,
                horizons_years=args.pca_ood_horizons_years,
                composed_step_years=float(args.pca_ood_composed_step_years),
                max_subjects_per_split=int(args.max_ood_subjects_per_split),
            )
            write_csv(output_dir / "pca_ood_forecasts.csv", ood_rows)
            write_csv(output_dir / "pca_ood_summary.csv", ood_summary)
            run_manifest["pca_ood_rows"] = len(ood_rows)
            run_manifest["pca_ood_summary_rows"] = len(ood_summary)
        except Exception as exc:
            failures.append(
                {
                    "artifact": "pca_ood",
                    "model": "pca150_flow",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if args.surface_placeholders:
        rows = select_surface_case_manifest(
            output_dir=output_dir,
            subjects_per_diagnosis=int(args.subjects_per_diagnosis),
            splits=args.splits,
            pair_source_mode=str(args.selected_pair_source_mode),
        )
        run_manifest["surface_change_case_manifest_rows"] = len(rows)
        run_manifest["subjects_per_diagnosis"] = int(args.subjects_per_diagnosis)
        run_manifest["selected_pair_source_mode"] = str(args.selected_pair_source_mode)

    write_csv(output_dir / "artifact_failures.csv", failures, fieldnames=["artifact", "model", "error"])
    run_manifest["failure_rows"] = len(failures)
    (output_dir / "run.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote artifacts to: {output_dir}")
    if failures:
        print(f"Failures: {len(failures)}; see {output_dir / 'artifact_failures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
