#!/usr/bin/env python3
"""Prepare a persistent, load-only cache for the BrainODE-vs-cocycle notebook.

The script reuses the unified registered-mesh evaluator output for scalar
metrics.  It performs model inference only for counterfactual trajectories,
selected example meshes, and diagnosis-level change maps.  Every result used by
the notebook is written as JSON or CSV; the notebook itself never loads a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from evaluate_unified_baselines import load_brainode, v4_checkpoint
from train_v1_anchored_ad_speed import make_model
from v1_speed_utils import (
    EXPERIMENT_DIR,
    PairRecord,
    build_pair_records,
    decode_pca_np,
    file_sha256,
    load_config,
    load_pca_model,
    load_split_archive,
    mesh_volume_np,
    pca_latents,
    pearson,
    resolve_device,
    resolve_repo_path,
    top_change_dice,
    vertex_normals_np,
)


SCHEMA_VERSION = 1
MODEL_LABELS = {
    "brainode": "PCA BrainODE",
    "v4": "PCA Cocycle/Consistency v4",
}
METRICS: dict[str, dict[str, Any]] = {
    "endpoint_vertex_mae": {"label": "Vertex MAE", "lower_is_better": True},
    "endpoint_pca_mse": {"label": "PCA MSE", "lower_is_better": True},
    "endpoint_vertex_hd95": {"label": "Vertex HD95", "lower_is_better": True},
    "volume_relative_error": {"label": "Volume relative error", "lower_is_better": True},
    "log_volume_rate_abs_error": {"label": "Log-volume-rate error", "lower_is_better": True},
    "local_normal_rate_mae": {"label": "Local normal-rate MAE", "lower_is_better": True},
    "local_normal_rate_pearson": {"label": "Local normal-rate Pearson", "lower_is_better": False},
    "local_normal_top_change_dice": {"label": "Top-change Dice", "lower_is_better": False},
}
REQUIRED_TABLES = (
    "cohort_counts.csv",
    "metric_summary.csv",
    "paired_metric_comparison.csv",
    "paired_bootstrap_ci.csv",
    "gap_summary.csv",
    "subject_volume_trends.csv",
    "cohort_volume_trend_summary.csv",
    "selected_volume_trajectories.csv",
    "counterfactual_trajectories.csv",
    "selected_mesh_metrics.csv",
    "group_hotspot_metrics.csv",
)
REQUIRED_MESHES = (
    "template_faces.csv",
    "selected_cn_vertices.csv",
    "selected_ad_vertices.csv",
    "group_cn_change_maps.csv",
    "group_ad_change_maps.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "v1_anchored_ad_speed_primary.json"))
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--run-name", default="v1_anchor_ad_speed_seed42")
    parser.add_argument("--checkpoint", default="best_feasible_volume")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--example-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-map-max-pairs-per-diagnosis", type=int, default=0)
    parser.add_argument("--skip-group-maps", action="store_true")
    parser.add_argument("--unified-cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_subject_id(value: Any) -> str:
    """Return a stable key without turning integer IDs into strings like 113.0."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_files(output_dir: Path) -> list[Path]:
    files = [output_dir / "manifest.json", output_dir / "provenance.json", output_dir / "selected_examples.json"]
    files.extend(output_dir / "tables" / name for name in REQUIRED_TABLES)
    files.extend(output_dir / "meshes" / name for name in REQUIRED_MESHES)
    return files


def manifest_is_complete(output_dir: Path, provenance: dict[str, Any]) -> bool:
    manifest_path = output_dir / "manifest.json"
    provenance_path = output_dir / "provenance.json"
    if not manifest_path.is_file() or not provenance_path.is_file():
        return False
    manifest = load_json(manifest_path)
    old_provenance = load_json(provenance_path)
    return (
        manifest.get("status") == "complete"
        and old_provenance == provenance
        and all(path.is_file() for path in expected_files(output_dir))
    )


def add_all_split(frame: pd.DataFrame) -> pd.DataFrame:
    combined = frame.copy()
    combined["split"] = "all"
    return pd.concat([frame, combined], ignore_index=True)


def prepare_pair_frame(unified_cache_dir: Path, splits: Sequence[str]) -> pd.DataFrame:
    pair_path = unified_cache_dir / "unified_registered_per_pair.csv"
    run_path = unified_cache_dir / "run.json"
    if not pair_path.is_file() or not run_path.is_file():
        raise FileNotFoundError(
            "The unified all-data cache is missing. Run evaluate_unified_baselines.py first.\n"
            f"Expected: {pair_path}\nExpected: {run_path}"
        )
    frame = pd.read_csv(pair_path)
    required = {
        "split",
        "diagnosis",
        "subject_id",
        "model",
        "transport_method",
        "direction",
        "source_visit_order",
        "target_visit_order",
    }.union(METRICS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Unified cache is missing columns: {missing}")
    frame = frame[
        frame["model"].isin(MODEL_LABELS)
        & frame["split"].isin(splits)
        & (frame["transport_method"] == "direct")
        & (frame["direction"] == "forward")
    ].copy()
    frame["subject_id_raw"] = frame["subject_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].map(canonical_subject_id)
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    if frame.empty:
        raise ValueError("No direct forward BrainODE/v4 rows remain after filtering")
    return frame


def pair_key_columns() -> list[str]:
    return [
        "split",
        "diagnosis",
        "subject_id",
        "source_visit_order",
        "target_visit_order",
        "gap_bin",
    ]


def paired_metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    keys = pair_key_columns()
    for metric, spec in METRICS.items():
        pivot = frame.pivot_table(index=keys, columns="model", values=metric, aggfunc="mean").dropna().reset_index()
        if not {"brainode", "v4"}.issubset(pivot.columns):
            continue
        lower = bool(spec["lower_is_better"])
        pivot["metric"] = metric
        pivot["metric_label"] = str(spec["label"])
        pivot["direction"] = "lower" if lower else "higher"
        pivot["direction_symbol"] = "↓" if lower else "↑"
        pivot["brainode_value"] = pivot["brainode"]
        pivot["cocycle_value"] = pivot["v4"]
        pivot["delta_favoring_cocycle"] = (
            pivot["brainode_value"] - pivot["cocycle_value"]
            if lower
            else pivot["cocycle_value"] - pivot["brainode_value"]
        )
        pivot["relative_gain_percent"] = np.where(
            lower,
            100.0 * pivot["delta_favoring_cocycle"] / pivot["brainode_value"].replace(0.0, np.nan),
            np.nan,
        )
        pivot["cocycle_wins"] = (pivot["delta_favoring_cocycle"] > 0.0).astype(int)
        rows.append(
            pivot[
                keys
                + [
                    "metric",
                    "metric_label",
                    "direction",
                    "direction_symbol",
                    "brainode_value",
                    "cocycle_value",
                    "delta_favoring_cocycle",
                    "relative_gain_percent",
                    "cocycle_wins",
                ]
            ]
        )
    paired = pd.concat(rows, ignore_index=True)
    expanded = add_all_split(paired)
    summary = (
        expanded.groupby(["split", "diagnosis", "metric", "metric_label", "direction", "direction_symbol"], as_index=False)
        .agg(
            pairs=("delta_favoring_cocycle", "size"),
            brainode_mean=("brainode_value", "mean"),
            cocycle_mean=("cocycle_value", "mean"),
            delta_favoring_cocycle_mean=("delta_favoring_cocycle", "mean"),
            delta_favoring_cocycle_median=("delta_favoring_cocycle", "median"),
            relative_gain_percent=("relative_gain_percent", "mean"),
            cocycle_win_fraction=("cocycle_wins", "mean"),
        )
    )
    summary["better_model"] = np.where(
        summary["delta_favoring_cocycle_mean"] > 0.0,
        MODEL_LABELS["v4"],
        np.where(summary["delta_favoring_cocycle_mean"] < 0.0, MODEL_LABELS["brainode"], "Tie"),
    )
    return paired, summary


def cohort_counts(frame: pd.DataFrame) -> pd.DataFrame:
    expanded = add_all_split(frame)
    return (
        expanded.groupby(["split", "diagnosis", "model", "model_label"], as_index=False)
        .agg(pairs=("subject_id", "size"), subjects=("subject_id", "nunique"))
    )


def gap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["split", "diagnosis", "gap_bin", "model", "model_label"], as_index=False)
    return grouped[list(METRICS)].mean()


def bootstrap_tables(paired: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    expanded = add_all_split(paired)
    output: list[dict[str, Any]] = []
    for index, ((split, diagnosis, metric), group) in enumerate(expanded.groupby(["split", "diagnosis", "metric"], sort=True)):
        subject_values = group.groupby("subject_id")["delta_favoring_cocycle"].mean().dropna().to_numpy(dtype=np.float64)
        if subject_values.size == 0:
            continue
        rng = np.random.default_rng(int(seed) + index * 1009)
        sample_means: list[np.ndarray] = []
        remaining = int(replicates)
        chunk = 512
        while remaining > 0:
            count = min(chunk, remaining)
            draw = rng.integers(0, subject_values.size, size=(count, subject_values.size))
            sample_means.append(subject_values[draw].mean(axis=1))
            remaining -= count
        samples = np.concatenate(sample_means)
        spec = METRICS[str(metric)]
        output.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "metric": metric,
                "metric_label": spec["label"],
                "direction": "lower" if spec["lower_is_better"] else "higher",
                "subjects": int(subject_values.size),
                "replicates": int(replicates),
                "delta_favoring_cocycle_mean": float(subject_values.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "probability_cocycle_better": float(np.mean(samples > 0.0)),
            }
        )
    return pd.DataFrame(output)


def volume_trend_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = frame[frame["source_visit_order"] == 0].copy()
    output: list[dict[str, Any]] = []
    for (split, diagnosis, subject_id, model, model_label), group in baseline.groupby(
        ["split", "diagnosis", "subject_id", "model", "model_label"], sort=True
    ):
        group = group.sort_values(["target_age_years", "target_visit_order"]).drop_duplicates("target_visit_order")
        if group.empty:
            continue
        source = group.iloc[0]
        times = np.concatenate([[0.0], group["target_age_years"].to_numpy(float) - float(source["source_age_years"])])
        observed_volumes = np.concatenate([[float(source["source_volume"])], group["target_volume"].to_numpy(float)])
        predicted_volumes = np.concatenate([[float(source["source_volume"])], group["predicted_volume"].to_numpy(float)])
        if np.unique(times).size < 2:
            continue
        observed_slope = float(np.polyfit(times, np.log(np.maximum(observed_volumes, 1e-8)), 1)[0])
        predicted_slope = float(np.polyfit(times, np.log(np.maximum(predicted_volumes, 1e-8)), 1)[0])
        last_time = float(times[-1])
        observed_rate = float((math.log(max(observed_volumes[-1], 1e-8)) - math.log(max(observed_volumes[0], 1e-8))) / last_time)
        predicted_rate = float((math.log(max(predicted_volumes[-1], 1e-8)) - math.log(max(predicted_volumes[0], 1e-8))) / last_time)
        output.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "subject_id": subject_id,
                "model": model,
                "model_label": model_label,
                "future_visits": int(len(group)),
                "maximum_followup_years": last_time,
                "observed_log_volume_slope_per_year": observed_slope,
                "predicted_log_volume_slope_per_year": predicted_slope,
                "slope_error": predicted_slope - observed_slope,
                "slope_absolute_error": abs(predicted_slope - observed_slope),
                "observed_first_last_log_rate": observed_rate,
                "predicted_first_last_log_rate": predicted_rate,
                "first_last_rate_error": predicted_rate - observed_rate,
                "first_last_rate_absolute_error": abs(predicted_rate - observed_rate),
            }
        )
    trends = pd.DataFrame(output)
    expanded = add_all_split(trends)
    summaries: list[dict[str, Any]] = []
    for (split, diagnosis, model, model_label), group in expanded.groupby(
        ["split", "diagnosis", "model", "model_label"], sort=True
    ):
        summaries.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "model": model,
                "model_label": model_label,
                "subjects": int(len(group)),
                "observed_slope_mean": float(group["observed_log_volume_slope_per_year"].mean()),
                "predicted_slope_mean": float(group["predicted_log_volume_slope_per_year"].mean()),
                "slope_absolute_error_mean": float(group["slope_absolute_error"].mean()),
                "slope_absolute_error_median": float(group["slope_absolute_error"].median()),
                "slope_pearson": pearson(
                    group["predicted_log_volume_slope_per_year"].to_numpy(float),
                    group["observed_log_volume_slope_per_year"].to_numpy(float),
                ),
                "observed_first_last_rate_mean": float(group["observed_first_last_log_rate"].mean()),
                "predicted_first_last_rate_mean": float(group["predicted_first_last_log_rate"].mean()),
                "first_last_rate_absolute_error_mean": float(group["first_last_rate_absolute_error"].mean()),
            }
        )
    return trends, pd.DataFrame(summaries)


def select_examples(frame: pd.DataFrame, split: str) -> dict[str, dict[str, Any]]:
    baseline = frame[(frame["split"] == split) & (frame["source_visit_order"] == 0)].copy()
    baseline = baseline.drop_duplicates(["diagnosis", "subject_id", "source_visit_order", "target_visit_order"])
    output: dict[str, dict[str, Any]] = {}
    for diagnosis in ("CN", "AD"):
        candidates = baseline[baseline["diagnosis"] == diagnosis]
        score = (
            candidates.groupby("subject_id", as_index=False)
            .agg(future_visits=("target_visit_order", "nunique"), maximum_followup_years=("gap_years", "max"))
            .sort_values(["future_visits", "maximum_followup_years", "subject_id"], ascending=[False, False, True])
        )
        if score.empty:
            raise ValueError(f"No baseline-to-future subject for {split} {diagnosis}")
        row = score.iloc[0]
        output[diagnosis] = {
            "split": split,
            "diagnosis": diagnosis,
            "subject_id": canonical_subject_id(row["subject_id"]),
            "future_visits": int(row["future_visits"]),
            "maximum_followup_years": float(row["maximum_followup_years"]),
            "selection_rule": "most future visits, then longest follow-up, then subject ID",
        }
    return output


def selected_volume_trajectories(frame: pd.DataFrame, selected: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diagnosis, info in selected.items():
        group = frame[
            (frame["split"] == info["split"])
            & (frame["diagnosis"] == diagnosis)
            & (frame["subject_id"] == info["subject_id"])
            & (frame["source_visit_order"] == 0)
        ].copy()
        if group.empty:
            raise ValueError(f"No cached rows for selected {diagnosis} subject {info['subject_id']}")
        source = group.iloc[0]
        rows.append(
            {
                "split": info["split"],
                "diagnosis": diagnosis,
                "subject_id": info["subject_id"],
                "age_years": float(source["source_age_years"]),
                "visit_order": int(source["source_visit_order"]),
                "series": "GT source",
                "model": "GT",
                "volume": float(source["source_volume"]),
            }
        )
        gt = group.sort_values("target_age_years").drop_duplicates("target_visit_order")
        for item in gt.itertuples(index=False):
            rows.append(
                {
                    "split": info["split"],
                    "diagnosis": diagnosis,
                    "subject_id": info["subject_id"],
                    "age_years": float(item.target_age_years),
                    "visit_order": int(item.target_visit_order),
                    "series": "GT future",
                    "model": "GT",
                    "volume": float(item.target_volume),
                }
            )
        for model, label in MODEL_LABELS.items():
            predicted = group[group["model"] == model].sort_values("target_age_years")
            for item in predicted.itertuples(index=False):
                rows.append(
                    {
                        "split": info["split"],
                        "diagnosis": diagnosis,
                        "subject_id": info["subject_id"],
                        "age_years": float(item.target_age_years),
                        "visit_order": int(item.target_visit_order),
                        "series": label,
                        "model": model,
                        "volume": float(item.predicted_volume),
                    }
                )
    return pd.DataFrame(rows)


class InferenceContext:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        metadata_dir: Path,
        checkpoint: Path,
        device: torch.device,
        splits: Sequence[str],
    ) -> None:
        payload = torch.load(checkpoint, map_location=device)
        resolved_metadata = metadata_dir
        if not resolved_metadata.is_absolute():
            resolved_metadata = resolve_repo_path(resolved_metadata)
        with np.load(resolved_metadata / "speed_feature_stats.npz", allow_pickle=False) as archive:
            metadata = {key: archive[key] for key in archive.files}
        self.v4 = make_model(config=payload.get("config", config), metadata=metadata, device=device)
        self.v4.load_state_dict(payload["model_state_dict"])
        self.v4.eval()
        self.brainode, self.brainode_substeps = load_brainode(resolve_repo_path(config["brainode_checkpoint"]), device)
        _, self.mean_flat, self.components, self.faces = load_pca_model(int(config["components"]))
        registered = np.load(resolve_repo_path(config["registered_mesh_tensors"]), allow_pickle=False)
        template = decode_pca_np(np.zeros((1, self.components.shape[0]), dtype=np.float32), self.mean_flat, self.components)[0]
        self.normals = (
            registered["template_vertex_normals"].astype(np.float32)
            if "template_vertex_normals" in registered.files
            else vertex_normals_np(template, self.faces)
        )
        self.template = template
        self.device = device
        self.archives: dict[str, dict[str, np.ndarray]] = {}
        self.latents: dict[str, np.ndarray] = {}
        self.records: dict[str, list[PairRecord]] = {}
        for split in splits:
            archive = load_split_archive(split)
            self.archives[split] = archive
            self.latents[split] = pca_latents(archive, int(config["components"]))
            self.records[split] = build_pair_records(archive)

    @torch.no_grad()
    def predict(self, split: str, records: Sequence[PairRecord], conditions: Sequence[int] | None = None) -> dict[str, np.ndarray]:
        if not records:
            empty = np.empty((0, self.components.shape[0]), dtype=np.float32)
            return {"source": empty, "target": empty, "brainode": empty, "v4": empty, "speed": np.empty(0, dtype=np.float32)}
        latent = self.latents[split]
        source = np.stack([latent[record.source_index] for record in records]).astype(np.float32)
        target = np.stack([latent[record.target_index] for record in records]).astype(np.float32)
        condition_values = np.asarray(
            [record.label_ad for record in records] if conditions is None else list(conditions), dtype=np.float32
        )
        source_t = torch.from_numpy(source).to(self.device)
        source_norm = torch.tensor([record.source_age_norm for record in records], dtype=torch.float32, device=self.device)
        target_norm = torch.tensor([record.target_age_norm for record in records], dtype=torch.float32, device=self.device)
        source_years = torch.tensor([record.source_age_years for record in records], dtype=torch.float32, device=self.device)
        target_years = torch.tensor([record.target_age_years for record in records], dtype=torch.float32, device=self.device)
        condition_t = torch.from_numpy(condition_values).to(self.device)
        v4, diagnostics = self.v4.transport(
            source_t, source_norm, target_norm, source_years, target_years, condition_t
        )
        from brainode_model import integrate_sequence_rk4

        times = torch.stack([source_norm, target_norm], dim=1)
        brainode = integrate_sequence_rk4(
            self.brainode,
            source_t,
            times,
            condition_t,
            substeps=self.brainode_substeps,
        )[:, -1]
        return {
            "source": source,
            "target": target,
            "brainode": brainode.cpu().numpy().astype(np.float32),
            "v4": v4.cpu().numpy().astype(np.float32),
            "speed": diagnostics["speed"].cpu().numpy().astype(np.float32),
        }


def matching_baseline_records(context: InferenceContext, split: str, subject_id: str) -> list[PairRecord]:
    key = canonical_subject_id(subject_id)
    return sorted(
        [
            record
            for record in context.records[split]
            if canonical_subject_id(record.subject_id) == key
            and record.direction == "forward"
            and int(record.source_visit_order) == 0
        ],
        key=lambda record: (record.target_visit_order, record.target_age_years),
    )


def batched_predictions(
    context: InferenceContext,
    split: str,
    records: Sequence[PairRecord],
    batch_size: int,
    condition: int | None = None,
) -> Iterable[tuple[Sequence[PairRecord], dict[str, np.ndarray]]]:
    for start in range(0, len(records), int(batch_size)):
        batch = records[start : start + int(batch_size)]
        conditions = None if condition is None else [int(condition)] * len(batch)
        yield batch, context.predict(split, batch, conditions)


def normal_change(source: np.ndarray, target: np.ndarray, delta_years: np.ndarray, normals: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_years, dtype=np.float32).reshape(-1, 1, 1)
    delta = np.where(np.abs(delta) < 1e-6, np.sign(delta) * 1e-6 + (delta == 0) * 1e-6, delta)
    return np.sum(((target - source) / delta) * normals[None, :, :], axis=2)


def hotspot_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(round(values.size * float(fraction))))
    mask = np.zeros(values.size, dtype=np.int8)
    mask[np.argsort(np.abs(values))[-count:]] = 1
    return mask


def build_counterfactual_table(
    context: InferenceContext,
    selected: dict[str, dict[str, Any]],
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diagnosis, info in selected.items():
        split = str(info["split"])
        subject_id = str(info["subject_id"])
        records = matching_baseline_records(context, split, subject_id)
        if not records:
            raise ValueError(f"No records for selected {split} {diagnosis} subject {subject_id}")
        source = context.latents[split][records[0].source_index]
        source_volume = float(mesh_volume_np(decode_pca_np(source[None], context.mean_flat, context.components)[0], context.faces)[0])
        rows.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "subject_id": subject_id,
                "age_years": records[0].source_age_years,
                "visit_order": records[0].source_visit_order,
                "condition": "GT",
                "model": "GT",
                "series": "GT source",
                "volume": source_volume,
                "v4_speed": np.nan,
            }
        )
        targets = context.latents[split][[record.target_index for record in records]]
        target_vertices = decode_pca_np(targets, context.mean_flat, context.components)
        target_volumes = mesh_volume_np(target_vertices, context.faces)
        for record, volume in zip(records, target_volumes):
            rows.append(
                {
                    "split": split,
                    "diagnosis": diagnosis,
                    "subject_id": subject_id,
                    "age_years": record.target_age_years,
                    "visit_order": record.target_visit_order,
                    "condition": "GT",
                    "model": "GT",
                    "series": "GT future",
                    "volume": float(volume),
                    "v4_speed": np.nan,
                }
            )
        conditions = (0, 1) if diagnosis == "CN" else (1,)
        for condition in conditions:
            condition_label = "AD" if condition else "CN"
            for batch_records, prediction in batched_predictions(context, split, records, batch_size, condition):
                for model in ("brainode", "v4"):
                    vertices = decode_pca_np(prediction[model], context.mean_flat, context.components)
                    volumes = mesh_volume_np(vertices, context.faces)
                    for index, (record, volume) in enumerate(zip(batch_records, volumes)):
                        rows.append(
                            {
                                "split": split,
                                "diagnosis": diagnosis,
                                "subject_id": subject_id,
                                "age_years": record.target_age_years,
                                "visit_order": record.target_visit_order,
                                "condition": condition_label,
                                "model": model,
                                "series": f"{MODEL_LABELS[model]} as {condition_label}",
                                "volume": float(volume),
                                "v4_speed": float(prediction["speed"][index]) if model == "v4" else np.nan,
                            }
                        )
    return pd.DataFrame(rows)


def selected_mesh_outputs(
    context: InferenceContext,
    selected: dict[str, dict[str, Any]],
    top_fraction: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    vertex_tables: dict[str, pd.DataFrame] = {}
    metrics: list[dict[str, Any]] = []
    for diagnosis, info in selected.items():
        split = str(info["split"])
        subject_id = str(info["subject_id"])
        records = matching_baseline_records(context, split, subject_id)
        record = max(records, key=lambda value: (value.abs_gap_years, value.target_visit_order))
        prediction = context.predict(split, [record])
        latent_vertices = {
            key: decode_pca_np(prediction[key], context.mean_flat, context.components)[0]
            for key in ("source", "target", "brainode", "v4")
        }
        source = latent_vertices["source"]
        target = latent_vertices["target"]
        delta = np.asarray([record.delta_years], dtype=np.float32)
        changes = {
            key: normal_change(source[None], latent_vertices[key][None], delta, context.normals)[0]
            for key in ("target", "brainode", "v4")
        }
        errors = {key: np.linalg.norm(latent_vertices[key] - target, axis=1) for key in ("brainode", "v4")}
        masks = {key: hotspot_mask(value, top_fraction) for key, value in changes.items()}
        table = pd.DataFrame(
            {
                "vertex_id": np.arange(source.shape[0], dtype=np.int64),
                "source_x": source[:, 0],
                "source_y": source[:, 1],
                "source_z": source[:, 2],
                "gt_x": target[:, 0],
                "gt_y": target[:, 1],
                "gt_z": target[:, 2],
                "brainode_x": latent_vertices["brainode"][:, 0],
                "brainode_y": latent_vertices["brainode"][:, 1],
                "brainode_z": latent_vertices["brainode"][:, 2],
                "cocycle_x": latent_vertices["v4"][:, 0],
                "cocycle_y": latent_vertices["v4"][:, 1],
                "cocycle_z": latent_vertices["v4"][:, 2],
                "brainode_endpoint_error": errors["brainode"],
                "cocycle_endpoint_error": errors["v4"],
                "gt_normal_change_rate": changes["target"],
                "brainode_normal_change_rate": changes["brainode"],
                "cocycle_normal_change_rate": changes["v4"],
                "gt_hotspot": masks["target"],
                "brainode_hotspot": masks["brainode"],
                "cocycle_hotspot": masks["v4"],
            }
        )
        for key, model in (("brainode", "brainode"), ("v4", "v4")):
            metrics.append(
                {
                    "split": split,
                    "diagnosis": diagnosis,
                    "subject_id": subject_id,
                    "source_visit_order": record.source_visit_order,
                    "target_visit_order": record.target_visit_order,
                    "gap_years": record.abs_gap_years,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "vertex_error_mean": float(errors[key].mean()),
                    "normal_change_mae": float(np.abs(changes[key] - changes["target"]).mean()),
                    "normal_change_pearson": pearson(changes[key], changes["target"]),
                    "hotspot_dice": top_change_dice(changes[key], changes["target"], top_fraction),
                    "v4_speed": float(prediction["speed"][0]) if model == "v4" else np.nan,
                }
            )
        vertex_tables[diagnosis] = table
    return vertex_tables, pd.DataFrame(metrics)


def evenly_limit(items: Sequence[Any], maximum: int) -> list[Any]:
    if maximum <= 0 or len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, int(maximum)).round().astype(int)
    return [items[int(index)] for index in indices]


def group_change_outputs(
    context: InferenceContext,
    splits: Sequence[str],
    batch_size: int,
    maximum_per_diagnosis: int,
    top_fraction: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    for diagnosis in ("CN", "AD"):
        items: list[tuple[str, PairRecord]] = []
        for split in splits:
            items.extend(
                (split, record)
                for record in context.records[split]
                if record.direction == "forward" and record.diagnosis == diagnosis
            )
        items = evenly_limit(items, maximum_per_diagnosis)
        per_subject: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "gt": None, "brainode": None, "v4": None}
        )
        for split in splits:
            split_items = [record for current_split, record in items if current_split == split]
            for batch_records, prediction in batched_predictions(context, split, split_items, batch_size):
                decoded = {
                    key: decode_pca_np(prediction[key], context.mean_flat, context.components)
                    for key in ("source", "target", "brainode", "v4")
                }
                delta = np.asarray([record.delta_years for record in batch_records], dtype=np.float32)
                maps = {
                    "gt": normal_change(decoded["source"], decoded["target"], delta, context.normals),
                    "brainode": normal_change(decoded["source"], decoded["brainode"], delta, context.normals),
                    "v4": normal_change(decoded["source"], decoded["v4"], delta, context.normals),
                }
                for index, record in enumerate(batch_records):
                    subject_key = f"{split}:{canonical_subject_id(record.subject_id)}"
                    accumulator = per_subject[subject_key]
                    accumulator["count"] += 1
                    for key in ("gt", "brainode", "v4"):
                        accumulator[key] = maps[key][index].astype(np.float64) if accumulator[key] is None else accumulator[key] + maps[key][index]
        if not per_subject:
            raise ValueError(f"No group-map pairs for {diagnosis}")
        subject_maps: dict[str, list[np.ndarray]] = {"gt": [], "brainode": [], "v4": []}
        for accumulator in per_subject.values():
            for key in subject_maps:
                subject_maps[key].append(accumulator[key] / float(accumulator["count"]))
        mean_maps = {key: np.mean(np.stack(value), axis=0) for key, value in subject_maps.items()}
        masks = {key: hotspot_mask(value, top_fraction) for key, value in mean_maps.items()}
        tables[diagnosis] = pd.DataFrame(
            {
                "vertex_id": np.arange(context.template.shape[0], dtype=np.int64),
                "template_x": context.template[:, 0],
                "template_y": context.template[:, 1],
                "template_z": context.template[:, 2],
                "gt_normal_change_rate": mean_maps["gt"],
                "brainode_normal_change_rate": mean_maps["brainode"],
                "cocycle_normal_change_rate": mean_maps["v4"],
                "gt_hotspot": masks["gt"],
                "brainode_hotspot": masks["brainode"],
                "cocycle_hotspot": masks["v4"],
            }
        )
        for model in ("brainode", "v4"):
            metric_rows.append(
                {
                    "diagnosis": diagnosis,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "pairs": int(len(items)),
                    "subjects": int(len(per_subject)),
                    "aggregation": "mean within subject, then mean across subjects",
                    "normal_change_mae": float(np.abs(mean_maps[model] - mean_maps["gt"]).mean()),
                    "normal_change_pearson": pearson(mean_maps[model], mean_maps["gt"]),
                    "hotspot_dice": top_change_dice(mean_maps[model], mean_maps["gt"], top_fraction),
                    "top_fraction": float(top_fraction),
                }
            )
    return tables, pd.DataFrame(metric_rows)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.group_map_max_pairs_per_diagnosis < 0:
        raise ValueError("--group-map-max-pairs-per-diagnosis must be nonnegative")
    config = load_config(args.config)
    checkpoint = v4_checkpoint(args.run_name, args.checkpoint)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else EXPERIMENT_DIR / "metadata"
    if not metadata_dir.is_absolute():
        metadata_dir = resolve_repo_path(metadata_dir)
    unified_cache_dir = (
        Path(args.unified_cache_dir)
        if args.unified_cache_dir
        else EXPERIMENT_DIR / "analysis" / "notebook_brainode_vs_cocycle_all_data" / "unified_baselines_all_data"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else EXPERIMENT_DIR / "analysis" / "notebook_brainode_vs_cocycle_all_data" / "prepared_cache_v1"
    )
    if not unified_cache_dir.is_absolute():
        unified_cache_dir = resolve_repo_path(unified_cache_dir)
    if not output_dir.is_absolute():
        output_dir = resolve_repo_path(output_dir)
    bootstrap_replicates = int(
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else config.get("evaluation", {}).get("bootstrap_replicates", 10000)
    )
    if bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "config": file_signature(Path(config["_config_path"])),
        "v4_checkpoint": file_signature(checkpoint),
        "brainode_checkpoint": file_signature(resolve_repo_path(config["brainode_checkpoint"])),
        "metadata": file_signature(metadata_dir / "speed_feature_stats.npz"),
        "unified_per_pair": file_signature(unified_cache_dir / "unified_registered_per_pair.csv"),
        "unified_run": file_signature(unified_cache_dir / "run.json"),
        "splits": list(args.splits),
        "example_split": args.example_split,
        "seed": int(args.seed),
        "bootstrap_replicates": bootstrap_replicates,
        "group_map_max_pairs_per_diagnosis": int(args.group_map_max_pairs_per_diagnosis),
        "skip_group_maps": bool(args.skip_group_maps),
    }
    if not args.force and manifest_is_complete(output_dir, provenance):
        print(f"Complete matching cache already exists: {output_dir}")
        return 0
    old_provenance_path = output_dir / "provenance.json"
    if output_dir.exists() and old_provenance_path.is_file() and not args.force:
        old_provenance = load_json(old_provenance_path)
        if old_provenance != provenance:
            raise RuntimeError(
                f"Existing cache provenance differs: {output_dir}\n"
                "Use a new --output-dir, or pass --force to rebuild known cache files."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "meshes").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "provenance.json", provenance)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "building",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "output_dir": str(output_dir),
        "stages": {},
        "required_files": [str(path.relative_to(output_dir)) for path in expected_files(output_dir) if path.name != "manifest.json"],
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and not args.force:
        previous = load_json(manifest_path)
        if previous.get("status") == "building":
            manifest["created_at"] = previous.get("created_at", manifest["created_at"])
            manifest["stages"] = previous.get("stages", {})
    write_json(manifest_path, manifest)

    frame = prepare_pair_frame(unified_cache_dir, args.splits)
    paired, metric_summary = paired_metric_tables(frame)
    counts = cohort_counts(frame)
    gaps = gap_summary(frame)
    bootstrap = bootstrap_tables(paired, bootstrap_replicates, int(args.seed))
    trends, trend_summary = volume_trend_tables(frame)
    selected = select_examples(frame, args.example_split)
    selected_trajectories = selected_volume_trajectories(frame, selected)
    write_frame(output_dir / "tables" / "cohort_counts.csv", counts)
    write_frame(output_dir / "tables" / "metric_summary.csv", metric_summary)
    write_frame(output_dir / "tables" / "paired_metric_comparison.csv", paired)
    write_frame(output_dir / "tables" / "paired_bootstrap_ci.csv", bootstrap)
    write_frame(output_dir / "tables" / "gap_summary.csv", gaps)
    write_frame(output_dir / "tables" / "subject_volume_trends.csv", trends)
    write_frame(output_dir / "tables" / "cohort_volume_trend_summary.csv", trend_summary)
    write_frame(output_dir / "tables" / "selected_volume_trajectories.csv", selected_trajectories)
    write_json(output_dir / "selected_examples.json", selected)
    manifest["stages"]["scalar_tables"] = {
        "status": "complete",
        "completed_at": utc_now(),
        "brainode_cocycle_rows": int(len(frame)),
        "paired_rows": int(len(paired)),
    }
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)

    device = resolve_device(args.device)
    context = InferenceContext(
        config=config,
        metadata_dir=metadata_dir,
        checkpoint=checkpoint,
        device=device,
        splits=args.splits,
    )
    counterfactual = build_counterfactual_table(context, selected, int(args.batch_size))
    selected_vertices, selected_metrics = selected_mesh_outputs(
        context,
        selected,
        float(config.get("evaluation", {}).get("top_change_fraction", 0.2)),
    )
    write_frame(output_dir / "tables" / "counterfactual_trajectories.csv", counterfactual)
    write_frame(output_dir / "tables" / "selected_mesh_metrics.csv", selected_metrics)
    face_frame = pd.DataFrame(
        {
            "face_id": np.arange(context.faces.shape[0], dtype=np.int64),
            "vertex_0": context.faces[:, 0],
            "vertex_1": context.faces[:, 1],
            "vertex_2": context.faces[:, 2],
        }
    )
    write_frame(output_dir / "meshes" / "template_faces.csv", face_frame)
    write_frame(output_dir / "meshes" / "selected_cn_vertices.csv", selected_vertices["CN"])
    write_frame(output_dir / "meshes" / "selected_ad_vertices.csv", selected_vertices["AD"])
    manifest["stages"]["selected_examples"] = {
        "status": "complete",
        "completed_at": utc_now(),
        "device": str(device),
    }
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)

    if args.skip_group_maps:
        empty_group_metrics = pd.DataFrame(
            columns=[
                "diagnosis",
                "model",
                "model_label",
                "pairs",
                "subjects",
                "aggregation",
                "normal_change_mae",
                "normal_change_pearson",
                "hotspot_dice",
                "top_fraction",
            ]
        )
        empty_group_vertices = pd.DataFrame(
            columns=[
                "vertex_id",
                "template_x",
                "template_y",
                "template_z",
                "gt_normal_change_rate",
                "brainode_normal_change_rate",
                "cocycle_normal_change_rate",
                "gt_hotspot",
                "brainode_hotspot",
                "cocycle_hotspot",
            ]
        )
        write_frame(output_dir / "tables" / "group_hotspot_metrics.csv", empty_group_metrics)
        write_frame(output_dir / "meshes" / "group_cn_change_maps.csv", empty_group_vertices)
        write_frame(output_dir / "meshes" / "group_ad_change_maps.csv", empty_group_vertices)
        group_details = {"status": "skipped", "completed_at": utc_now()}
    else:
        group_tables, group_metrics = group_change_outputs(
            context,
            args.splits,
            int(args.batch_size),
            int(args.group_map_max_pairs_per_diagnosis),
            float(config.get("evaluation", {}).get("top_change_fraction", 0.2)),
        )
        write_frame(output_dir / "tables" / "group_hotspot_metrics.csv", group_metrics)
        write_frame(output_dir / "meshes" / "group_cn_change_maps.csv", group_tables["CN"])
        write_frame(output_dir / "meshes" / "group_ad_change_maps.csv", group_tables["AD"])
        group_details = {
            "status": "complete",
            "completed_at": utc_now(),
            "maximum_pairs_per_diagnosis": int(args.group_map_max_pairs_per_diagnosis),
        }
    manifest["stages"]["group_change_maps"] = group_details
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    manifest["selected_examples"] = selected
    manifest["rows"] = {
        "brainode_cocycle": int(len(frame)),
        "paired_metrics": int(len(paired)),
        "subject_trends": int(len(trends)),
    }
    write_json(manifest_path, manifest)
    missing = [path for path in expected_files(output_dir) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cache ended with missing files: {[str(path) for path in missing]}")
    print(f"Prepared complete notebook cache: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
