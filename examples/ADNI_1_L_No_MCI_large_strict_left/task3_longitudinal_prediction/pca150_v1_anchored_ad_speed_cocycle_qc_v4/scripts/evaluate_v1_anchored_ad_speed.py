#!/usr/bin/env python3
"""Evaluate v4 with registered-mesh, volume, local and matched baseline metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_v1_anchored_ad_speed import make_model
from v1_speed_utils import (
    EXPERIMENT_DIR,
    SPLITS,
    PairRecord,
    build_pair_records,
    build_volume_trends,
    decode_pca_np,
    finite_mean,
    first_visit_sequence_starts,
    limit_records_stratified,
    load_config,
    load_pca_model,
    load_split_archive,
    pca_latents,
    prediction_metrics,
    read_csv_rows,
    resolve_device,
    resolve_repo_path,
    summarize_prediction_rows,
    subject_end,
    vertex_normals_np,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "v1_anchored_ad_speed_primary.json"))
    parser.add_argument("--run-name", default="v1_anchor_ad_speed_seed42")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--checkpoint", default="best_feasible_volume")
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--transport-methods", nargs="+", choices=["direct", "composed_observed"], default=["direct", "composed_observed"])
    parser.add_argument("--include-backward-eval", action="store_true")
    parser.add_argument("--include-cycle-eval", action="store_true")
    parser.add_argument("--include-sequence-eval", action="store_true")
    parser.add_argument("--max-pairs-per-split", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def run_dir_for(args: argparse.Namespace) -> Path:
    root = Path(args.experiment_dir) if args.experiment_dir else EXPERIMENT_DIR / "runs"
    return root / str(args.run_name)


def checkpoint_for(run_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    candidate = run_dir / "checkpoints" / (value if value.endswith(".pth") else f"{value}.pth")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


@torch.no_grad()
def direct_prediction(model: torch.nn.Module, source: np.ndarray, record: PairRecord, device: torch.device) -> tuple[np.ndarray, float]:
    latent = torch.from_numpy(source[None, :]).float().to(device)
    prediction, diagnostics = model.transport(
        latent,
        torch.tensor([record.source_age_norm], dtype=torch.float32, device=device),
        torch.tensor([record.target_age_norm], dtype=torch.float32, device=device),
        torch.tensor([record.source_age_years], dtype=torch.float32, device=device),
        torch.tensor([record.target_age_years], dtype=torch.float32, device=device),
        torch.tensor([float(record.label_ad)], dtype=torch.float32, device=device),
    )
    return prediction[0].cpu().numpy().astype(np.float32), float(diagnostics["speed"][0].item())


@torch.no_grad()
def composed_prediction(
    model: torch.nn.Module,
    archive: dict[str, np.ndarray],
    latents: np.ndarray,
    record: PairRecord,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    current = torch.from_numpy(latents[record.source_index : record.source_index + 1]).float().to(device)
    step = 1 if record.target_index > record.source_index else -1
    speed_values: list[float] = []
    for index in range(record.source_index, record.target_index, step):
        next_index = index + step
        current, diagnostics = model.transport(
            current,
            torch.tensor([float(archive["visit_continuous_age_norm"][index])], dtype=torch.float32, device=device),
            torch.tensor([float(archive["visit_continuous_age_norm"][next_index])], dtype=torch.float32, device=device),
            torch.tensor([float(archive["visit_continuous_age_years"][index])], dtype=torch.float32, device=device),
            torch.tensor([float(archive["visit_continuous_age_years"][next_index])], dtype=torch.float32, device=device),
            torch.tensor([float(record.label_ad)], dtype=torch.float32, device=device),
        )
        speed_values.append(float(diagnostics["speed"][0].item()))
    return current[0].cpu().numpy().astype(np.float32), finite_mean(speed_values)


def evaluate_split(
    *,
    model: torch.nn.Module,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    checkpoint: str,
    top_fraction: float,
) -> list[dict[str, Any]]:
    archive = load_split_archive(split)
    latents = pca_latents(archive, components.shape[0])
    records = build_pair_records(archive, include_backward=bool(args.include_backward_eval))
    if args.max_pairs_per_split > 0:
        records = limit_records_stratified(records, int(args.max_pairs_per_split))
    rows: list[dict[str, Any]] = []
    for record in records:
        source, target = latents[record.source_index], latents[record.target_index]
        for method in args.transport_methods:
            if method == "direct":
                prediction, speed = direct_prediction(model, source, record, device)
            else:
                prediction, speed = composed_prediction(model, archive, latents, record, device)
            rows.append(
                prediction_metrics(
                    record=record,
                    split=split,
                    transport_method=method,
                    checkpoint=checkpoint,
                    model_name="pca150_v1_anchored_ad_speed_cocycle_qc_v4",
                    family="frozen v1 PCA flow plus AD speed calibration",
                    source=source,
                    target=target,
                    predicted=prediction,
                    mean_flat=mean_flat,
                    components=components,
                    faces=faces,
                    template_normals=normals,
                    top_fraction=top_fraction,
                    ad_speed=speed,
                )
            )
    return rows


@torch.no_grad()
def evaluate_first_visit_trajectories(
    *,
    model: torch.nn.Module,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    checkpoint: str,
    top_fraction: float,
) -> list[dict[str, Any]]:
    """Forecast each multi-visit subject from its first observed shape only.

    ``sequence_direct_from_first`` independently maps the baseline shape to
    every future age. ``sequence_recursive_from_first`` maps its own previous
    prediction onward, so it exposes trajectory drift without leaking observed
    intermediate shapes.
    """
    archive = load_split_archive(split)
    latents = pca_latents(archive, components.shape[0])
    pair_lookup = {
        (record.source_index, record.target_index): record
        for record in build_pair_records(archive)
    }
    trajectory_records: list[PairRecord] = []
    for start in first_visit_sequence_starts(archive):
        end = subject_end(archive, start)
        trajectory_records.extend(pair_lookup[(start, target)] for target in range(start + 1, end))
    if args.max_pairs_per_split > 0:
        trajectory_records = limit_records_stratified(trajectory_records, int(args.max_pairs_per_split))
    selected = {(record.source_index, record.target_index) for record in trajectory_records}
    rows: list[dict[str, Any]] = []
    for start in first_visit_sequence_starts(archive):
        end = subject_end(archive, start)
        source = latents[start]
        recursive = torch.from_numpy(source[None, :]).float().to(device)
        for target_index in range(start + 1, end):
            step_record = pair_lookup[(target_index - 1, target_index)]
            recursive, recursive_diagnostics = model.transport(
                recursive,
                torch.tensor([step_record.source_age_norm], dtype=torch.float32, device=device),
                torch.tensor([step_record.target_age_norm], dtype=torch.float32, device=device),
                torch.tensor([step_record.source_age_years], dtype=torch.float32, device=device),
                torch.tensor([step_record.target_age_years], dtype=torch.float32, device=device),
                torch.tensor([float(step_record.label_ad)], dtype=torch.float32, device=device),
            )
            record = pair_lookup[(start, target_index)]
            if (start, target_index) not in selected:
                continue
            direct, direct_speed = direct_prediction(model, source, record, device)
            recursive_prediction = recursive[0].cpu().numpy().astype(np.float32)
            rows.append(
                prediction_metrics(
                    record=record,
                    split=split,
                    transport_method="sequence_direct_from_first",
                    checkpoint=checkpoint,
                    model_name="pca150_v1_anchored_ad_speed_cocycle_qc_v4",
                    family="frozen v1 PCA flow plus AD speed calibration",
                    source=source,
                    target=latents[target_index],
                    predicted=direct,
                    mean_flat=mean_flat,
                    components=components,
                    faces=faces,
                    template_normals=normals,
                    top_fraction=top_fraction,
                    ad_speed=direct_speed,
                )
            )
            rows.append(
                prediction_metrics(
                    record=record,
                    split=split,
                    transport_method="sequence_recursive_from_first",
                    checkpoint=checkpoint,
                    model_name="pca150_v1_anchored_ad_speed_cocycle_qc_v4",
                    family="frozen v1 PCA flow plus AD speed calibration",
                    source=source,
                    target=latents[target_index],
                    predicted=recursive_prediction,
                    mean_flat=mean_flat,
                    components=components,
                    faces=faces,
                    template_normals=normals,
                    top_fraction=top_fraction,
                    ad_speed=float(recursive_diagnostics["speed"][0].item()),
                )
            )
    return rows


@torch.no_grad()
def cycle_rows(
    model: torch.nn.Module,
    split: str,
    device: torch.device,
    mean_flat: np.ndarray,
    components: np.ndarray,
    maximum: int,
) -> list[dict[str, Any]]:
    archive = load_split_archive(split)
    latents = pca_latents(archive, components.shape[0])
    rows: list[dict[str, Any]] = []
    mean_tensor = torch.from_numpy(mean_flat).to(device)
    component_tensor = torch.from_numpy(components).to(device)
    records = build_pair_records(archive)
    if maximum > 0:
        records = limit_records_stratified(records, maximum)
    for record in records:
        source = torch.from_numpy(latents[record.source_index : record.source_index + 1]).float().to(device)
        forward, _ = model.transport(source, torch.tensor([record.source_age_norm], device=device), torch.tensor([record.target_age_norm], device=device), torch.tensor([record.source_age_years], device=device), torch.tensor([record.target_age_years], device=device), torch.tensor([float(record.label_ad)], device=device))
        cycle, _ = model.transport(forward, torch.tensor([record.target_age_norm], device=device), torch.tensor([record.source_age_norm], device=device), torch.tensor([record.target_age_years], device=device), torch.tensor([record.source_age_years], device=device), torch.tensor([float(record.label_ad)], device=device))
        source_vertices = (source @ component_tensor + mean_tensor).reshape(1, -1, 3)
        cycle_vertices = (cycle @ component_tensor + mean_tensor).reshape(1, -1, 3)
        rows.append({"split": split, "diagnosis": record.diagnosis, "subject_id": record.subject_id, "cycle_pca_mse": float(torch.mean((cycle - source) ** 2).item()), "cycle_vertex_euclidean": float(torch.linalg.norm(cycle_vertices - source_vertices, dim=2).mean().item())})
    return rows


def summarize_cycle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["split"], row["diagnosis"]), []).append(row)
    return [
        {"split": split, "diagnosis": diagnosis, "rows": len(items), "cycle_pca_mse_mean": finite_mean(float(item["cycle_pca_mse"]) for item in items), "cycle_vertex_euclidean_mean": finite_mean(float(item["cycle_vertex_euclidean"]) for item in items)}
        for (split, diagnosis), items in groups.items()
    ]


def compare_to_v1(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = EXPERIMENT_DIR.parent / "pca150_direct_cocycle_flow_qc_v1" / "analysis" / "checkpoint_best_val_endpoint_vertex_mae" / "pca_flow_per_pair.csv"
    v1 = {(row["split"], row["source_scan_id"], row["target_scan_id"], row["transport_method"]): row for row in read_csv_rows(path)}
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        baseline = v1.get((str(row["split"]), str(row["source_scan_id"]), str(row["target_scan_id"]), str(row["transport_method"])))
        if baseline is None:
            continue
        for metric in ("endpoint_vertex_mae", "endpoint_pca_mse", "volume_relative_error"):
            if metric not in baseline:
                continue
            value, base = float(row[metric]), float(baseline[metric])
            comparisons.append({"split": row["split"], "diagnosis": row["diagnosis"], "transport_method": row["transport_method"], "subject_id": row["subject_id"], "metric": metric, "v4": value, "v1": base, "improvement": base - value})
    return comparisons, summarize_comparison(comparisons, "v4", "v1")


def compare_to_brainode(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = EXPERIMENT_DIR.parent / "brainode_pca150_qc_stable" / "training" / "core_attention_pca150_qc_stable" / "evaluation" / "best" / "all_trajectory_metrics.csv"
    lookup: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in read_csv_rows(path):
        if row.get("record_set") != "all_pairs" or row.get("direction") != "forward":
            continue
        start = int(float(row["start_visit_order"]))
        target = start + int(float(row["length"])) - 1
        lookup[(row["split"], row["subject_id"], start, target)] = row
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["transport_method"] != "direct" or row["direction"] != "forward":
            continue
        baseline = lookup.get((str(row["split"]), str(row["subject_id"]), int(row["source_visit_order"]), int(row["target_visit_order"])))
        if baseline is None:
            continue
        for metric in ("endpoint_vertex_mae", "endpoint_pca_mse"):
            value, base = float(row[metric]), float(baseline[metric])
            comparisons.append({"split": row["split"], "diagnosis": row["diagnosis"], "transport_method": row["transport_method"], "subject_id": row["subject_id"], "metric": metric, "v4": value, "brainode": base, "improvement": base - value})
    return comparisons, summarize_comparison(comparisons, "v4", "brainode")


def summarize_comparison(rows: list[dict[str, Any]], current_key: str, base_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["split"], row["diagnosis"], row["transport_method"], row["metric"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (split, diagnosis, method, metric), items in groups.items():
        improvements = [float(item["improvement"]) for item in items]
        output.append({"split": split, "diagnosis": diagnosis, "transport_method": method, "metric": metric, "rows": len(items), f"{current_key}_mean": finite_mean(float(item[current_key]) for item in items), f"{base_key}_mean": finite_mean(float(item[base_key]) for item in items), "improvement_mean": finite_mean(improvements), "beats_baseline_fraction": finite_mean(1.0 if value > 0 else 0.0 for value in improvements)})
    return output


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    run_dir = run_dir_for(args)
    checkpoint = checkpoint_for(run_dir, args.checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else Path(payload["metadata_dir"])
    with np.load(metadata_dir / "speed_feature_stats.npz", allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    model = make_model(config=payload.get("config", config), metadata=metadata, device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    _, mean_flat, components, faces = load_pca_model(int(config["components"]))
    registered = np.load(resolve_repo_path(config["registered_mesh_tensors"]), allow_pickle=False)
    normals = registered["template_vertex_normals"].astype(np.float32) if "template_vertex_normals" in registered.files else vertex_normals_np(decode_pca_np(np.zeros((1, components.shape[0]), dtype=np.float32), mean_flat, components)[0], faces)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis" / f"checkpoint_{checkpoint.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in args.splits:
        current = evaluate_split(model=model, split=split, args=args, device=device, mean_flat=mean_flat, components=components, faces=faces, normals=normals, checkpoint=checkpoint.stem, top_fraction=float(config["evaluation"]["top_change_fraction"]))
        rows.extend(current)
    summary = summarize_prediction_rows(rows)
    trends, trend_summary = build_volume_trends(rows)
    v1_rows, v1_summary = compare_to_v1(rows)
    brainode_rows, brainode_summary = compare_to_brainode(rows)
    write_csv(output_dir / "forward_per_pair.csv", [row for row in rows if row["direction"] == "forward" and row["transport_method"] == "direct"])
    write_csv(output_dir / "registered_flow_per_pair.csv", rows)
    write_csv(output_dir / "registered_flow_summary.csv", summary)
    write_csv(output_dir / "volume_trends.csv", trends)
    write_csv(output_dir / "volume_slope_summary.csv", trend_summary)
    write_csv(output_dir / "v1_comparison_matched.csv", v1_rows)
    write_csv(output_dir / "v1_comparison_summary.csv", v1_summary)
    write_csv(output_dir / "brainode_comparison_matched.csv", brainode_rows)
    write_csv(output_dir / "brainode_comparison_summary.csv", brainode_summary)
    sequence_rows: list[dict[str, Any]] = []
    if args.include_sequence_eval:
        for split in args.splits:
            sequence_rows.extend(
                evaluate_first_visit_trajectories(
                    model=model,
                    split=split,
                    args=args,
                    device=device,
                    mean_flat=mean_flat,
                    components=components,
                    faces=faces,
                    normals=normals,
                    checkpoint=checkpoint.stem,
                    top_fraction=float(config["evaluation"]["top_change_fraction"]),
                )
            )
        sequence_trends: list[dict[str, Any]] = []
        sequence_trend_summary: list[dict[str, Any]] = []
        for method in ("sequence_direct_from_first", "sequence_recursive_from_first"):
            current_trends, current_summary = build_volume_trends(sequence_rows, transport_method=method)
            sequence_trends.extend(current_trends)
            sequence_trend_summary.extend(current_summary)
        write_csv(output_dir / "sequence_trajectory_per_pair.csv", sequence_rows)
        write_csv(output_dir / "sequence_trajectory_summary.csv", summarize_prediction_rows(sequence_rows))
        write_csv(output_dir / "sequence_volume_trends.csv", sequence_trends)
        write_csv(output_dir / "sequence_volume_slope_summary.csv", sequence_trend_summary)
    if args.include_cycle_eval:
        cycle: list[dict[str, Any]] = []
        for split in args.splits:
            cycle.extend(cycle_rows(model, split, device, mean_flat, components, int(args.max_pairs_per_split)))
        write_csv(output_dir / "cycle_consistency.csv", cycle)
        write_csv(output_dir / "cycle_consistency_summary.csv", summarize_cycle(cycle))
    write_json(output_dir / "run.json", {"run_dir": str(run_dir), "checkpoint": str(checkpoint), "device": str(device), "rows": len(rows), "sequence_rows": len(sequence_rows), "splits": args.splits, "transport_methods": args.transport_methods, "checkpoint_metrics": payload.get("metrics", {})})
    print(f"Wrote evaluation to {output_dir}; rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
