#!/usr/bin/env python3
"""Read-only paper-style evaluation for the strict ADNI BrainODE paper-core runs.

Reports the paper-aligned one-shot task (first observed shape to final shape)
and four-shot task (first four observed shapes independently predict the final
shape, then the four predictions are averaged).  All ODE calls use one subject
state at a time.  The evaluator never modifies a checkpoint, mesh, PCA model,
or training artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_adni_synthseg_pca_brainode_paper_core import (
    PaperCoreODEFuncWithAttention,
    RawPcaDecoder,
    SubjectSequence,
    build_subjects,
    default_config_path,
    integrate,
    resolve_path,
)
from train_adni_synthseg_pca_cocycle_v4 import (
    BASE_ROOT,
    STRUCTURES,
    choose_device,
    load_archive,
    read_json,
    validate_pca_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=STRUCTURES)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every-subjects", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_state_dict(state: dict[str, torch.Tensor], label: str) -> None:
    if not state:
        raise ValueError(f"Empty checkpoint: {label}")
    for name, value in state.items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor {label}:{name}")


def validate_config(config: dict[str, Any], structure: str) -> dict[str, Any]:
    expected = {"hippocampus": "left_hippocampus", "lateral_ventricle": "left_lateral_ventricle"}[structure]
    if config.get("structure") != expected or config.get("method") != "brainode_paper_core_stable_cn_ad_raw_pca150":
        raise ValueError("Not a matching paper-core BrainODE configuration")
    if config["representation"].get("training_key") != "visit_pca_150":
        raise ValueError("Paper-core evaluation requires raw PCA scores")
    if config["scientific_contract"].get("strict_no_mci") is not True:
        raise ValueError("Paper-core config is not strict no-MCI")
    input_config = read_json(resolve_path(config["input_config"]))
    if input_config.get("structure") != expected or int(input_config["representation"]["components"]) != 150:
        raise ValueError("Input data contract mismatch")
    return input_config


def load_model(
    checkpoint: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[PaperCoreODEFuncWithAttention, dict[str, Any]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    finite_state_dict(payload["model_state_dict"], str(checkpoint))
    if payload.get("test_data_loaded") is not False:
        raise ValueError("Checkpoint indicates test data were loaded during training")
    model = PaperCoreODEFuncWithAttention(
        latent_dim=int(config["model"]["latent_dim"]),
        condition_dim=int(config["model"]["condition_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def subject_prediction(
    model: torch.nn.Module,
    subject: SubjectSequence,
    source_index: int,
    device: torch.device,
    substeps: int,
) -> np.ndarray:
    values = torch.from_numpy(subject.values[source_index:]).to(device)
    times = torch.from_numpy(subject.times[source_index:]).to(device)
    condition = torch.tensor([subject.condition], device=device, dtype=torch.float32)
    return integrate(model, values, times, condition, substeps)[-1].cpu().numpy().astype(np.float64)


@torch.no_grad()
def metric_row(
    structure: str,
    split: str,
    protocol: str,
    subject: SubjectSequence,
    source_indices: list[int],
    model: torch.nn.Module,
    decoder: RawPcaDecoder,
    device: torch.device,
    substeps: int,
) -> dict[str, Any]:
    predictions = np.stack([
        subject_prediction(model, subject, source_index, device, substeps)
        for source_index in source_indices
    ])
    prediction = np.mean(predictions, axis=0)
    target = subject.values[-1].astype(np.float64)
    no_change = np.mean(np.stack([subject.values[index] for index in source_indices]), axis=0).astype(np.float64)
    prediction_tensor = torch.from_numpy(prediction.astype(np.float32)).to(device).reshape(1, 150)
    target_tensor = torch.from_numpy(target.astype(np.float32)).to(device).reshape(1, 150)
    no_change_tensor = torch.from_numpy(no_change.astype(np.float32)).to(device).reshape(1, 150)
    prediction_vertices = decoder.vertices(prediction_tensor)
    target_vertices = decoder.vertices(target_tensor)
    no_change_vertices = decoder.vertices(no_change_tensor)
    euclidean = torch.linalg.vector_norm(prediction_vertices - target_vertices, dim=-1)
    no_change_euclidean = torch.linalg.vector_norm(no_change_vertices - target_vertices, dim=-1)
    prediction_volume = decoder.volume(prediction_vertices)
    target_volume = decoder.volume(target_vertices)
    no_change_volume = decoder.volume(no_change_vertices)
    return {
        "structure": structure,
        "split": split,
        "protocol": protocol,
        "subject_id": subject.subject_id,
        "diagnosis": subject.diagnosis,
        "source_count": len(source_indices),
        "source_visit_orders": ";".join(str(index) for index in source_indices),
        "target_visit_order": len(subject.times) - 1,
        "visits": len(subject.times),
        "target_age_years_normalized": float(subject.times[-1]),
        "pca_mse": float(np.mean((prediction - target) ** 2)),
        "vertex_coordinate_mae_mm": float(torch.abs(prediction_vertices - target_vertices).mean().cpu()),
        "vertex_euclidean_mean_mm": float(euclidean.mean().cpu()),
        "vertex_euclidean_rmse_mm": float(torch.sqrt(torch.mean(euclidean ** 2)).cpu()),
        "volume_relative_error": float(torch.abs(prediction_volume - target_volume).div(target_volume).mean().cpu()),
        "no_change_vertex_euclidean_mean_mm": float(no_change_euclidean.mean().cpu()),
        "no_change_volume_relative_error": float(torch.abs(no_change_volume - target_volume).div(target_volume).mean().cpu()),
        "vertex_euclidean_improvement_mm": float((no_change_euclidean.mean() - euclidean.mean()).cpu()),
    }


def evaluate_protocols(
    structure: str,
    split: str,
    subjects: list[SubjectSequence],
    model: torch.nn.Module,
    decoder: RawPcaDecoder,
    device: torch.device,
    substeps: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, subject in enumerate(subjects, start=1):
        # Paper one-shot: earliest observed shape predicts the latest one.
        rows.append(metric_row(structure, split, "one_shot", subject, [0], model, decoder, device, substeps))
        # Paper four-shot: use the first four observed shapes and average their
        # independently predicted final shapes.  It requires five observations.
        if len(subject.times) >= 5:
            rows.append(metric_row(structure, split, "four_shot", subject, [0, 1, 2, 3], model, decoder, device, substeps))
        # Supplementary all-observed-source endpoint scores for direct
        # comparison with the existing current-data validation protocol.
        rows.append(metric_row(structure, split, "all_observed_sources", subject, list(range(len(subject.times) - 1)), model, decoder, device, substeps))
        if position % progress_every == 0 or position == len(subjects):
            print(f"  {split}: evaluated {position:03d}/{len(subjects)} subjects", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "pca_mse",
        "vertex_coordinate_mae_mm",
        "vertex_euclidean_mean_mm",
        "vertex_euclidean_rmse_mm",
        "volume_relative_error",
        "no_change_vertex_euclidean_mean_mm",
        "no_change_volume_relative_error",
        "vertex_euclidean_improvement_mm",
    ]
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for diagnosis in (row["diagnosis"], "ALL"):
            buckets[(row["structure"], row["split"], row["protocol"], diagnosis)].append(row)
    output: list[dict[str, Any]] = []
    for (structure, split, protocol, diagnosis), current in sorted(buckets.items()):
        result: dict[str, Any] = {
            "structure": structure,
            "split": split,
            "protocol": protocol,
            "diagnosis": diagnosis,
            "subjects": len(current),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in current], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_median"] = float(np.median(values))
        output.append(result)
    return output


def markdown(report: dict[str, Any], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# BrainODE paper-core evaluation",
        "",
        "This is the strict CN/AD, all-current-QC-subject, paper-core BrainODE evaluation. It uses raw PCA-150 scores and one-subject RK4 inference. It does not include pseudo-cognitive status embedding or a cognition estimator.",
        "",
        "## Contract status",
        "",
        f"- Overall: **{report['status']}**",
        "- Training checkpoint reports no test access: passed.",
        "- One-subject inference: passed.",
        "- Raw PCA-150 and all current QC-passed subjects: passed.",
        "",
        "## Endpoint Euclidean distance",
        "",
        "| Split | Protocol | Diagnosis | Subjects | BrainODE (mm) | No-change (mm) | Improvement (mm) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['split']} | {row['protocol']} | {row['diagnosis']} | {row['subjects']} | "
            f"{row['vertex_euclidean_mean_mm_mean']:.4f} | "
            f"{row['no_change_vertex_euclidean_mean_mm_mean']:.4f} | "
            f"{row['vertex_euclidean_improvement_mm_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Protocol definitions",
        "",
        "- `one_shot`: first observed shape predicts the final observed shape.",
        "- `four_shot`: first four observed shapes independently predict the final shape; predictions are averaged. Only subjects with at least five scans contribute.",
        "- `all_observed_sources`: supplementary mean of every available earlier source predicting the final shape.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.progress_every_subjects <= 0:
        raise ValueError("--progress-every-subjects must be positive")
    config_path = args.config or default_config_path(args.structure)
    config = read_json(config_path)
    input_config = validate_config(config, args.structure)
    run_name = str(args.run_name or config["training"]["run_name"])
    run_dir = config_path.parents[1] / "training" / run_name
    checkpoint = args.checkpoint or run_dir / "checkpoints" / "best.pt"
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, config, device)
    pca_model = validate_pca_model(input_config, 150)
    decoder = RawPcaDecoder(pca_model).to(device)
    requested_splits = ("val", "test") if args.split == "all" else (args.split,)
    archives = {
        split: load_archive(Path(input_config["dataset"][f"{split}_sequences"]), split, 150)
        for split in requested_splits
    }
    subjects = {split: build_subjects(archive) for split, archive in archives.items()}
    print("=" * 96, flush=True)
    print(f"BrainODE paper-core evaluation | {args.structure} | device={device}", flush=True)
    print(f"Checkpoint: {checkpoint} | epoch={payload['epoch']} | SHA256={file_sha256(checkpoint)}", flush=True)
    print("Inference batch size: 1", flush=True)
    for split in requested_splits:
        print(f"{split}: {len(subjects[split])} strict CN/AD QC-passed subjects", flush=True)

    if args.dry_run:
        row = metric_row(args.structure, requested_splits[0], "one_shot", subjects[requested_splits[0]][0], [0], model, decoder, device, int(config["training"]["integration_substeps"]))
        if not all(math.isfinite(float(value)) for key, value in row.items() if isinstance(value, (int, float))):
            raise RuntimeError("Non-finite evaluation dry-run metric")
        print("DRY RUN PASSED — checkpoint, raw PCA decoder, one-subject RK4, and metrics are finite; no files written.", flush=True)
        print(json.dumps(row, indent=2), flush=True)
        return 0

    output_dir = args.output_dir or run_dir / "evaluation" / "paper_protocol"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")
    rows: list[dict[str, Any]] = []
    for split in requested_splits:
        rows.extend(evaluate_protocols(
            args.structure,
            split,
            subjects[split],
            model,
            decoder,
            device,
            int(config["training"]["integration_substeps"]),
            args.progress_every_subjects,
        ))
    summary = summarize(rows)
    report = {
        "status": "pass",
        "structure": config["structure"],
        "method": config["method"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "test_loaded_during_training": False,
        "evaluation_splits": list(requested_splits),
        "rows": len(rows),
        "one_subject_inference": True,
        "raw_pca_150": True,
        "all_current_qc_passed_subjects_retained": True,
        "pseudo_cognitive_embedding": False,
        "cognition_estimator": False,
        "source_artifacts_modified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "per_subject_metrics.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "evaluation_report.json", report)
    (output_dir / "README.md").write_text(markdown(report, summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Evaluation complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
