#!/usr/bin/env python3
"""Paper-oriented evaluation of fixed-label and voxel-feedback PCA128 BrainODE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import brainode_cognition as B
import c4_objective as O
import common as C
import evaluate as E
from models import build_ode
from train_brainode_cognition import ode_config, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--evaluation-name", default="brainode_cognition_evaluation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cognition_probabilities(
    model: B.VoxelCognitionCNN,
    masks: np.ndarray,
    device: torch.device,
    temperature: float,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(masks), int(batch_size)):
            tensor = torch.from_numpy(masks[start : start + int(batch_size)]).to(device, dtype=torch.float32)
            output.append(torch.sigmoid(model(tensor) / float(temperature)).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def decoded_cognition_probabilities(
    feedback: B.CognitionFeedbackTransport,
    latent: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    output = []
    for start in range(0, len(latent), int(batch_size)):
        output.append(feedback.estimate_condition(latent[start : start + int(batch_size)]).cpu().numpy())
    return np.concatenate(output)


def metric_groups(records: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    return {
        diagnosis: {
            "rows": int(sum(row["diagnosis"] == diagnosis for row in records)) if diagnosis != "overall" else len(records),
            **{
                f"{metric}_mean": float(np.mean([
                    row[metric] for row in records if diagnosis == "overall" or row["diagnosis"] == diagnosis
                ]))
                for metric in metrics
                if any(diagnosis == "overall" or row["diagnosis"] == diagnosis for row in records)
            },
        }
        for diagnosis in ("CN", "AD", "overall")
    }


def pair_groups_from_rows(records: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = {"subject", "diagnosis", "pair_type", "source_index", "target_index", "delta_years"}
    metrics = sorted(set(records[0]).difference(metadata)) if records else []
    output: dict[str, Any] = {}
    for diagnosis in ("CN", "AD", "overall"):
        selected = [row for row in records if diagnosis == "overall" or row["diagnosis"] == diagnosis]
        output[diagnosis] = {"rows": len(selected)}
        if selected:
            output[diagnosis].update({
                f"{metric}_mean": float(np.mean([row[metric] for row in selected])) for metric in metrics
            })
    return output


@torch.no_grad()
def condition_injectivity(
    fixed: E.ODETransport,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    rows: list[C.PairRow],
    batch_size: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), int(batch_size)):
        current = rows[start : start + int(batch_size)]
        source = torch.tensor([row.source for row in current], device=values["z"].device)
        target = torch.tensor([row.target for row in current], device=values["z"].device)
        z = values["z"][source]
        source_age, target_age = values["age"][source], values["age"][target]
        cn = fixed.transport(z, source_age, target_age, torch.zeros(len(current), device=z.device))
        ad = fixed.transport(z, source_age, target_age, torch.ones(len(current), device=z.device))
        source_vertices = geometry.vertices(z)
        cn_vertices, ad_vertices = geometry.vertices(cn), geometry.vertices(ad)
        cn_volume = geometry.volume_from_vertices(cn_vertices)
        ad_volume = geometry.volume_from_vertices(ad_vertices)
        for index, row in enumerate(current):
            records.append({
                "subject": row.subject,
                "diagnosis": row.diagnosis,
                "latent_rmse_ad_vs_cn": float(torch.sqrt(torch.mean((ad[index] - cn[index]).square())).cpu()),
                "vertex_coordinate_mae_ad_vs_cn_mm": float(torch.mean(torch.abs(ad_vertices[index] - cn_vertices[index])).cpu()),
                "vertex_euclidean_ad_vs_cn_mm": float(torch.linalg.vector_norm(ad_vertices[index] - cn_vertices[index], dim=1).mean().cpu()),
                "signed_volume_difference_ad_minus_cn_relative": float(((ad_volume[index] - cn_volume[index]) / cn_volume[index]).cpu()),
                "cn_displacement_from_source_mm": float(torch.linalg.vector_norm(cn_vertices[index] - source_vertices[index], dim=1).mean().cpu()),
                "ad_displacement_from_source_mm": float(torch.linalg.vector_norm(ad_vertices[index] - source_vertices[index], dim=1).mean().cpu()),
            })
    metrics = [key for key in records[0] if key not in {"subject", "diagnosis"}] if records else []
    return {"interpretation": "same source and ages, c=0 versus c=1; condition injectivity, not conversion", "groups": metric_groups(records, metrics)}


@torch.no_grad()
def n_shot_metrics(
    transport: torch.nn.Module,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    raw_vertices: np.ndarray,
    shots: list[int],
    max_subjects: int | None,
) -> dict[str, Any]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subject_limit = len(offsets) - 1 if max_subjects is None else min(len(offsets) - 1, int(max_subjects))
    output: dict[str, Any] = {}
    for shot in shots:
        records: list[dict[str, Any]] = []
        for subject_index in range(subject_limit):
            first, last = int(offsets[subject_index]), int(offsets[subject_index + 1])
            if last - first <= int(shot):
                continue
            sources = torch.arange(first, first + int(shot), device=values["z"].device)
            target_index = last - 1
            target = torch.full((int(shot),), target_index, dtype=torch.long, device=values["z"].device)
            predictions = transport.transport(
                values["z"][sources], values["age"][sources], values["age"][target], values["label"][sources]
            )
            predicted_vertices = geometry.vertices(predictions).mean(dim=0, keepdim=True)
            target_vertices = values["reference_vertices"][target_index : target_index + 1]
            raw_target = torch.from_numpy(np.asarray(raw_vertices[target_index : target_index + 1], dtype=np.float32).copy()).to(predicted_vertices.device)
            source_vertices = values["reference_vertices"][sources].mean(dim=0, keepdim=True)
            predicted_volume = geometry.volume_from_vertices(predicted_vertices)
            target_volume = geometry.volume_from_vertices(target_vertices)
            records.append({
                "subject": str(archive["subject_ids"][subject_index]),
                "diagnosis": str(archive["subject_diagnoses"][subject_index]),
                "transport_coordinate_mae_mm": float(torch.mean(torch.abs(predicted_vertices - target_vertices)).cpu()),
                "transport_euclidean_mm": float(torch.linalg.vector_norm(predicted_vertices - target_vertices, dim=2).mean().cpu()),
                "end_to_end_rmse_mm": float(torch.sqrt(torch.mean((predicted_vertices - raw_target).square())).cpu()),
                "volume_relative": float((torch.abs(predicted_volume - target_volume) / target_volume).cpu()),
                "nochange_coordinate_mae_mm": float(torch.mean(torch.abs(source_vertices - target_vertices)).cpu()),
            })
        metrics = [key for key in records[0] if key not in {"subject", "diagnosis"}] if records else []
        output[f"{shot}_shot"] = {
            "definition": f"average mesh prediction from the first {shot} observed visits to the final visit",
            "groups": metric_groups(records, metrics),
        }
    return output


@torch.no_grad()
def feedback_condition_summary(
    feedback: B.CognitionFeedbackTransport,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    observed_probabilities: np.ndarray,
    max_subjects: int | None,
) -> dict[str, Any]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subject_limit = len(offsets) - 1 if max_subjects is None else min(len(offsets) - 1, int(max_subjects))
    records = []
    for index in range(subject_limit):
        first, last = int(offsets[index]), int(offsets[index + 1])
        start, target = first, last - 1
        predicted = feedback.transport(
            values["z"][start : start + 1], values["age"][start : start + 1], values["age"][target : target + 1]
        )
        start_condition = float(feedback.estimate_condition(values["z"][start : start + 1]).cpu())
        end_condition = float(feedback.estimate_condition(predicted).cpu())
        records.append({
            "subject": str(archive["subject_ids"][index]),
            "diagnosis": str(archive["subject_diagnoses"][index]),
            "predicted_start_condition": start_condition,
            "predicted_end_condition": end_condition,
            "predicted_condition_change": end_condition - start_condition,
            "observed_start_condition": float(observed_probabilities[start]),
            "observed_target_condition": float(observed_probabilities[target]),
            "endpoint_condition_absolute_error_to_observed_shape_score": abs(end_condition - float(observed_probabilities[target])),
        })
    metrics = [key for key in records[0] if key not in {"subject", "diagnosis"}] if records else []
    return {
        "interpretation": "condition is calibrated AD-like anatomy probability, not clinical cognition or conversion probability",
        "groups": metric_groups(records, metrics),
        "subject_records": records,
    }


def evaluate_transport(
    transport: torch.nn.Module,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    raw_vertices: np.ndarray,
    categories: dict[str, list[C.PairRow]],
    forward: list[C.PairRow],
    defect_statistics: dict[str, Any],
    batch_size: int,
    max_subjects: int | None,
    bootstrap_samples: int,
) -> dict[str, Any]:
    started = time.time()
    # Compute each direction once. Adjacent/non-adjacent, first-last, and
    # horizon strata are exact subsets and can be aggregated from per-pair
    # metrics without repeating expensive mesh voxelization.
    bases = {
        "forward": O.evaluate_pairs(
            transport, geometry, values, categories["all_forward"], raw_vertices, batch_size, include_rows=True
        ),
        "backward": O.evaluate_pairs(
            transport, geometry, values, categories["all_backward"], raw_vertices, batch_size, include_rows=True
        ),
    }
    pairs: dict[str, Any] = {}
    for name, rows in categories.items():
        if not rows:
            continue
        direction = "backward" if name.endswith("backward") else "forward"
        base_rows = bases[direction]["row_metrics"]
        requested = {(row.subject, int(row.source), int(row.target)) for row in rows}
        selected = [
            row for row in base_rows
            if (str(row["subject"]), int(row["source_index"]), int(row["target_index"])) in requested
        ]
        if len(selected) != len(requested):
            # A separately balanced diagnostic limit can choose first-last
            # cases absent from the limited all-pair set.
            pairs[name] = O.evaluate_pairs(
                transport, geometry, values, rows, raw_vertices, batch_size, include_rows=(name == "all_forward")
            )
        else:
            pairs[name] = {"groups": pair_groups_from_rows(selected)}
            if name == "all_forward":
                pairs[name]["row_metrics"] = selected
    return {
        "pair_metrics": pairs,
        "sequence_metrics": E.sequence_metrics(transport, geometry, values, archive, raw_vertices, max_subjects),
        "consistency_defects": O.cocycle_defects(transport, values, forward, defect_statistics, batch_size),
        "subject_bootstrap": E.subject_bootstrap(pairs["all_forward"].get("row_metrics", []), bootstrap_samples),
        "evaluation_seconds": time.time() - started,
    }


def main() -> int:
    args = parse_args()
    C.validate_run_name(args.evaluation_name)
    run_dir = args.run_dir.expanduser().resolve()
    resolved = C.read_json(run_dir / "resolved_config.json")
    config = resolved["config"]
    validate_config(config)
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "combined_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bool(checkpoint.get("test_data_loaded", True)):
        raise ValueError("Checkpoint does not certify test isolation")
    device = C.choose_device(args.device)
    registry = C.load_registry()
    train_archive = C.load_archive("pca128", "train", registry)
    archive = C.load_archive("pca128", args.split, registry)
    resolution = int(config["cognition_estimator"]["voxel_resolution"])
    voxels, grid = B.load_voxel_archive(registry, args.split, resolution, archive)
    if grid.mapping() != checkpoint["voxel_grid"]:
        raise ValueError("Evaluation voxel grid differs from selected checkpoint")
    geometry = C.build_geometry("pca128", train_archive, device, registry)
    values = C.values_on_device(archive, device)
    O.attach_reference_geometry(values, geometry, 256)
    raw_vertices = C.cached_vertices(args.split, registry)
    function = build_ode(ode_config(config)).to(device)
    function.load_state_dict(checkpoint["ode_state_dict"], strict=True)
    function.eval()
    estimator = B.VoxelCognitionCNN(
        int(config["cognition_estimator"]["base_channels"]), float(config["cognition_estimator"]["dropout"])
    ).to(device)
    estimator.load_state_dict(checkpoint["cognition_state_dict"], strict=True)
    estimator.eval()
    substeps = int(config["training"]["integration_substeps"])
    fixed = E.ODETransport(function, substeps).to(device).eval()
    feedback = B.CognitionFeedbackTransport(
        function, geometry, estimator, grid, substeps, float(checkpoint["temperature"])
    ).to(device).eval()
    observed_probabilities = cognition_probabilities(
        estimator, voxels["masks"], device, float(checkpoint["temperature"]),
        int(config["cognition_estimator"]["evaluation_batch_size"]),
    )
    cognition_metrics = B.scan_and_subject_metrics(
        archive["visit_label_ad"].astype(np.int64), observed_probabilities, archive["visit_subject_ids"]
    )
    decoded_probabilities = decoded_cognition_probabilities(feedback, values["z"], 16)
    decoded_cognition_metrics = B.scan_and_subject_metrics(
        archive["visit_label_ad"].astype(np.int64), decoded_probabilities, archive["visit_subject_ids"]
    )
    forward = E.limited_pairs(C.load_pairs(args.split, archive, registry), args.max_pairs)
    first_last = E.limited_pairs(C.first_last_pairs(archive), args.max_pairs, first_last=True)
    backward, first_last_backward = E.reverse_rows(forward), E.reverse_rows(first_last)
    categories = {
        "all_forward": forward,
        "adjacent_forward": [row for row in forward if row.pair_type == "adjacent"],
        "nonadjacent_forward": [row for row in forward if row.pair_type == "nonadjacent"],
        "first_last_forward": first_last,
        "all_backward": backward,
        "adjacent_backward": [row for row in backward if row.pair_type.endswith("adjacent") and "nonadjacent" not in row.pair_type],
        "nonadjacent_backward": [row for row in backward if row.pair_type.endswith("nonadjacent")],
        "first_last_backward": first_last_backward,
    }
    categories.update({f"horizon_{name}_forward": rows for name, rows in E.horizon_groups(forward).items() if rows})
    categories.update({f"horizon_{name}_backward": rows for name, rows in E.horizon_groups(backward).items() if rows})
    train_pairs = C.load_pairs("train", train_archive, registry)
    displacement = np.asarray([
        np.sqrt(np.mean((train_archive["visit_latent_standardized_128"][row.target] - train_archive["visit_latent_standardized_128"][row.source]) ** 2))
        for row in train_pairs
    ])
    defect_statistics = {"normalization_scales": {"displacement": float(max(np.median(displacement), 1.0e-6))}}
    fixed_metrics = evaluate_transport(
        fixed, geometry, values, archive, raw_vertices, categories, forward, defect_statistics,
        128, args.max_subjects, args.bootstrap_samples,
    )
    feedback_metrics = evaluate_transport(
        feedback, geometry, values, archive, raw_vertices, categories, forward, defect_statistics,
        16, args.max_subjects, args.bootstrap_samples,
    )
    shots = [int(value) for value in config["inference"]["n_shot"]]
    fixed_metrics["n_shot"] = n_shot_metrics(fixed, geometry, values, archive, raw_vertices, shots, args.max_subjects)
    feedback_metrics["n_shot"] = n_shot_metrics(feedback, geometry, values, archive, raw_vertices, shots, args.max_subjects)
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "representation": "pca128",
        "method": "brainode_cognition",
        "anatomy": "left_hippocampus",
        "representation_floor": E.representation_floor(values, raw_vertices, geometry, 256),
        "cognition_estimator": {
            "input": B.VoxelCognitionCNN.input_contract,
            "temperature": float(checkpoint["temperature"]),
            "raw_observed_mesh_metrics": cognition_metrics,
            "pca_decoded_observed_latent_metrics": decoded_cognition_metrics,
            "raw_vs_pca_decoded_probability_mae": float(np.mean(np.abs(observed_probabilities - decoded_probabilities))),
            "feedback_domain": "pca_decoded_observed_latent_metrics",
        },
        "transports": {"fixed_observed_label": fixed_metrics, "voxel_cognition_feedback": feedback_metrics},
        "condition_injectivity": condition_injectivity(fixed, geometry, values, first_last, 128),
        "feedback_condition_trajectories": feedback_condition_summary(
            feedback, values, archive, observed_probabilities, args.max_subjects
        ),
        "cohort_contract": {
            "diagnoses": sorted(np.unique(archive["visit_diagnoses"].astype(str)).tolist()),
            "mci_scans": int(np.count_nonzero(archive["visit_diagnoses"].astype(str) == "MCI")),
            "converter_subjects": 0,
            "pseudo_cognition_samples": 0,
            "conversion_claim_allowed": False,
            "valid_claim": "conditional CN-versus-AD anatomy trajectories and volume/surface trends",
        },
        "paper_mapping": {
            "pca_latent": "128-D here versus 150-D in the paper",
            "shape_cognition_estimator": True,
            "fixed_condition_ablation": True,
            "estimator_without_pseudo_sampling_ablation": True,
            "converter_pseudo_sampling": False,
            "one_two_four_shot": True,
            "forward_backward_irregular_horizon_evaluation": True,
            "condition_injectivity": True,
        },
        "test_loaded_during_training": False,
        "source_meshes_modified": False,
    }
    if args.dry_run:
        print("BRAINODE COGNITION EVALUATION DRY RUN PASSED — no files written.")
        print(json.dumps({
            "split": args.split, "cognition_estimator": report["cognition_estimator"],
            "fixed_first_last": fixed_metrics["pair_metrics"]["first_last_forward"]["groups"]["overall"],
            "feedback_first_last": feedback_metrics["pair_metrics"]["first_last_forward"]["groups"]["overall"],
        }, indent=2, sort_keys=True))
        return 0
    destination = run_dir / args.evaluation_name / args.split
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    C.atomic_json(destination / "summary.json", report)
    print(f"WROTE {destination / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
