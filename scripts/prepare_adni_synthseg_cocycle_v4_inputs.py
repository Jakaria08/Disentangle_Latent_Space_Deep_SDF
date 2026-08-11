#!/usr/bin/env python3
"""Prepare independent PCA-150 Cocycle-v4 inputs for hippocampus and LV.

For each structure, this script packages QC-approved, subject-disjoint PCA
sequences and all forward within-subject pairs.  It writes a standalone
age-and-diagnosis Cocycle-v4 configuration.  No pretrained legacy checkpoint,
cross-structure feature, PCA refit, Cocycle training, or Brain-ODE training is
used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
EXPERIMENTS: dict[str, dict[str, str]] = {
    "hippocampus": {
        "structure": "left_hippocampus",
        "root": str(BASE_ROOT / "hippocampus_pca_cocycle_v4"),
        "manifest": "hippocampus_qc_keep_manifest.csv",
    },
    "lateral_ventricle": {
        "structure": "left_lateral_ventricle",
        "root": str(BASE_ROOT / "lateral_ventricle_pca_cocycle_v4"),
        "manifest": "lateral_ventricle_qc_keep_manifest.csv",
    },
}
SPLITS = ("train", "val", "test")
SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}
COMPONENTS = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", choices=("hippocampus", "lateral_ventricle", "all"), default="all")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def string_array(values: list[object]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Cocycle input directory: {path}")


def load_inputs(name: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], Path, dict[str, Any]]:
    spec = EXPERIMENTS[name]
    root = Path(spec["root"])
    manifest_path = root / "metadata" / spec["manifest"]
    pca_archive_path = root / "pca" / "coefficients" / "all_coefficients_safe.npz"
    pca_validation_path = root / "pca" / "metadata" / "pca_archive_finalization.json"
    if not manifest_path.is_file() or not pca_archive_path.is_file() or not pca_validation_path.is_file():
        raise FileNotFoundError("A validated structure manifest, safe PCA archive, and archive-finalization report are required.")
    manifest = pd.read_csv(manifest_path, dtype={"scan_id": str, "subject_id": str, "VISCODE": str})
    report = json.loads(pca_validation_path.read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("safe_all_archive") != str(pca_archive_path):
        raise ValueError(f"PCA archive finalization is not valid for {name}.")
    with np.load(pca_archive_path, allow_pickle=False) as archive:
        pca = {key: archive[key] for key in archive.files}
    required_manifest = {
        "scan_id", "subject_id", "split", "diagnosis", "label_ad", "visit_order", "visit_month", "months_from_baseline",
        "age_years", "age_norm_train", "correspondence_volume_mm3", "correspondence_surface_area_mm2",
    }
    required_pca = {
        "scan_ids", "subject_ids", "splits", "diagnoses", "label_ad", "visit_orders", "visit_months", "age_years",
        "age_norm_train", "pca_150", "pca_standardized_150", "train_pca_mean_150", "train_pca_std_150",
    }
    missing_manifest = sorted(required_manifest.difference(manifest.columns))
    missing_pca = sorted(required_pca.difference(pca))
    if missing_manifest or missing_pca:
        raise KeyError(f"Missing manifest={missing_manifest}, PCA={missing_pca}")
    if len(manifest) != len(pca["scan_ids"]) or not np.array_equal(manifest["scan_id"].astype(str).to_numpy(), pca["scan_ids"].astype(str)):
        raise ValueError("Safe PCA archive scan order does not match its structure manifest.")
    if pca["pca_150"].shape != (len(manifest), COMPONENTS) or pca["pca_standardized_150"].shape != (len(manifest), COMPONENTS):
        raise ValueError("PCA coefficient dimensions are not PCA-150.")
    if not np.isfinite(pca["pca_150"]).all() or not np.isfinite(pca["pca_standardized_150"]).all():
        raise ValueError("PCA coefficients contain non-finite values.")
    return manifest, pca, pca_archive_path, report


def ordered_frame(manifest: pd.DataFrame, pca: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = manifest.copy()
    frame["_pca_index"] = np.arange(len(frame), dtype=np.int64)
    frame["_split_order"] = frame["split"].map(SPLIT_ORDER)
    frame = frame.sort_values(["_split_order", "subject_id", "visit_order", "visit_month", "scan_id"], kind="stable").reset_index(drop=True)
    if frame.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("Subject split leakage in Cocycle input manifest.")
    if (frame.groupby("subject_id")["scan_id"].nunique() < 2).any():
        raise ValueError("Every Cocycle participant must have two visits.")
    if frame.duplicated(["subject_id", "visit_month"], keep=False).any():
        raise ValueError("Duplicate visit time in Cocycle input.")
    for _, group in frame.groupby("subject_id", sort=False):
        if (group["visit_month"].diff().dropna() <= 0).any():
            raise ValueError("Non-increasing visit time in Cocycle input.")
        if group["diagnosis"].nunique() != 1 or group["label_ad"].nunique() != 1:
            raise ValueError("Diagnosis must remain fixed within each Cocycle subject sequence.")
    # Reorder PCA rows by the same canonical sequence ordering.
    order = frame["_pca_index"].to_numpy(dtype=np.int64)
    for key in ("pca_150", "pca_standardized_150"):
        if pca[key].shape[0] != len(order):
            raise ValueError(f"PCA row count mismatch for {key}")
    return frame


def package_sequences(frame: pd.DataFrame, pca: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    order = frame["_pca_index"].to_numpy(dtype=np.int64)
    raw_scores = pca["pca_150"][order].astype(np.float32)
    standardized_scores = pca["pca_standardized_150"][order].astype(np.float32)
    subject_rows = frame.drop_duplicates("subject_id", keep="first")
    offsets = [0]
    pairs: list[dict[str, Any]] = []
    for subject_index, (_, group) in enumerate(frame.groupby("subject_id", sort=False)):
        start = int(offsets[-1])
        end = start + len(group)
        local = group.reset_index(drop=True)
        for source_local in range(len(local) - 1):
            for target_local in range(source_local + 1, len(local)):
                source = local.iloc[source_local]
                target = local.iloc[target_local]
                intermediate_local = source_local + (target_local - source_local) // 2 if target_local - source_local > 1 else -1
                delta_years = (float(target["visit_month"]) - float(source["visit_month"])) / 12.0
                if delta_years <= 0:
                    raise ValueError("Non-positive forward time interval while packaging pairs.")
                pairs.append(
                    {
                        "split": str(source["split"]),
                        "diagnosis": str(source["diagnosis"]),
                        "label_ad": int(source["label_ad"]),
                        "subject_id": str(source["subject_id"]),
                        "source_index": start + source_local,
                        "target_index": start + target_local,
                        "intermediate_index": start + intermediate_local if intermediate_local >= 0 else -1,
                        "source_scan_id": str(source["scan_id"]),
                        "target_scan_id": str(target["scan_id"]),
                        "source_visit_order": int(source["visit_order"]),
                        "target_visit_order": int(target["visit_order"]),
                        "pair_type": "adjacent" if target_local - source_local == 1 else "nonadjacent",
                        "delta_years": delta_years,
                    }
                )
        offsets.append(end)

    arrays = {
        "subject_ids": string_array(subject_rows["subject_id"].tolist()),
        "subject_splits": string_array(subject_rows["split"].tolist()),
        "subject_diagnoses": string_array(subject_rows["diagnosis"].tolist()),
        "subject_label_ad": subject_rows["label_ad"].to_numpy(dtype=np.int64),
        "subject_baseline_age_years": subject_rows.groupby("subject_id", sort=False)["age_years"].first().to_numpy(dtype=np.float32),
        "subject_visit_offsets": np.asarray(offsets, dtype=np.int64),
        "visit_scan_ids": string_array(frame["scan_id"].tolist()),
        "visit_subject_ids": string_array(frame["subject_id"].tolist()),
        "visit_splits": string_array(frame["split"].tolist()),
        "visit_diagnoses": string_array(frame["diagnosis"].tolist()),
        "visit_label_ad": frame["label_ad"].to_numpy(dtype=np.int64),
        "visit_orders": frame["visit_order"].to_numpy(dtype=np.int64),
        "visit_months_from_baseline": frame["months_from_baseline"].to_numpy(dtype=np.float32),
        "visit_time_years_from_baseline": (frame["months_from_baseline"].to_numpy(dtype=np.float32) / np.float32(12.0)),
        "visit_age_years": frame["age_years"].to_numpy(dtype=np.float32),
        "visit_age_norm_train": frame["age_norm_train"].to_numpy(dtype=np.float32),
        "visit_pca_150": raw_scores,
        "visit_pca_standardized_150": standardized_scores,
        "visit_volume_mm3": frame["correspondence_volume_mm3"].to_numpy(dtype=np.float32),
        "visit_surface_area_mm2": frame["correspondence_surface_area_mm2"].to_numpy(dtype=np.float32),
        "train_pca_mean_150": pca["train_pca_mean_150"].astype(np.float32),
        "train_pca_std_150": pca["train_pca_std_150"].astype(np.float32),
    }
    return arrays, pairs


def subset_archive(all_archive: dict[str, np.ndarray], split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    subject_mask = all_archive["subject_splits"].astype(str) == split
    subject_indices = np.flatnonzero(subject_mask)
    visit_indices: list[int] = []
    offsets = [0]
    original_offsets = all_archive["subject_visit_offsets"]
    for subject_index in subject_indices:
        start, end = int(original_offsets[subject_index]), int(original_offsets[subject_index + 1])
        visit_indices.extend(range(start, end))
        offsets.append(len(visit_indices))
    indices = np.asarray(visit_indices, dtype=np.int64)
    output: dict[str, np.ndarray] = {}
    for key, value in all_archive.items():
        if key.startswith("subject_") and key != "subject_visit_offsets":
            output[key] = value[subject_indices]
        elif key.startswith("visit_"):
            output[key] = value[indices]
        elif key == "subject_visit_offsets":
            output[key] = np.asarray(offsets, dtype=np.int64)
        else:
            output[key] = value
    return output, indices


def package_split_pairs(all_pairs: list[dict[str, Any]], split: str, all_archive: dict[str, np.ndarray], split_archive: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Remap global sequence indices to split-local indices for convenience."""
    split_subjects = set(split_archive["subject_ids"].astype(str))
    global_to_local: dict[str, int] = {str(scan_id): index for index, scan_id in enumerate(split_archive["visit_scan_ids"].astype(str))}
    rows = []
    for row in all_pairs:
        if row["split"] != split or row["subject_id"] not in split_subjects:
            continue
        current = dict(row)
        current["source_index"] = global_to_local[current["source_scan_id"]]
        current["target_index"] = global_to_local[current["target_scan_id"]]
        if current["intermediate_index"] >= 0:
            global_scan = str(all_archive["visit_scan_ids"][current["intermediate_index"]])
            current["intermediate_index"] = global_to_local[global_scan]
        rows.append(current)
    return rows


def validate_archive(path: Path, expected_split: str | None, expected_components: int = COMPONENTS) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        if value.dtype == object:
            raise RuntimeError(f"Pickle-dependent array in Cocycle archive: {path} / {key}")
    visits = len(arrays["visit_scan_ids"])
    subjects = len(arrays["subject_ids"])
    offsets = arrays["subject_visit_offsets"]
    if offsets.shape != (subjects + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != visits:
        raise RuntimeError(f"Invalid subject offsets in {path}")
    if arrays["visit_pca_150"].shape != (visits, expected_components) or arrays["visit_pca_standardized_150"].shape != (visits, expected_components):
        raise RuntimeError(f"Invalid PCA shape in {path}")
    if not np.isfinite(arrays["visit_pca_150"]).all() or not np.isfinite(arrays["visit_pca_standardized_150"]).all():
        raise RuntimeError(f"Non-finite PCA values in {path}")
    if expected_split is not None and set(arrays["visit_splits"].astype(str)) != {expected_split}:
        raise RuntimeError(f"Split contamination in {path}")
    for subject_index in range(subjects):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        if end - start < 2:
            raise RuntimeError(f"Subject with fewer than two visits in {path}")
        time = arrays["visit_time_years_from_baseline"][start:end]
        if np.any(np.diff(time) <= 0):
            raise RuntimeError(f"Non-increasing subject time in {path}")
    return {"visits": visits, "subjects": subjects, "pca_shape": list(arrays["visit_pca_150"].shape)}


def config_payload(name: str, root: Path, pca_archive_path: Path, pca_report: dict[str, Any]) -> dict[str, Any]:
    spec = EXPERIMENTS[name]
    cocycle = root / "cocycle_v4"
    return {
        "name": f"adni_synthseg_{name}_pca150_age_disease_cocycle_v4",
        "status": "prior-v4 trainer ready; no training has run",
        "structure": spec["structure"],
        "representation": {
            "components": COMPONENTS,
            "training_latent_key": "visit_pca_standardized_150",
            "reconstruction_latent_key": "visit_pca_150",
            "standardization": "train PCA-score mean/std only",
            "pca_archive": str(pca_archive_path),
            "pca_archive_finalization": str(root / "pca" / "metadata" / "pca_archive_finalization.json"),
            "pca_model_dir": str(root / "pca" / "model"),
            "pca_refitted": False,
        },
        "dataset": {
            "all_sequences": str(cocycle / "dataset" / "all_subject_sequences.npz"),
            "train_sequences": str(cocycle / "dataset" / "train_subject_sequences.npz"),
            "val_sequences": str(cocycle / "dataset" / "val_subject_sequences.npz"),
            "test_sequences": str(cocycle / "dataset" / "test_subject_sequences.npz"),
            "train_pairs": str(cocycle / "pairs" / "train_forward_pairs.csv"),
            "val_pairs": str(cocycle / "pairs" / "val_forward_pairs.csv"),
            "test_pairs": str(cocycle / "pairs" / "test_forward_pairs.csv"),
            "split_unit": "subject",
            "pair_policy": "store each observed within-subject pair once in chronological order; derive reverse transport from the same endpoints during training; retain adjacent/nonadjacent labels; no cross-subject pairs",
            "time": "absolute age_norm_train for transport; elapsed years retained for reporting",
            "condition": "stable diagnosis label_ad (CN=0, AD=1)",
        },
        "model": {
            "type": "standalone_direct_age_disease_cocycle_v4",
            "latent_dim": COMPONENTS,
            "condition_dim": 1,
            "hidden_dims": [128, 128],
            "activation": "silu",
            "dropout": 0.05,
            "zero_initialize_output": True,
            "include_delta_time_input": True,
            "legacy_checkpoint_used": False,
        },
        "training": {
            "epochs": 150,
            "batch_size": 128,
            "sequence_batch_size": 32,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "validation_frequency": 5,
            "early_stopping_patience": 50,
            "seed": 42,
            "pair_sampling": "mixed_adjacent_nonadjacent with equal 0.5 probability, adapted from prior v4 mixed_adjacent_far sampling",
        },
        "loss": {
            "real_pair_forward_weight": 1.0,
            "real_pair_backward_weight": 1.0,
            "cocycle_forward_weight": 0.01,
            "cocycle_backward_weight": 0.01,
            "zero_displacement_weight": 0.001,
            "speed_guard_weight": 0.0,
            "pair_forward_enabled": False,
            "pair_backward_enabled": False,
            "general_cocycle_enabled": False,
        },
        "prior_v4_losses": {
            "source_experiment": "ADNI longitudinal_age_disease_conditioned_cocycle_shape_multiple_pairs_real_pair_shape_reconstruction",
            "UseCocycleLoss": True,
            "UseCocycleBackwardLoss": True,
            "UsePairForwardLoss": False,
            "UsePairBackwardLoss": False,
            "UseGeneralCocycleLoss": False,
            "UseGeneralCocycleBackwardLoss": False,
            "UseRealScanPairLoss": True,
            "UseRealScanPairForward": True,
            "UseRealScanPairBackward": True,
            "CocycleLossLambda": 0.01,
            "CocycleBackwardLossLambda": 0.01,
            "ZeroDisplacementLambda": 0.001,
            "adaptation_note": "PCA standardized-score endpoint MSE replaces SDF reconstruction/latent supervision; its forward and backward weights are 1.0 because it is the primary observed-pair objective.",
        },
        "evaluation": {
            "splits": ["val", "test"],
            "metrics": ["standardized_pca_mse", "decoded_vertex_mae_mm", "decoded_volume_relative_error_pct", "adjacent_and_long_gap_error", "CN_AD_speed_difference"],
            "biological_check": "hippocampus expected to decline and LV expected to expand on average; this is evaluated, not hard-constrained",
        },
        "pca_validation": pca_report["safe_all_archive"],
    }


def prepare_structure(name: str) -> dict[str, Any]:
    spec = EXPERIMENTS[name]
    root = Path(spec["root"])
    cocycle_root = root / "cocycle_v4"
    require_empty(cocycle_root)
    manifest, pca, pca_archive_path, pca_report = load_inputs(name)
    frame = ordered_frame(manifest, pca)
    all_archive, all_pairs = package_sequences(frame, pca)
    cocycle_root.joinpath("dataset").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cocycle_root / "dataset" / "all_subject_sequences.npz", **all_archive)
    pair_fields = [
        "split", "diagnosis", "label_ad", "subject_id", "source_index", "target_index", "intermediate_index",
        "source_scan_id", "target_scan_id", "source_visit_order", "target_visit_order", "pair_type", "delta_years",
    ]
    split_summaries: dict[str, Any] = {}
    for split in SPLITS:
        archive, _indices = subset_archive(all_archive, split)
        archive_path = cocycle_root / "dataset" / f"{split}_subject_sequences.npz"
        np.savez_compressed(archive_path, **archive)
        pairs = package_split_pairs(all_pairs, split, all_archive, archive)
        pair_path = cocycle_root / "pairs" / f"{split}_forward_pairs.csv"
        write_csv(pair_path, pairs, pair_fields)
        archive_summary = validate_archive(archive_path, split)
        split_summaries[split] = {
            **archive_summary,
            "forward_pairs": int(len(pairs)),
            "adjacent_pairs": int(sum(row["pair_type"] == "adjacent" for row in pairs)),
            "nonadjacent_pairs": int(sum(row["pair_type"] == "nonadjacent" for row in pairs)),
            "pairs_by_diagnosis": {
                diagnosis: int(sum(row["diagnosis"] == diagnosis for row in pairs)) for diagnosis in ("CN", "AD")
            },
        }
    all_summary = validate_archive(cocycle_root / "dataset" / "all_subject_sequences.npz", None)
    config = config_payload(name, root, pca_archive_path, pca_report)
    config_path = cocycle_root / "configs" / "cocycle_v4_primary.json"
    write_json(config_path, config)
    validation = {
        "passed": True,
        "source_meshes_modified": False,
        "pca_refitted": False,
        "cocycle_trained": False,
        "brainode_trained": False,
        "structure": spec["structure"],
        "safe_pca_archive": str(pca_archive_path),
        "all_sequences": all_summary,
        "splits": split_summaries,
        "config": str(config_path),
        "checks": {
            "allow_pickle_false": True,
            "subject_split_leakage": False,
            "cross_subject_pairs": False,
            "minimum_visits": 2,
            "strictly_increasing_time": True,
            "condition_is_diagnosis_not_cognition": True,
            "cross_structure_features": False,
            "legacy_checkpoint_used": False,
        },
    }
    write_json(cocycle_root / "metadata" / "cocycle_input_validation.json", validation)
    write_json(cocycle_root / "metadata" / "cocycle_dataset_summary.json", {
        "status": "pass", "structure": spec["structure"], "all": all_summary, "splits": split_summaries,
        "pca_archive": str(pca_archive_path), "config": str(config_path),
    })
    return validation


def main() -> int:
    args = parse_args()
    names = tuple(EXPERIMENTS) if args.structure == "all" else (args.structure,)
    print("=" * 88)
    print("Prepare independent PCA-150 Cocycle-v4 input datasets (no training)")
    print("=" * 88, flush=True)
    results = {}
    for name in names:
        print(f"Preparing {name}…", flush=True)
        results[name] = prepare_structure(name)
        print(f"  complete: {results[name]['config']}", flush=True)
    print(json.dumps({name: result["splits"] for name, result in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
