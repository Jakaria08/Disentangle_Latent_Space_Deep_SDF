#!/usr/bin/env python3

import argparse
import csv
import json
import os
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

NORMALIZATION_AGE_MIN = 57.0
NORMALIZATION_AGE_MAX = 91.0
RECORD_FIELDS = [
    "scan_id",
    "mesh_path",
    "sdf_npz_path",
    "subject_id",
    "split",
    "visit_order",
    "months_from_baseline",
    "elapsed_years",
    "baseline_age_years",
    "continuous_age_years",
    "continuous_age_norm",
    "diagnosis",
    "label_ad",
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[4]
    experiment_dir = Path(__file__).resolve().parents[1]
    default_manifest_dir = repo_root / "examples" / "ADNI_1_L_No_MCI" / "brainode_comparison_task1_manifest_original"
    default_task2_dir = (
        repo_root
        / "examples"
        / "ADNI_1_L_No_MCI"
        / "brainode_comparison_task2_representations_original"
        / "inr"
        / "deepsdf_eikonal_spec_fast"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare ADNI No-MCI longitudinal metadata with continuous age and "
            "create a Task 2 DeepSDF compatibility bridge for longitudinal training."
        )
    )
    parser.add_argument(
        "--source-manifest-dir",
        type=Path,
        default=default_manifest_dir,
        help="Task 1 original manifest directory.",
    )
    parser.add_argument(
        "--source-task2-dir",
        type=Path,
        default=default_task2_dir,
        help="Task 2 original DeepSDF directory containing checkpoints/ and latents/.",
    )
    parser.add_argument(
        "--output-experiment-dir",
        type=Path,
        default=experiment_dir,
        help="Output longitudinal experiment directory.",
    )
    return parser.parse_args()


def ensure_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_clean_rows(clean_csv_path):
    with clean_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in clean manifest: {clean_csv_path}")
    return rows


def load_json_list(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def count_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def parse_float(row, key):
    try:
        return float(row[key])
    except Exception as exc:
        raise RuntimeError(f"Could not parse float field '{key}' in row '{row.get('scan_id', '?')}'") from exc


def parse_int(row, key):
    try:
        return int(float(row[key]))
    except Exception as exc:
        raise RuntimeError(f"Could not parse int field '{key}' in row '{row.get('scan_id', '?')}'") from exc


def build_longitudinal_rows(rows):
    by_subject = defaultdict(list)
    for row in rows:
        diagnosis = row["diagnosis"]
        if diagnosis not in {"CN", "AD"}:
            raise RuntimeError(
                f"Found unsupported diagnosis '{diagnosis}' in No-MCI manifest for scan '{row['scan_id']}'"
            )
        by_subject[row["subject_id"]].append(row)

    augmented_by_scan = {}
    repeated_original_age_norm_adjacent = 0
    repeated_continuous_age_norm_adjacent = 0

    for subject_id, subject_rows in by_subject.items():
        ordered = sorted(
            subject_rows,
            key=lambda row: (
                parse_float(row, "months_from_baseline"),
                parse_int(row, "visit_order"),
                row["scan_id"],
            ),
        )

        baseline_month = parse_float(ordered[0], "months_from_baseline")
        baseline_age = parse_float(ordered[0], "age_years")
        if abs(baseline_month) > 1e-8:
            raise RuntimeError(
                f"Subject {subject_id} baseline month is {baseline_month}, expected 0."
            )

        prev_month = None
        prev_original_age_norm = None
        prev_continuous_age_norm = None
        for row in ordered:
            month = parse_float(row, "months_from_baseline")
            if prev_month is not None and month <= prev_month + 1e-8:
                raise RuntimeError(
                    f"Subject {subject_id} has non-increasing months_from_baseline near scan '{row['scan_id']}'."
                )
            elapsed_years = month / 12.0
            continuous_age_years = baseline_age + elapsed_years
            continuous_age_norm = (continuous_age_years - NORMALIZATION_AGE_MIN) / (
                NORMALIZATION_AGE_MAX - NORMALIZATION_AGE_MIN
            )
            if continuous_age_norm < -1e-8 or continuous_age_norm > 1.0 + 1e-8:
                raise RuntimeError(
                    f"Continuous age norm out of range for scan '{row['scan_id']}': {continuous_age_norm}"
                )

            original_age_norm = parse_float(row, "age_norm")
            if prev_original_age_norm is not None and abs(original_age_norm - prev_original_age_norm) <= 1e-8:
                repeated_original_age_norm_adjacent += 1
            if prev_continuous_age_norm is not None and abs(continuous_age_norm - prev_continuous_age_norm) <= 1e-8:
                repeated_continuous_age_norm_adjacent += 1

            augmented_by_scan[row["scan_id"]] = {
                "scan_id": row["scan_id"],
                "mesh_path": row["mesh_path"],
                "sdf_npz_path": row["sdf_npz_path"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "visit_order": int(float(row["visit_order"])),
                "months_from_baseline": float(month),
                "elapsed_years": float(elapsed_years),
                "baseline_age_years": float(baseline_age),
                "continuous_age_years": float(continuous_age_years),
                "continuous_age_norm": float(continuous_age_norm),
                "diagnosis": row["diagnosis"],
                "label_ad": int(float(row["label_ad"])),
            }

            prev_month = month
            prev_original_age_norm = original_age_norm
            prev_continuous_age_norm = continuous_age_norm

    augmented_rows = [augmented_by_scan[row["scan_id"]] for row in rows]
    return (
        augmented_rows,
        {
            "repeated_original_age_norm_adjacent_pairs": repeated_original_age_norm_adjacent,
            "repeated_continuous_age_norm_adjacent_pairs": repeated_continuous_age_norm_adjacent,
        },
    )


def write_metadata_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in RECORD_FIELDS})


def write_labels_pt(path, rows):
    records = [{key: row[key] for key in RECORD_FIELDS} for row in rows]
    payload = {
        "records": records,
        "metadata": {
            "age_min_train": NORMALIZATION_AGE_MIN,
            "age_max_train": NORMALIZATION_AGE_MAX,
            "age_range_years": NORMALIZATION_AGE_MAX - NORMALIZATION_AGE_MIN,
            "elapsed_years_formula": "months_from_baseline / 12",
            "continuous_age_years_formula": "baseline_age_years + months_from_baseline / 12",
            "continuous_age_norm_formula": (
                "(continuous_age_years - 57) / (91 - 57)"
            ),
        },
    }
    torch.save(payload, path)


def extract_latent_weight(latent_codes):
    if torch.is_tensor(latent_codes):
        weight = latent_codes
    elif isinstance(latent_codes, OrderedDict) or isinstance(latent_codes, dict):
        if "weight" not in latent_codes:
            raise RuntimeError("Latent codes dict has no 'weight' entry.")
        weight = latent_codes["weight"]
    else:
        raise RuntimeError(f"Unsupported latent code container type: {type(latent_codes)}")

    if weight.dim() == 3 and weight.shape[1] == 1:
        weight = weight.squeeze(1)
    if weight.dim() != 2:
        raise RuntimeError(f"Expected rank-2 latent weight, got shape {tuple(weight.shape)}")
    return weight.detach().cpu().float()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def refresh_symlink(link_path, target_path):
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Cannot replace directory with symlink: {link_path}")
        link_path.unlink()
    rel_target = os.path.relpath(str(target_path), start=str(link_path.parent))
    link_path.symlink_to(rel_target)


def build_task2_bridge(
    source_task2_dir,
    source_train_split_path,
    bridge_dir,
    metadata_relative_path,
):
    checkpoint_path = ensure_file(source_task2_dir / "checkpoints" / "best.pth")
    run_config_path = ensure_file(source_task2_dir / "run_config.json")
    latent_npz_path = ensure_file(source_task2_dir / "latents" / "train_latents.npz")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_latent_weight = extract_latent_weight(checkpoint["latent_codes"])
    epoch = int(checkpoint.get("epoch", -1))

    train_split_entries = load_json_list(source_train_split_path)
    split_scan_ids = [Path(entry).stem for entry in train_split_entries]
    ckpt_scan_ids = list(checkpoint.get("train_scan_ids", []))
    if not ckpt_scan_ids:
        raise RuntimeError(f"Checkpoint does not contain train_scan_ids: {checkpoint_path}")

    train_npz = np.load(latent_npz_path)
    npz_scan_ids = [str(scan_id) for scan_id in train_npz["scan_ids"].tolist()]
    npz_latents = torch.from_numpy(train_npz["latents"]).detach().cpu().float()

    if split_scan_ids != ckpt_scan_ids:
        raise RuntimeError("Task 1 train split order does not match Task 2 checkpoint scan order.")
    if split_scan_ids != npz_scan_ids:
        raise RuntimeError("Task 1 train split order does not match Task 2 latent npz scan order.")
    if npz_latents.shape[0] != len(split_scan_ids):
        raise RuntimeError(
            f"Latent row count mismatch: {npz_latents.shape[0]} vs {len(split_scan_ids)} train scans"
        )
    if npz_latents.shape[1] != int(run_config["latent_size"]):
        raise RuntimeError(
            f"Latent dim mismatch: {npz_latents.shape[1]} vs config latent size {run_config['latent_size']}"
        )
    if checkpoint_latent_weight.shape != npz_latents.shape:
        raise RuntimeError(
            "Checkpoint latent weight shape does not match train_latents.npz shape: "
            f"{tuple(checkpoint_latent_weight.shape)} vs {tuple(npz_latents.shape)}"
        )
    latent_diff_max_abs = float((checkpoint_latent_weight - npz_latents).abs().max().item())

    model_dir = ensure_dir(bridge_dir / "ModelParameters")
    latent_dir = ensure_dir(bridge_dir / "LatentCodes")

    bridge_split_path = bridge_dir / "train_split.json"
    bridge_specs_path = bridge_dir / "specs.json"
    bridge_latent_best_path = latent_dir / "best.pth"
    bridge_latent_latest_path = latent_dir / "latest.pth"
    bridge_model_best_path = model_dir / "best.pth"
    bridge_model_latest_path = model_dir / "latest.pth"

    bridge_latent_payload = {
        "epoch": epoch,
        "latent_codes": {"weight": npz_latents.clone()},
        "source_train_scan_ids": npz_scan_ids,
        "source_checkpoint": str(checkpoint_path),
        "source_train_latents_npz": str(latent_npz_path),
    }
    torch.save(bridge_latent_payload, bridge_latent_best_path)
    refresh_symlink(bridge_latent_latest_path, bridge_latent_best_path)
    refresh_symlink(bridge_model_best_path, checkpoint_path)
    refresh_symlink(bridge_model_latest_path, bridge_model_best_path)

    bridge_specs = {
        "Description": [
            "Compatibility bridge exposing the Task 2 ADNI No-MCI DeepSDF decoder and training scan latents.",
            "This bridge is intended for pretrained decoder loading and subject-anchor initialization in longitudinal training.",
        ],
        "NetworkArch": run_config["network_arch"],
        "NetworkSpecs": run_config["network_specs"],
        "CodeLength": int(run_config["latent_size"]),
        "ClampingDistance": float(run_config["clamp_distance"]),
        "TrainSplit": bridge_split_path.name,
        "SourceTask1TrainSplit": str(source_train_split_path),
        "SourceTask2RunConfig": str(run_config_path),
        "SourceTask2Checkpoint": str(checkpoint_path),
        "SourceTask2TrainLatents": str(latent_npz_path),
        "LongitudinalMetadataFile": metadata_relative_path,
    }
    write_json(bridge_split_path, train_split_entries)
    write_json(bridge_specs_path, bridge_specs)

    return {
        "bridge_dir": str(bridge_dir),
        "checkpoint_epoch": epoch,
        "train_scan_count": len(split_scan_ids),
        "latent_dim": int(npz_latents.shape[1]),
        "latent_source": "train_latents.npz",
        "latent_order_matches_task1_train_split": True,
        "latent_checkpoint_vs_npz_max_abs_diff": latent_diff_max_abs,
        "bridge_specs_path": str(bridge_specs_path),
        "bridge_train_split_path": str(bridge_split_path),
        "bridge_model_best_path": str(bridge_model_best_path),
        "bridge_model_latest_path": str(bridge_model_latest_path),
        "bridge_latent_best_path": str(bridge_latent_best_path),
        "bridge_latent_latest_path": str(bridge_latent_latest_path),
    }


def verify_split_alignment(rows, split_entries_by_name):
    observed = defaultdict(list)
    for row in rows:
        observed[row["split"]].append(Path(row["scan_id"]).with_suffix(".obj").name)

    report = {}
    for split_name, expected_entries in split_entries_by_name.items():
        observed_set = set(observed[split_name])
        expected_set = set(expected_entries)
        if observed_set != expected_set:
            missing = sorted(expected_set - observed_set)[:5]
            extra = sorted(observed_set - expected_set)[:5]
            raise RuntimeError(
                f"Split alignment mismatch for {split_name}. Missing sample={missing}, extra sample={extra}"
            )
        report[split_name] = {
            "count": len(expected_entries),
            "set_match": True,
        }
    return report


def summarize_rows(
    rows,
    stats,
    bridge_summary,
    source_manifest_dir,
    source_task2_dir,
    split_entries_by_name,
    pair_triplet_counts,
):
    split_counts = Counter(row["split"] for row in rows)
    diagnosis_counts = Counter(row["diagnosis"] for row in rows)
    subject_counts = Counter()
    for row in rows:
        subject_counts[row["subject_id"]] += 1

    continuous_ages = [float(row["continuous_age_years"]) for row in rows]
    continuous_age_norms = [float(row["continuous_age_norm"]) for row in rows]
    unique_subjects = sorted({row["subject_id"] for row in rows})

    return {
        "source_manifest_dir": str(source_manifest_dir),
        "source_task2_dir": str(source_task2_dir),
        "num_scans": len(rows),
        "num_subjects": len(unique_subjects),
        "split_counts": dict(sorted(split_counts.items())),
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "subject_visit_count_histogram": dict(sorted(Counter(subject_counts.values()).items())),
        "metadata_fields_per_scan": RECORD_FIELDS,
        "input_split_json_counts": {
            split_name: len(entries) for split_name, entries in split_entries_by_name.items()
        },
        "input_pair_triplet_counts": pair_triplet_counts,
        "split_alignment": verify_split_alignment(rows, split_entries_by_name),
        "continuous_age_years": {
            "min": min(continuous_ages),
            "max": max(continuous_ages),
        },
        "continuous_age_norm": {
            "min": min(continuous_age_norms),
            "max": max(continuous_age_norms),
            "one_year_delta": 1.0 / (NORMALIZATION_AGE_MAX - NORMALIZATION_AGE_MIN),
        },
        "normalization_constants": {
            "age_min": NORMALIZATION_AGE_MIN,
            "age_max": NORMALIZATION_AGE_MAX,
        },
        "formulas": {
            "elapsed_years": "months_from_baseline / 12",
            "continuous_age_years": "baseline_age_years + months_from_baseline / 12",
            "continuous_age_norm": "(continuous_age_years - 57) / (91 - 57)",
        },
        "adjacent_age_norm_repeat_counts": stats,
        "task2_bridge": bridge_summary,
    }


def main():
    args = parse_args()
    source_manifest_dir = args.source_manifest_dir.resolve()
    source_task2_dir = args.source_task2_dir.resolve()
    output_experiment_dir = args.output_experiment_dir.resolve()

    clean_csv_path = ensure_file(
        source_manifest_dir / "metadata" / "adni_no_mci_left_original_clean.csv"
    )
    age_stats_path = ensure_file(source_manifest_dir / "metadata" / "age_norm_stats.json")
    train_split_path = ensure_file(source_manifest_dir / "splits" / "train_clean.json")
    val_split_path = ensure_file(source_manifest_dir / "splits" / "val_clean.json")
    test_split_path = ensure_file(source_manifest_dir / "splits" / "test_clean.json")
    pair_paths = {
        "train": ensure_file(source_manifest_dir / "pairs" / "pairs_train.csv"),
        "val": ensure_file(source_manifest_dir / "pairs" / "pairs_val.csv"),
        "test": ensure_file(source_manifest_dir / "pairs" / "pairs_test.csv"),
        "all": ensure_file(source_manifest_dir / "pairs" / "pairs_all.csv"),
    }
    triplet_paths = {
        "train": ensure_file(source_manifest_dir / "triplets" / "triplets_train.csv"),
        "val": ensure_file(source_manifest_dir / "triplets" / "triplets_val.csv"),
        "test": ensure_file(source_manifest_dir / "triplets" / "triplets_test.csv"),
        "all": ensure_file(source_manifest_dir / "triplets" / "triplets_all.csv"),
    }

    metadata_dir = ensure_dir(output_experiment_dir / "metadata")
    bridge_dir = ensure_dir(output_experiment_dir / "pretrained_task2_deepsdf")

    metadata_csv_path = metadata_dir / "adni_no_mci_longitudinal_records.csv"
    labels_pt_path = metadata_dir / "adni_no_mci_longitudinal_labels.pt"
    summary_json_path = metadata_dir / "preparation_report.json"

    rows = load_clean_rows(clean_csv_path)
    age_stats = json.loads(age_stats_path.read_text(encoding="utf-8"))
    if float(age_stats["age_min_train"]) != NORMALIZATION_AGE_MIN or float(age_stats["age_max_train"]) != NORMALIZATION_AGE_MAX:
        raise RuntimeError(
            "Task 1 age normalization stats do not match the required constants 57 and 91."
        )

    split_entries_by_name = {
        "train": load_json_list(train_split_path),
        "val": load_json_list(val_split_path),
        "test": load_json_list(test_split_path),
    }
    pair_triplet_counts = {
        "pairs": {name: count_csv_rows(path) for name, path in pair_paths.items()},
        "triplets": {name: count_csv_rows(path) for name, path in triplet_paths.items()},
    }

    augmented_rows, stats = build_longitudinal_rows(rows)
    write_metadata_csv(metadata_csv_path, augmented_rows)
    write_labels_pt(labels_pt_path, augmented_rows)

    bridge_summary = build_task2_bridge(
        source_task2_dir=source_task2_dir,
        source_train_split_path=train_split_path,
        bridge_dir=bridge_dir,
        metadata_relative_path=os.path.relpath(labels_pt_path, start=bridge_dir),
    )

    summary = summarize_rows(
        rows=augmented_rows,
        stats=stats,
        bridge_summary=bridge_summary,
        source_manifest_dir=source_manifest_dir,
        source_task2_dir=source_task2_dir,
        split_entries_by_name=split_entries_by_name,
        pair_triplet_counts=pair_triplet_counts,
    )
    write_json(summary_json_path, summary)

    print(f"Wrote metadata CSV: {metadata_csv_path}")
    print(f"Wrote metadata labels: {labels_pt_path}")
    print(f"Wrote preparation summary: {summary_json_path}")
    print(f"Prepared Task 2 bridge: {bridge_dir}")
    print(
        "Counts:",
        json.dumps(
            {
                "num_scans": summary["num_scans"],
                "num_subjects": summary["num_subjects"],
                "splits": summary["split_counts"],
                "diagnoses": summary["diagnosis_counts"],
            },
            sort_keys=True,
        ),
    )
    print(
        "Adjacent repeated age_norm pairs:",
        json.dumps(summary["adjacent_age_norm_repeat_counts"], sort_keys=True),
    )


if __name__ == "__main__":
    main()
