#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from networks.deep_sdf_decoder import Decoder


EXPECTED_SCAN_COUNT = 727
EXPECTED_SUBJECT_COUNT = 244
EXPECTED_SPLIT_COUNTS = {"train": 617, "val": 39, "test": 71}
EXPECTED_DIAGNOSES = {"CN", "AD"}
EXPECTED_LABEL_AD = {0, 1}
EXPECTED_LATENT_DIM = 256
EXPECTED_TRAIN_LATENT_COUNT = 617
EXPECTED_TIME_KEY = "continuous_age_norm"
EXPECTED_SUBJECT_KEY = "subject_id"
EXPECTED_LABEL_KEY = "label_ad"


def parse_args():
    experiment_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate ADNI No-MCI longitudinal metadata, bridge, and specs before training."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=experiment_dir,
        help="Longitudinal experiment directory.",
    )
    return parser.parse_args()


def ensure_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def read_json(path):
    with ensure_file(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path):
    with ensure_file(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def resolve_path(path_str, base_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (Path.cwd() / path).resolve()


def normalize_state_dict_for_model(model, state_dict):
    model_keys = model.state_dict().keys()
    has_module_model = any(key.startswith("module.") for key in model_keys)
    has_module_state = any(key.startswith("module.") for key in state_dict.keys())
    if has_module_model and not has_module_state:
        return {f"module.{key}": value for key, value in state_dict.items()}
    if not has_module_model and has_module_state:
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def add_check(results, name, ok, details):
    results[name] = {
        "status": "pass" if ok else "fail",
        "details": details,
    }


def load_pair_triplet_ids(rows, kind):
    if kind == "pairs":
        fields = ("source_scan_id", "target_scan_id")
    elif kind == "triplets":
        fields = ("scan_s", "scan_r", "scan_t")
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    ids = []
    for row in rows:
        ids.extend(row[field] for field in fields)
    return ids


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    metadata_dir = experiment_dir / "metadata"
    source_manifest_dir = experiment_dir.parent / "brainode_comparison_task1_manifest_original"
    source_task2_dir = (
        experiment_dir.parent
        / "brainode_comparison_task2_representations_original"
        / "inr"
        / "deepsdf_eikonal_spec_fast"
    )
    bridge_dir = experiment_dir / "pretrained_task2_deepsdf"

    specs_path = ensure_file(experiment_dir / "specs.json")
    records_csv_path = ensure_file(metadata_dir / "adni_no_mci_longitudinal_records.csv")
    labels_pt_path = ensure_file(metadata_dir / "adni_no_mci_longitudinal_labels.pt")
    prep_report_path = ensure_file(metadata_dir / "preparation_report.json")
    output_report_path = metadata_dir / "input_validation_report.json"

    source_clean_csv_path = ensure_file(
        source_manifest_dir / "metadata" / "adni_no_mci_left_original_clean.csv"
    )
    split_paths = {
        "train": ensure_file(source_manifest_dir / "splits" / "train_clean.json"),
        "val": ensure_file(source_manifest_dir / "splits" / "val_clean.json"),
        "test": ensure_file(source_manifest_dir / "splits" / "test_clean.json"),
    }
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
    task2_run_config_path = ensure_file(source_task2_dir / "run_config.json")
    task2_train_latents_path = ensure_file(source_task2_dir / "latents" / "train_latents.npz")
    task2_per_scan_dir = source_task2_dir / "latents" / "per_scan"
    task2_checkpoint_path = ensure_file(source_task2_dir / "checkpoints" / "best.pth")
    bridge_specs_path = ensure_file(bridge_dir / "specs.json")
    bridge_model_best_path = ensure_file(bridge_dir / "ModelParameters" / "best.pth")
    bridge_latent_best_path = ensure_file(bridge_dir / "LatentCodes" / "best.pth")
    bridge_train_split_path = ensure_file(bridge_dir / "train_split.json")

    results = {}
    report = {
        "experiment_dir": str(experiment_dir),
        "status": "fail",
        "checks": results,
    }

    try:
        specs = read_json(specs_path)
        bridge_specs = read_json(bridge_specs_path)
        prep_report = read_json(prep_report_path)
        source_rows, _ = read_csv_rows(source_clean_csv_path)
        record_rows, record_fields = read_csv_rows(records_csv_path)
        labels_obj = torch.load(labels_pt_path, map_location="cpu")
        if not isinstance(labels_obj, dict) or "records" not in labels_obj:
            raise RuntimeError(f"Labels file must be a dict with a 'records' key: {labels_pt_path}")
        label_rows = labels_obj["records"]
        if not isinstance(label_rows, list):
            raise RuntimeError(f"Labels 'records' must be a list: {labels_pt_path}")

        train_split = read_json(split_paths["train"])
        val_split = read_json(split_paths["val"])
        test_split = read_json(split_paths["test"])
        bridge_train_split = read_json(bridge_train_split_path)

        split_sets = {
            "train": set(train_split),
            "val": set(val_split),
            "test": set(test_split),
        }

        record_scan_ids = [row["scan_id"] for row in record_rows]
        label_scan_ids = [row["scan_id"] for row in label_rows]
        source_scan_ids = [row["scan_id"] for row in source_rows]

        record_subject_ids = [row["subject_id"] for row in record_rows]
        record_split_counts = Counter(row["split"] for row in record_rows)
        record_diagnoses = {row["diagnosis"] for row in record_rows}
        record_label_ad = {int(row["label_ad"]) for row in record_rows}

        add_check(
            results,
            "metadata_record_schema",
            record_fields == [
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
            ],
            {"fields": record_fields},
        )
        add_check(
            results,
            "counts_and_splits",
            len(record_rows) == EXPECTED_SCAN_COUNT
            and len(set(record_subject_ids)) == EXPECTED_SUBJECT_COUNT
            and dict(record_split_counts) == EXPECTED_SPLIT_COUNTS,
            {
                "num_records": len(record_rows),
                "num_subjects": len(set(record_subject_ids)),
                "split_counts": dict(record_split_counts),
            },
        )
        add_check(
            results,
            "diagnosis_values",
            record_diagnoses == EXPECTED_DIAGNOSES and "MCI" not in record_diagnoses and record_label_ad == EXPECTED_LABEL_AD,
            {
                "diagnoses": sorted(record_diagnoses),
                "label_ad_values": sorted(record_label_ad),
            },
        )
        add_check(
            results,
            "metadata_alignment_with_source_and_labels",
            record_scan_ids == label_scan_ids and set(record_scan_ids) == set(source_scan_ids),
            {
                "records_count": len(record_scan_ids),
                "labels_count": len(label_scan_ids),
                "source_count": len(source_scan_ids),
                "same_order_records_vs_labels": record_scan_ids == label_scan_ids,
            },
        )

        missing_mesh = []
        missing_sdf = []
        missing_repr = []
        bad_repr_shape = []
        for row in record_rows:
            scan_id = row["scan_id"]
            mesh_path = Path(row["mesh_path"])
            sdf_path = Path(row["sdf_npz_path"])
            repr_path = task2_per_scan_dir / f"{scan_id}.npy"
            if not mesh_path.is_file():
                missing_mesh.append(scan_id)
            if not sdf_path.is_file():
                missing_sdf.append(scan_id)
            if not repr_path.is_file():
                missing_repr.append(scan_id)
            else:
                arr = np.load(repr_path)
                if arr.ndim != 1 or arr.shape[0] != EXPECTED_LATENT_DIM:
                    bad_repr_shape.append(
                        {"scan_id": scan_id, "shape": tuple(arr.shape)}
                    )

        add_check(
            results,
            "scan_assets_and_task2_representations",
            not missing_mesh and not missing_sdf and not missing_repr and not bad_repr_shape,
            {
                "missing_mesh_count": len(missing_mesh),
                "missing_sdf_count": len(missing_sdf),
                "missing_task2_representation_count": len(missing_repr),
                "bad_task2_representation_shape_count": len(bad_repr_shape),
                "missing_mesh_examples": missing_mesh[:5],
                "missing_sdf_examples": missing_sdf[:5],
                "missing_task2_representation_examples": missing_repr[:5],
                "bad_task2_representation_shape_examples": bad_repr_shape[:5],
            },
        )

        train_latents_npz = np.load(task2_train_latents_path)
        train_latent_scan_ids = [str(scan_id) for scan_id in train_latents_npz["scan_ids"].tolist()]
        train_latents = train_latents_npz["latents"]
        add_check(
            results,
            "training_latent_table",
            train_latents.shape == (EXPECTED_TRAIN_LATENT_COUNT, EXPECTED_LATENT_DIM)
            and train_latent_scan_ids == [Path(item).stem for item in train_split]
            and bridge_train_split == train_split,
            {
                "latents_shape": list(train_latents.shape),
                "train_split_count": len(train_split),
                "bridge_train_split_count": len(bridge_train_split),
                "same_order_as_train_split": train_latent_scan_ids == [Path(item).stem for item in train_split],
                "bridge_split_matches_source_train_split": bridge_train_split == train_split,
            },
        )

        subject_to_splits = defaultdict(set)
        subject_to_rows = defaultdict(list)
        for row in record_rows:
            subject_to_splits[row["subject_id"]].add(row["split"])
            subject_to_rows[row["subject_id"]].append(row)

        cross_split_subjects = {
            subject: sorted(list(splits))
            for subject, splits in subject_to_splits.items()
            if len(splits) != 1
        }
        add_check(
            results,
            "subject_split_isolation",
            not cross_split_subjects,
            {
                "cross_split_subject_count": len(cross_split_subjects),
                "examples": dict(list(cross_split_subjects.items())[:5]),
            },
        )

        bad_time_subjects = []
        for subject_id, rows in subject_to_rows.items():
            ordered = sorted(rows, key=lambda row: int(row["visit_order"]))
            times = [float(row["continuous_age_norm"]) for row in ordered]
            if any(curr <= prev for prev, curr in zip(times, times[1:])):
                bad_time_subjects.append(
                    {
                        "subject_id": subject_id,
                        "visit_order": [int(row["visit_order"]) for row in ordered],
                        "times": times,
                    }
                )
        add_check(
            results,
            "strictly_increasing_subject_time",
            not bad_time_subjects,
            {
                "bad_subject_count": len(bad_time_subjects),
                "examples": bad_time_subjects[:5],
            },
        )

        pair_rows = {}
        triplet_rows = {}
        for split_name, path in pair_paths.items():
            rows, _ = read_csv_rows(path)
            pair_rows[split_name] = rows
        for split_name, path in triplet_paths.items():
            rows, _ = read_csv_rows(path)
            triplet_rows[split_name] = rows

        metadata_by_scan = {row["scan_id"]: row for row in record_rows}
        bad_pair_rows = []
        for split_name, rows in pair_rows.items():
            for idx, row in enumerate(rows):
                for field in ("source_scan_id", "target_scan_id"):
                    scan_id = row[field]
                    meta = metadata_by_scan.get(scan_id)
                    if meta is None:
                        bad_pair_rows.append(
                            {"split": split_name, "row_index": idx, "reason": f"missing {field}", "scan_id": scan_id}
                        )
                        continue
                    if split_name != "all" and meta["split"] != split_name:
                        bad_pair_rows.append(
                            {
                                "split": split_name,
                                "row_index": idx,
                                "reason": f"{field} split mismatch",
                                "scan_id": scan_id,
                                "metadata_split": meta["split"],
                            }
                        )
        bad_triplet_rows = []
        for split_name, rows in triplet_rows.items():
            for idx, row in enumerate(rows):
                for field in ("scan_s", "scan_r", "scan_t"):
                    scan_id = row[field]
                    meta = metadata_by_scan.get(scan_id)
                    if meta is None:
                        bad_triplet_rows.append(
                            {"split": split_name, "row_index": idx, "reason": f"missing {field}", "scan_id": scan_id}
                        )
                        continue
                    if split_name != "all" and meta["split"] != split_name:
                        bad_triplet_rows.append(
                            {
                                "split": split_name,
                                "row_index": idx,
                                "reason": f"{field} split mismatch",
                                "scan_id": scan_id,
                                "metadata_split": meta["split"],
                            }
                        )
        add_check(
            results,
            "pair_triplet_scan_id_alignment",
            not bad_pair_rows and not bad_triplet_rows,
            {
                "pair_row_count": {key: len(value) for key, value in pair_rows.items()},
                "triplet_row_count": {key: len(value) for key, value in triplet_rows.items()},
                "bad_pair_row_count": len(bad_pair_rows),
                "bad_triplet_row_count": len(bad_triplet_rows),
                "bad_pair_examples": bad_pair_rows[:5],
                "bad_triplet_examples": bad_triplet_rows[:5],
            },
        )

        task2_run_config = read_json(task2_run_config_path)
        add_check(
            results,
            "specs_core_paths_and_keys",
            resolve_path(specs["TrainSplit"], experiment_dir) == split_paths["train"].resolve()
            and resolve_path(specs["TestSplit"], experiment_dir) == split_paths["val"].resolve()
            and resolve_path(specs["FinalTestSplit"], experiment_dir) == split_paths["test"].resolve()
            and resolve_path(specs["LongitudinalMetadataFile"], experiment_dir) == labels_pt_path.resolve()
            and resolve_path(specs["AgeMetadataFile"], experiment_dir) == labels_pt_path.resolve()
            and resolve_path(specs["LongitudinalTimeMetadataFile"], experiment_dir) == labels_pt_path.resolve()
            and specs["LongitudinalSubjectKey"] == EXPECTED_SUBJECT_KEY
            and specs["LongitudinalTimeKey"] == EXPECTED_TIME_KEY
            and specs["AgeConditionKey"] == EXPECTED_LABEL_KEY
            and specs["AgeConditionKeys"] == [EXPECTED_LABEL_KEY]
            and specs["PretrainedSubjectAnchorExpectedScansPerSubject"] is None,
            {
                "TrainSplit": specs.get("TrainSplit"),
                "TestSplit": specs.get("TestSplit"),
                "FinalTestSplit": specs.get("FinalTestSplit"),
                "LongitudinalMetadataFile": specs.get("LongitudinalMetadataFile"),
                "AgeMetadataFile": specs.get("AgeMetadataFile"),
                "LongitudinalTimeMetadataFile": specs.get("LongitudinalTimeMetadataFile"),
                "LongitudinalSubjectKey": specs.get("LongitudinalSubjectKey"),
                "LongitudinalTimeKey": specs.get("LongitudinalTimeKey"),
                "AgeConditionKey": specs.get("AgeConditionKey"),
                "AgeConditionKeys": specs.get("AgeConditionKeys"),
                "PretrainedSubjectAnchorExpectedScansPerSubject": specs.get(
                    "PretrainedSubjectAnchorExpectedScansPerSubject"
                ),
            },
        )
        add_check(
            results,
            "decoder_architecture_match",
            specs["NetworkArch"] == bridge_specs["NetworkArch"] == task2_run_config["network_arch"]
            and int(specs["CodeLength"]) == int(bridge_specs["CodeLength"]) == int(task2_run_config["latent_size"]) == EXPECTED_LATENT_DIM
            and specs["NetworkSpecs"] == bridge_specs["NetworkSpecs"] == task2_run_config["network_specs"],
            {
                "longitudinal_network_arch": specs.get("NetworkArch"),
                "bridge_network_arch": bridge_specs.get("NetworkArch"),
                "task2_network_arch": task2_run_config.get("network_arch"),
                "longitudinal_code_length": specs.get("CodeLength"),
                "bridge_code_length": bridge_specs.get("CodeLength"),
                "task2_latent_size": task2_run_config.get("latent_size"),
            },
        )

        bridge_checkpoint = torch.load(bridge_model_best_path, map_location="cpu")
        bridge_decoder = Decoder(
            latent_size=int(bridge_specs["CodeLength"]),
            **bridge_specs["NetworkSpecs"],
        )
        bridge_state_dict = normalize_state_dict_for_model(
            bridge_decoder, bridge_checkpoint["model_state_dict"]
        )
        bridge_decoder.load_state_dict(bridge_state_dict, strict=True)

        longitudinal_decoder = Decoder(
            latent_size=int(specs["CodeLength"]),
            **specs["NetworkSpecs"],
        )
        longitudinal_state_dict = normalize_state_dict_for_model(
            longitudinal_decoder, bridge_checkpoint["model_state_dict"]
        )
        longitudinal_decoder.load_state_dict(longitudinal_state_dict, strict=True)

        bridge_latent_payload = torch.load(bridge_latent_best_path, map_location="cpu")
        bridge_latent_weight = bridge_latent_payload["latent_codes"]["weight"]
        add_check(
            results,
            "bridge_decoder_and_latent_bridge_load",
            bridge_latent_weight.shape == (EXPECTED_TRAIN_LATENT_COUNT, EXPECTED_LATENT_DIM),
            {
                "bridge_checkpoint_epoch": int(bridge_checkpoint.get("epoch", -1)),
                "bridge_latent_shape": list(bridge_latent_weight.shape),
                "bridge_model_best_path": str(bridge_model_best_path),
                "bridge_latent_best_path": str(bridge_latent_best_path),
            },
        )

        add_check(
            results,
            "preparation_report_status",
            prep_report.get("task2_bridge", {}).get("latent_order_matches_task1_train_split") is True,
            {
                "preparation_report_path": str(prep_report_path),
                "latent_order_matches_task1_train_split": prep_report.get("task2_bridge", {}).get(
                    "latent_order_matches_task1_train_split"
                ),
            },
        )

        overall_ok = all(item["status"] == "pass" for item in results.values())
        report["status"] = "pass" if overall_ok else "fail"
        report["summary"] = {
            "expected_scans": EXPECTED_SCAN_COUNT,
            "expected_subjects": EXPECTED_SUBJECT_COUNT,
            "expected_split_counts": EXPECTED_SPLIT_COUNTS,
            "expected_diagnoses": sorted(EXPECTED_DIAGNOSES),
            "expected_latent_dim": EXPECTED_LATENT_DIM,
        }
    except Exception as exc:
        add_check(
            results,
            "validator_runtime",
            False,
            {"error": str(exc)},
        )
        report["status"] = "fail"

    output_report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote validation report: {output_report_path}")
    print(f"Validation status: {report['status']}")
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
