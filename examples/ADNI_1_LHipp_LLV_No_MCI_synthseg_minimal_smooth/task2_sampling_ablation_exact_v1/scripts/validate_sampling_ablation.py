#!/usr/bin/env python3
"""Static, data, sampling, and checkpoint validation for the ablation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPTS = SCRIPT_DIR.parent.parent / "task2_inr_multires_single_field_v1" / "scripts"
for path in (SCRIPT_DIR, BASE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablation_common import (  # noqa: E402
    BASE_SCRIPTS,
    BULK_WRAPPER,
    EVALUATE_SCRIPT,
    EXPORT_SCRIPT,
    MATRIX_PATH,
    TRAIN_SCRIPT,
    all_persistent_paths,
    experiment_definition,
    load_json,
    load_matrix,
    materialize_config,
    read_manifest,
    require_bulk_path,
    resolve_repo_path,
    runtime_config_path,
    sha256_file,
)
from multires_common import load_sdf_arrays  # noqa: E402
from sampling_dataset import shell_eligibility_report, validate_shell_sampling  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--sampling-scans", type=int, default=3)
    return parser.parse_args()


def _canonical_non_sampling(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    for key in ("name", "description", "output_dir", "sampling", "sampling_ablation"):
        value.pop(key, None)
    return value


def _sampling_count(config: dict[str, Any], group: str) -> int:
    keys = (
        ("global_near_samples_per_scene", "global_positive_samples_per_scene", "global_negative_samples_per_scene")
        if group == "global"
        else ("local_ultra_near_samples_per_scene", "local_positive_samples_per_scene", "local_negative_samples_per_scene")
    )
    return sum(int(config["sampling"][key]) for key in keys)


def _band_eligibility(pos: np.ndarray, neg: np.ndarray, config: dict[str, Any]) -> dict[str, int]:
    sampling = config["sampling"]
    ultra = float(sampling["ultra_near_band"])
    near = float(sampling["near_band"])
    return {
        "positive_within_ultra": int(np.sum(np.abs(pos[:, 3]) <= ultra)),
        "negative_within_ultra": int(np.sum(np.abs(neg[:, 3]) <= ultra)),
        "positive_within_near": int(np.sum(np.abs(pos[:, 3]) <= near)),
        "negative_within_near": int(np.sum(np.abs(neg[:, 3]) <= near)),
    }


def validate(require_data: bool, sampling_scans: int) -> dict[str, Any]:
    if sampling_scans < 1:
        raise ValueError("--sampling-scans must be positive.")
    matrix = load_matrix()
    base_path = resolve_repo_path(matrix["base_config"])
    if sha256_file(base_path) != matrix["base_config_sha256"]:
        raise RuntimeError("Pinned base configuration checksum changed.")
    required_scripts = (
        TRAIN_SCRIPT,
        EVALUATE_SCRIPT,
        EXPORT_SCRIPT,
        BULK_WRAPPER,
        SCRIPT_DIR / "prepare_population_initialization.py",
        SCRIPT_DIR / "sampling_dataset.py",
        SCRIPT_DIR / "compare_sampling_runs.py",
        SCRIPT_DIR / "analyze_latent_drift.py",
        SCRIPT_DIR / "summarize_ceiling_pilot.py",
        SCRIPT_DIR / "validate_latent_export.py",
    )
    missing_scripts = [str(path) for path in required_scripts if not path.is_file()]
    if missing_scripts:
        raise FileNotFoundError(f"Missing ablation scripts: {missing_scripts}")
    source_checkpoint = require_bulk_path(matrix["source_checkpoint"], "source checkpoint")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    source_hash = sha256_file(source_checkpoint)
    if source_hash != matrix["source_checkpoint_sha256"]:
        raise RuntimeError("Pinned source checkpoint checksum changed.")

    payload = torch.load(source_checkpoint, map_location="cpu")
    if payload.get("architecture") != "single_field_dense_multiresolution_sdf":
        raise ValueError("Source checkpoint architecture is not the expected multiresolution field.")
    if int(payload.get("epoch", -1)) != int(matrix["source_checkpoint_epoch"]):
        raise ValueError("Source checkpoint epoch differs from the matrix.")
    latent_table = payload.get("latent_codes")
    if not isinstance(latent_table, torch.Tensor) or tuple(latent_table.shape) != (401, 256):
        raise ValueError("Source checkpoint must contain a 401x256 training latent table.")
    if not torch.isfinite(latent_table).all():
        raise ValueError("Source latent table contains non-finite values.")

    configs = {
        name: materialize_config(matrix, name) for name in matrix["experiments"]
    }
    canonical = _canonical_non_sampling(next(iter(configs.values())))
    for name, config in configs.items():
        experiment_definition(matrix, name)
        if _canonical_non_sampling(config) != canonical:
            raise ValueError(f"{name} differs from the other arms outside training sampling metadata.")
        all_persistent_paths(config)
        if int(config["latent_size"]) != 256 or int(config["total_epochs"]) != 300:
            raise ValueError(f"Unexpected latent size or epoch count in {name}.")
        if config["periodic_evaluation"]["splits"] != ["val"]:
            raise ValueError(f"Test leakage in periodic evaluation for {name}.")
        if config["latent_fit"]["sampling"].get("mode") == "shell_stratified":
            raise ValueError(f"Inference sampling accidentally follows the training arm in {name}.")
        if (
            float(config["latent_fit"]["sampling"]["ultra_near_band"]) != 0.03
            or float(config["latent_fit"]["sampling"]["near_band"]) != 0.1
        ):
            raise ValueError(f"Inference sampling is not the fixed control distribution in {name}.")
        if any(
            int(entry["start_epoch"]) != 1 or int(entry["end_epoch"]) != 1
            for entry in config["level_schedule"]
        ):
            raise ValueError(f"All pretrained levels must be active from epoch 1 in {name}.")
        eikonal = config["eikonal"]
        if not eikonal["enabled"] or float(eikonal["weight"]) != 0.01:
            raise ValueError(f"Light Eikonal is not fixed in {name}.")
        if float(eikonal["target_band"]) != 0.03:
            raise ValueError(f"Eikonal target band changed in {name}.")
        if float(eikonal["epsilon_cell_scale"]) != 0.5:
            raise ValueError(f"Finite-difference scale changed in {name}.")
        if bool(config.get("second_order", {}).get("enabled", False)):
            raise ValueError(f"Second-order regularization must remain disabled in {name}.")
        if _sampling_count(config, "global") != 8192 or _sampling_count(config, "local") != 8192:
            raise ValueError(f"{name} changed the per-shape training sample totals.")
        validate_shell_sampling(config["sampling"])

    fixed_selection = load_json(matrix["fixed_validation_selection"])
    selected_val = fixed_selection.get("scan_ids_by_split", {}).get("val", [])
    if len(selected_val) != 100 or len(set(selected_val)) != 100:
        raise ValueError("Fixed validation selection must contain 100 unique scans.")

    result: dict[str, Any] = {
        "passed": True,
        "matrix": str(MATRIX_PATH),
        "experiments": sorted(configs),
        "runtime_configs": {name: str(runtime_config_path(matrix, name)) for name in configs},
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_hash,
        "source_epoch": int(payload["epoch"]),
        "source_model_tensors": len(payload["model_state_dict"]),
        "source_latent_shape": list(latent_table.shape),
        "source_optimizer_will_be_loaded": False,
        "source_rng_will_be_loaded": False,
        "fixed_validation_scan_count": len(selected_val),
        "periodic_selection_splits": ["val"],
        "test_locked_during_selection": True,
        "global_samples_per_shape": 8192,
        "local_samples_per_shape": 8192,
        "persistent_output_policy": "runtime configs, checkpoints, logs, meshes, metrics, caches, and temporary files are restricted to /mnt/bulk10tb",
    }
    if not require_data:
        result["data"] = "not requested"
        return result

    exact_audit = load_json(matrix["exact_sdf_audit"])
    if not exact_audit.get("passed") or not exact_audit.get("source_coordinates_bitwise_preserved"):
        raise RuntimeError("The exact-triangle SDF audit did not pass.")
    rows = read_manifest(configs["control_u030_n100"]["manifest"])
    split_counts = {
        split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")
    }
    if split_counts != {"train": 401, "val": 100, "test": 100}:
        raise ValueError(f"Unexpected exact manifest split counts: {split_counts}")
    ids = [row["scan_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Exact manifest scan IDs are not unique.")
    train_ids = [row["scan_id"] for row in rows if row["split"] == "train"]
    if train_ids != list(payload["training_scan_ids"]):
        raise ValueError("Manifest training order differs from the pretrained latent order.")
    val_ids = {row["scan_id"] for row in rows if row["split"] == "val"}
    if set(selected_val) != val_ids:
        raise ValueError("Fixed validation selection is not exactly the 100-scan validation split.")
    for row in rows:
        if not Path(row["mesh_path"]).is_file() or not Path(row["sdf_npz_path"]).is_file():
            raise FileNotFoundError(f"Missing mesh or exact SDF for {row['scan_id']}.")
        require_bulk_path(row["sdf_npz_path"], "exact SDF")

    audited_rows = [row for row in rows if row["split"] == "train"][:sampling_scans]
    sampling_reports: dict[str, list[dict[str, Any]]] = {name: [] for name in configs}
    for row in audited_rows:
        pos, neg = load_sdf_arrays(row["sdf_npz_path"])
        for name, config in configs.items():
            if config["sampling"]["mode"] == "shell_stratified":
                report = shell_eligibility_report(pos, neg, config["sampling"])
                for group in ("global", "local"):
                    for shell in report[group]:
                        if shell["eligible_positive"] < shell["requested_positive"]:
                            raise ValueError(f"Insufficient positive candidates for {name} {row['scan_id']}.")
                        if shell["eligible_negative"] < shell["requested_negative"]:
                            raise ValueError(f"Insufficient negative candidates for {name} {row['scan_id']}.")
            else:
                report = _band_eligibility(pos, neg, config)
                if min(report.values()) <= 0:
                    raise ValueError(f"Empty requested band for {name} {row['scan_id']}.")
            sampling_reports[name].append({"scan_id": row["scan_id"], **report})
    result["data"] = {
        "exact_audit": matrix["exact_sdf_audit"],
        "exact_audit_passed": True,
        "scan_count": len(rows),
        "split_counts": split_counts,
        "training_order_matches_pretrained_latents": True,
        "sampling_scans_audited": len(audited_rows),
        "sampling_reports": sampling_reports,
    }
    return result


def main() -> None:
    args = parse_args()
    print(json.dumps(validate(args.require_data, args.sampling_scans), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
