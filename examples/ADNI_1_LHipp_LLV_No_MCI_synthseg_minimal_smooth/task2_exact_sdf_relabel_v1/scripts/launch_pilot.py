#!/usr/bin/env python3
"""Validate, materialize, train, or evaluate the paired exact-SDF pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from pipeline_common import (
    REPO_ROOT,
    SCRIPT_DIR,
    atomic_write_json,
    read_manifest,
    require_bulk_path,
    resolve_path,
    split_subject_leakage,
)


MATRIX_PATH = SCRIPT_DIR.parent / "configs" / "pilot_experiment_matrix.json"
PYTHON = Path("/home/jakaria/anaconda3/envs/inr_sdf/bin/python")
BULK_WRAPPER = (
    SCRIPT_DIR / "run_on_bulk.sh"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Validate pinned configs and optionally relabelled data.")
    check.add_argument("--require-data", action="store_true")

    show = subparsers.add_parser("show", help="Print a fully materialized runtime config.")
    show.add_argument("--experiment", required=True)

    train = subparsers.add_parser("train", help="Print a training command; --execute runs it.")
    train.add_argument("--experiment", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", default=None)
    train.add_argument("--skip-periodic-evaluation", action="store_true")
    train.add_argument("--execute", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Print an evaluation command; --execute runs it.")
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument("--checkpoint", default="best_mesh")
    evaluate.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["val"])
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--per-split", type=int, default=100)
    evaluate.add_argument("--resolution", type=int, default=256)
    evaluate.add_argument("--latent-steps", type=int, default=500)
    evaluate.add_argument("--surface-points", type=int, default=30_000)
    evaluate.add_argument("--overwrite-meshes", action="store_true")
    evaluate.add_argument("--confirm-test", action="store_true")
    evaluate.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_matrix() -> dict[str, Any]:
    with MATRIX_PATH.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    require_bulk_path(matrix["bulk_output_root"], "matrix bulk_output_root")
    return matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_experiment_name(matrix: dict[str, Any], name: str) -> dict[str, str]:
    if name not in matrix["experiments"]:
        raise KeyError(
            f"Unknown experiment {name!r}; choose one of {sorted(matrix['experiments'])}."
        )
    return matrix["experiments"][name]


def materialize_config(matrix: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    experiment = validate_experiment_name(matrix, experiment_name)
    family_name = experiment["family"]
    labels = experiment["labels"]
    family = matrix["families"][family_name]
    base_path = resolve_path(family["base_config"])
    actual_hash = sha256(base_path)
    if actual_hash != family["base_config_sha256"]:
        raise RuntimeError(
            f"Pinned {family_name} base config changed: expected "
            f"{family['base_config_sha256']}, found {actual_hash}. Audit before updating the pin."
        )
    with base_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    policy = matrix["training_policy"]
    warm_policy = policy["decoder_warm_start"]
    root = require_bulk_path(matrix["bulk_output_root"])
    config["name"] = f"hippocampus_exact_sdf_pilot_{experiment_name}"
    config["description"] = (
        f"Controlled {family_name} pilot trained with {labels} SDF labels; "
        "256-D latent, unchanged architecture/sampling, Eikonal disabled."
    )
    config["run_role"] = f"exact_sdf_pilot_{labels}_paired_control"
    config["manifest"] = matrix["manifests"][labels]
    config["output_dir"] = str(root / "runs" / experiment_name)
    config["require_bulk_output"] = True
    config["seed"] = int(policy["seed"])
    config["latent_size"] = int(policy["latent_size"])
    config["checkpoint_every_epochs"] = int(policy["checkpoint_every_epochs"])
    config["periodic_evaluation"]["enabled"] = True
    config["periodic_evaluation"]["every_epochs"] = int(
        policy["periodic_evaluation_every_epochs"]
    )
    config["periodic_evaluation"]["also_final_epoch"] = True
    config["periodic_evaluation"]["splits"] = list(policy["periodic_splits"])
    config["periodic_evaluation"]["per_split"] = int(policy["periodic_per_split"])
    config["periodic_evaluation"]["source_unseen_validation_only"] = True
    config["periodic_evaluation"].pop("fixed_selection_json", None)
    config["sdf_supervision"] = {
        "label_kind": (
            "exact_point_to_triangle_signed_distance"
            if labels == "exact"
            else "original_approximate_preprocessmesh"
        ),
        "exact_triangle_distance": labels == "exact",
        "sign_convention": "negative_inside_positive_outside",
        "query_coordinates": "identical_between_paired_manifests",
        "eikonal_enabled": False,
    }
    checkpoint_key = f"{family_name}_checkpoint"
    checkpoint_sha_key = f"{family_name}_checkpoint_sha256"
    checkpoint = warm_policy[checkpoint_key]
    config["decoder_warm_start"] = {
        "enabled": True,
        "mode": str(warm_policy["mode"]),
        "checkpoint": checkpoint,
        "checkpoint_sha256": str(warm_policy[checkpoint_sha_key]),
        "latent_adapt_epochs": int(warm_policy["latent_adapt_epochs"]),
        "copy_source_latents": False,
        "load_source_optimizer": False,
        "load_source_rng": False,
    }
    if family_name == "compact":
        config.pop("global_checkpoint", None)
        config.pop("global_checkpoint_scan_id_suffix", None)
        config["reuse_matching_source_latents"] = False
        config["schedule"].update(
            {
                "latent_adapt_epochs": int(warm_policy["latent_adapt_epochs"]),
                "latent_adapt_fusion_alpha": 1.0,
                "global_adapt_epochs": 0,
                "local_warmup_epochs": 0,
                "joint_fusion_ramp_epochs": 1,
                "total_epochs": int(policy["total_epochs"]),
            }
        )
        config["stage_learning_rates"]["latent_adapt"] = dict(
            warm_policy["compact_latent_adapt_learning_rates"]
        )
        config["stage_learning_rates"]["joint"] = dict(
            warm_policy["compact_joint_learning_rates"]
        )
        joint = config["schedule"]["total_epochs"] - config["schedule"]["latent_adapt_epochs"]
        if joint != int(warm_policy["joint_epochs"]):
            raise ValueError(f"Compact schedule has {joint} joint epochs, not the requested value.")
    else:
        config["total_epochs"] = int(policy["total_epochs"])
        config["eikonal"]["enabled"] = False
        config["eikonal"]["weight"] = 0.0
        config["learning_rates"] = dict(warm_policy["multires_joint_learning_rates"])
        config["decoder_warm_start"]["latent_adapt_learning_rates"] = dict(
            warm_policy["multires_latent_adapt_learning_rates"]
        )
        config["level_schedule"] = [
            {"resolution": int(resolution), "start_epoch": 1, "end_epoch": 1}
            for resolution in config["network_specs"]["grid_resolutions"]
        ]
    return config


def runtime_config_path(matrix: dict[str, Any], experiment_name: str) -> Path:
    root = require_bulk_path(matrix["bulk_output_root"])
    return root / "runtime_configs" / f"{experiment_name}.json"


def write_runtime_config(
    matrix: dict[str, Any], experiment_name: str, config: dict[str, Any]
) -> Path:
    path = runtime_config_path(matrix, experiment_name)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != config:
            raise FileExistsError(
                f"Runtime config differs from the currently materialized definition: {path}. "
                "Move the old pilot root or explicitly reconcile it; refusing silent overwrite."
            )
        return path
    return atomic_write_json(path, config)


def paired_config_signature(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    for field in ("name", "description", "run_role", "manifest", "output_dir", "sdf_supervision"):
        value.pop(field, None)
    return value


def check_data(matrix: dict[str, Any]) -> dict[str, Any]:
    manifests = {label: read_manifest(path) for label, path in matrix["manifests"].items()}
    by_label = {
        label: {row["scan_id"]: row for row in rows}
        for label, rows in manifests.items()
    }
    if set(by_label["approx"]) != set(by_label["exact"]):
        raise ValueError("Approximate and exact manifests do not have identical scan IDs.")
    for label, rows in manifests.items():
        leakage = split_subject_leakage(rows)
        if leakage:
            raise ValueError(f"{label} manifest has subject leakage.")
        for row in rows:
            if not Path(row["mesh_path"]).is_file() or not Path(row["sdf_npz_path"]).is_file():
                raise FileNotFoundError(f"Missing {label} input for {row['scan_id']}")
            require_bulk_path(row["sdf_npz_path"], f"{label} SDF input")
    for scan_id, approximate in by_label["approx"].items():
        exact = by_label["exact"][scan_id]
        for key in ("subject_id", "split", "mesh_path"):
            if approximate[key] != exact[key]:
                raise ValueError(f"Paired data differ for {scan_id}: {key}")
    audit_path = require_bulk_path(matrix["bulk_output_root"]) / "audits" / "exact_sdf_audit_summary.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Run the exact-SDF audit first: {audit_path}")
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if not audit.get("passed") or not audit.get("source_coordinates_bitwise_preserved"):
        raise RuntimeError("Exact-SDF audit did not pass coordinate and distance checks.")
    if Path(audit.get("approximate_manifest", "")).resolve() != Path(
        matrix["manifests"]["approx"]
    ).resolve() or Path(audit.get("exact_manifest", "")).resolve() != Path(
        matrix["manifests"]["exact"]
    ).resolve():
        raise RuntimeError("Exact-SDF audit refers to different manifests and is stale.")
    if int(audit.get("scan_count", -1)) != len(by_label["exact"]):
        raise RuntimeError("Exact-SDF audit scan count does not match the current manifest.")
    return {
        "scan_count": len(by_label["exact"]),
        "subject_count": len({row["subject_id"] for row in manifests["exact"]}),
        "split_scan_counts": {
            split: sum(row["split"] == split for row in manifests["exact"])
            for split in ("train", "val", "test")
        },
        "exact_audit": str(audit_path),
    }


def run_check(matrix: dict[str, Any], require_data: bool) -> dict[str, Any]:
    configs = {
        name: materialize_config(matrix, name) for name in matrix["experiments"]
    }
    for family_name, family in matrix["families"].items():
        for key in ("base_config", "train_script", "evaluate_script"):
            if not resolve_path(family[key]).is_file():
                raise FileNotFoundError(f"Missing {family_name} {key}: {family[key]}")
    for name, config in configs.items():
        output = require_bulk_path(config["output_dir"], f"{name} output_dir")
        if output == require_bulk_path(matrix["bulk_output_root"]):
            raise ValueError("An experiment output cannot equal the pilot root.")
        periodic = config["periodic_evaluation"]
        if "test" in periodic["splits"]:
            raise ValueError(f"Test leakage in periodic model selection for {name}.")
        if not bool(periodic.get("source_unseen_validation_only", False)):
            raise ValueError(f"{name} periodic selection must exclude source-seen validation subjects.")
        if int(config["latent_size"]) != 256 or int(config["seed"]) != 42:
            raise ValueError(f"Unexpected latent size or seed in {name}.")
        warm_start = config.get("decoder_warm_start", {})
        checkpoint = resolve_path(warm_start.get("checkpoint", ""))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing full decoder warm-start for {name}: {checkpoint}")
        expected_checkpoint_hash = str(warm_start.get("checkpoint_sha256", ""))
        actual_checkpoint_hash = sha256(checkpoint)
        if actual_checkpoint_hash != expected_checkpoint_hash:
            raise RuntimeError(
                f"Pinned decoder warm-start changed for {name}: expected "
                f"{expected_checkpoint_hash}, found {actual_checkpoint_hash}."
            )
        if warm_start.get("mode") != "decoder_only":
            raise ValueError(f"{name} must use decoder-only warm-start.")
        if any(
            bool(warm_start.get(key, True))
            for key in ("copy_source_latents", "load_source_optimizer", "load_source_rng")
        ):
            raise ValueError(f"{name} must not import source latents, optimizer, or RNG state.")
        if name.startswith("compact_"):
            if "global_checkpoint" in config or bool(config.get("reuse_matching_source_latents", False)):
                raise ValueError(f"{name} retains a legacy Compact-only warm-start path.")
        if bool(periodic.get("compare_pca", False)):
            if not resolve_path(periodic["pca_model_dir"]).is_dir():
                raise FileNotFoundError(f"Missing PCA model directory for {name}.")
            for key in ("pca_coefficients", "rescale_details_csv"):
                if not resolve_path(periodic[key]).is_file():
                    raise FileNotFoundError(f"Missing {key} for {name}: {periodic[key]}")
    for family in ("compact", "multires"):
        approximate = configs[f"{family}_approx"]
        exact = configs[f"{family}_exact"]
        if paired_config_signature(approximate) != paired_config_signature(exact):
            raise ValueError(f"{family} approximate/exact configs differ beyond labels and output.")
    compact = configs["compact_exact"]
    schedule = compact["schedule"]
    compact_epochs = {
        "latent_adapt": int(schedule.get("latent_adapt_epochs", 0)),
        "global": int(schedule["global_adapt_epochs"]),
        "local": int(schedule["local_warmup_epochs"]),
        "joint": int(schedule["total_epochs"])
        - int(schedule.get("latent_adapt_epochs", 0))
        - int(schedule["global_adapt_epochs"])
        - int(schedule["local_warmup_epochs"]),
    }
    if compact_epochs != {"latent_adapt": 100, "global": 0, "local": 0, "joint": 900}:
        raise ValueError(f"Unexpected Compact schedule: {compact_epochs}")
    if int(configs["multires_exact"]["total_epochs"]) != 1000:
        raise ValueError("Unexpected multires total epoch count.")
    if any(
        int(item["start_epoch"]) != 1 or int(item["end_epoch"]) != 1
        for item in configs["multires_exact"]["level_schedule"]
    ):
        raise ValueError("Warm-started multires decoder must expose every learned grid from epoch 1.")
    result = {
        "passed": True,
        "matrix": str(MATRIX_PATH),
        "experiments": sorted(configs),
        "runtime_configs": {
            name: str(runtime_config_path(matrix, name)) for name in configs
        },
        "persistent_output_policy": "all generated configs, logs, checkpoints, latents, meshes, and reports are below /mnt/bulk10tb",
        "compact_schedule": compact_epochs,
        "multires_total_epochs": 1000,
        "decoder_warm_start": {
            family: {
                "checkpoint": configs[f"{family}_exact"]["decoder_warm_start"]["checkpoint"],
                "latent_adapt_epochs": configs[f"{family}_exact"]["decoder_warm_start"]["latent_adapt_epochs"],
                "source_latents_loaded": False,
                "source_optimizer_loaded": False,
                "source_rng_loaded": False,
            }
            for family in ("compact", "multires")
        },
        "latent_size": 256,
        "eikonal_enabled": False,
        "periodic_splits": ["train", "val"],
        "test_locked_during_selection": True,
        "pca_interpretation": "descriptive full-training PCA reference; not a data-matched pilot baseline",
    }
    result["data"] = check_data(matrix) if require_data else "not requested"
    return result


def command_prefix() -> list[str]:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not BULK_WRAPPER.is_file():
        raise FileNotFoundError(BULK_WRAPPER)
    return ["/bin/bash", str(BULK_WRAPPER)]


def execute_or_print(command: list[str], execute: bool) -> None:
    print(shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    else:
        print("Dry run only. Add --execute to run this command.", flush=True)


def main() -> None:
    args = parse_args()
    matrix = load_matrix()
    if args.command == "check":
        print(json.dumps(run_check(matrix, args.require_data), indent=2, sort_keys=True))
        return
    config = materialize_config(matrix, args.experiment)
    if args.command == "show":
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    experiment = validate_experiment_name(matrix, args.experiment)
    family = matrix["families"][experiment["family"]]
    config_path = runtime_config_path(matrix, args.experiment)
    if args.execute:
        run_check(matrix, require_data=True)
        config_path = write_runtime_config(matrix, args.experiment, config)
    if args.command == "train":
        command = command_prefix() + [
            str(resolve_path(family["train_script"])),
            "--config",
            str(config_path),
            "--device",
            args.device,
        ]
        if args.resume:
            command.extend(("--resume", args.resume))
        if args.skip_periodic_evaluation:
            command.append("--skip-periodic-evaluation")
        execute_or_print(command, args.execute)
        return

    if "test" in args.splits and not args.confirm_test:
        raise PermissionError(
            "Test evaluation is locked during model selection. Add --confirm-test only after "
            "choosing the architecture/checkpoint from validation results."
        )
    split_label = "_".join(args.splits)
    checkpoint_label = Path(args.checkpoint).stem
    if not checkpoint_label or checkpoint_label in {".", ".."}:
        raise ValueError(f"Cannot derive a safe output label from checkpoint {args.checkpoint!r}.")
    output = require_bulk_path(
        require_bulk_path(config["output_dir"])
        / "manual_evaluation"
        / checkpoint_label
        / split_label,
        "manual evaluation output",
    )
    command = command_prefix() + [
        str(resolve_path(family["evaluate_script"])),
        "--config",
        str(config_path),
        "--checkpoint",
        args.checkpoint,
        "--output-dir",
        str(output),
        "--device",
        args.device,
        "--splits",
        *args.splits,
        "--per-split",
        str(args.per_split),
        "--resolution",
        str(args.resolution),
        "--latent-steps",
        str(args.latent_steps),
        "--surface-points",
        str(args.surface_points),
    ]
    if args.overwrite_meshes:
        command.append("--overwrite-meshes")
    execute_or_print(command, args.execute)


if __name__ == "__main__":
    main()
