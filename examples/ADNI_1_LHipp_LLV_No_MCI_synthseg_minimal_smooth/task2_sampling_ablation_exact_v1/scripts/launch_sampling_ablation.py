#!/usr/bin/env python3
"""Validate, prepare, smoke-test, train, evaluate, and export sampling arms."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
from pathlib import Path

from ablation_common import (
    BULK_WRAPPER,
    EVALUATE_SCRIPT,
    EXPORT_SCRIPT,
    REPO_ROOT,
    SCRIPT_DIR,
    TRAIN_SCRIPT,
    initialization_path,
    load_matrix,
    materialize_config,
    require_bulk_path,
    run_dir,
    runtime_config_path,
    write_runtime_config,
)
from validate_sampling_ablation import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run read-only static and optional data checks.")
    check.add_argument("--require-data", action="store_true")
    check.add_argument("--sampling-scans", type=int, default=3)

    show = subparsers.add_parser("show", help="Print one fully materialized config.")
    show.add_argument("--experiment", required=True)

    materialize = subparsers.add_parser("materialize", help="Write a runtime config to bulk storage.")
    materialize.add_argument("--experiment", required=True)

    prepare = subparsers.add_parser("prepare", help="Create/verify the epoch-0 population checkpoint.")
    prepare.add_argument("--experiment", required=True)
    prepare.add_argument("--execute", action="store_true")

    smoke = subparsers.add_parser("smoke", help="Run the no-write three-step smoke test.")
    smoke.add_argument("--experiment", required=True)
    smoke.add_argument("--device", default="cuda:1")
    smoke.add_argument("--execute", action="store_true")

    train = subparsers.add_parser("train", help="Print a train command; --execute starts it.")
    train.add_argument("--experiment", required=True)
    train.add_argument("--device", default="cuda:1")
    train.add_argument("--resume", default=None, help="Use latest/best/path only for an existing target run.")
    train.add_argument("--skip-periodic-evaluation", action="store_true")
    train.add_argument("--execute", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a completed target checkpoint.")
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument("--checkpoint", default="best_mesh")
    evaluate.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["val"])
    evaluate.add_argument("--device", default="cuda:1")
    evaluate.add_argument("--per-split", type=int, default=100)
    evaluate.add_argument("--resolution", type=int, default=256)
    evaluate.add_argument("--latent-steps", type=int, default=500)
    evaluate.add_argument("--surface-points", type=int, default=30000)
    evaluate.add_argument("--overwrite-meshes", action="store_true")
    evaluate.add_argument("--confirm-test", action="store_true")
    evaluate.add_argument("--execute", action="store_true")

    ceiling = subparsers.add_parser("ceiling", help="Run/print the 20-val latent-step and resolution pilot.")
    ceiling.add_argument("--device", default="cuda:1")
    ceiling.add_argument("--execute", action="store_true")

    export = subparsers.add_parser("export", help="Export final fitted latent codes.")
    export.add_argument("--experiment", required=True)
    export.add_argument("--checkpoint", default="best_mesh")
    export.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    export.add_argument("--steps", type=int, default=500)
    export.add_argument("--device", default="cuda:1")
    export.add_argument("--confirm-test", action="store_true")
    export.add_argument("--execute", action="store_true")

    audit_export = subparsers.add_parser("audit-export", help="Read-only audit of an exported latent table.")
    audit_export.add_argument("--experiment", required=True)
    audit_export.add_argument("--checkpoint", default="best_mesh")
    audit_export.add_argument("--require-all-splits", action="store_true")
    audit_export.add_argument("--execute", action="store_true")
    return parser.parse_args()


def command_prefix() -> list[str]:
    if not BULK_WRAPPER.is_file():
        raise FileNotFoundError(BULK_WRAPPER)
    return ["/bin/bash", str(BULK_WRAPPER)]


def execute_or_print(command: list[str], execute: bool) -> None:
    print(shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    else:
        print("Dry run only. Add --execute to run this command.", flush=True)


def ensure_materialized(matrix: dict, experiment: str, execute: bool) -> tuple[dict, Path]:
    config = materialize_config(matrix, experiment)
    path = runtime_config_path(matrix, experiment)
    if execute:
        validate(require_data=True, sampling_scans=3)
        path = write_runtime_config(matrix, experiment, config)
    return config, path


def prepare_command(config_path: Path) -> list[str]:
    return command_prefix() + [
        str(SCRIPT_DIR / "prepare_population_initialization.py"),
        "--config",
        str(config_path),
    ]


def require_test_confirmation(splits: list[str], confirmed: bool) -> None:
    if "test" in splits and not confirmed:
        raise PermissionError(
            "The test split is locked during selection. Add --confirm-test only after the final "
            "sampler and checkpoint were selected using validation data."
        )


def main() -> None:
    args = parse_args()
    matrix = load_matrix()
    if args.command == "check":
        print(json.dumps(validate(args.require_data, args.sampling_scans), indent=2, sort_keys=True))
        return
    if args.command == "show":
        print(json.dumps(materialize_config(matrix, args.experiment), indent=2, sort_keys=True))
        return
    if args.command == "materialize":
        config = materialize_config(matrix, args.experiment)
        print(write_runtime_config(matrix, args.experiment, config))
        return

    if args.command == "ceiling":
        config, _config_path = ensure_materialized(matrix, "control_u030_n100", args.execute)
        ceiling_config = copy.deepcopy(config)
        ceiling_config["name"] = "source_checkpoint_ceiling_pilot"
        ceiling_config["output_dir"] = str(require_bulk_path(matrix["bulk_output_root"]) / "ceiling_pilot")
        ceiling_config["periodic_evaluation"]["compare_pca"] = False
        ceiling_path = require_bulk_path(matrix["bulk_output_root"]) / "runtime_configs" / "ceiling_source.json"
        if args.execute:
            # The ceiling config has its own immutable filename and intentionally differs from control.
            from ablation_common import atomic_write_json, load_json
            if ceiling_path.is_file():
                existing = load_json(ceiling_path)
                if existing != ceiling_config:
                    raise FileExistsError("Existing ceiling config differs; refusing overwrite.")
            else:
                atomic_write_json(ceiling_path, ceiling_config)
        combinations = ((256, 500), (384, 500), (512, 500), (256, 1000), (256, 2000))
        for resolution, steps in combinations:
            output = require_bulk_path(matrix["bulk_output_root"]) / "ceiling_pilot" / f"resolution_{resolution}_steps_{steps}"
            command = command_prefix() + [
                str(EVALUATE_SCRIPT), "--config", str(ceiling_path),
                "--checkpoint", matrix["source_checkpoint"],
                "--output-dir", str(output), "--device", args.device,
                "--splits", "val", "--per-split", "20",
                "--resolution", str(resolution), "--latent-steps", str(steps),
                "--surface-points", "30000",
            ]
            execute_or_print(command, args.execute)
        return

    config, config_path = ensure_materialized(matrix, args.experiment, args.execute)
    init_path = initialization_path(matrix, args.experiment)
    if args.command == "prepare":
        execute_or_print(prepare_command(config_path), args.execute)
        return
    if args.command in {"smoke", "train"} and args.execute:
        subprocess.run(prepare_command(config_path), cwd=REPO_ROOT, check=True)
    if args.command == "smoke":
        command = command_prefix() + [
            str(TRAIN_SCRIPT), "--config", str(config_path), "--device", args.device,
            "--resume", str(init_path), "--smoke-test", "--skip-periodic-evaluation",
        ]
        execute_or_print(command, args.execute)
        return
    if args.command == "train":
        history = run_dir(matrix, args.experiment) / "logs" / "training_history.csv"
        if args.execute and history.is_file() and args.resume is None:
            raise FileExistsError(
                f"Target run already has training history: {history}. Use --resume latest to continue."
            )
        resume = args.resume or str(init_path)
        command = command_prefix() + [
            str(TRAIN_SCRIPT), "--config", str(config_path), "--device", args.device,
            "--resume", resume,
        ]
        if args.skip_periodic_evaluation:
            command.append("--skip-periodic-evaluation")
        execute_or_print(command, args.execute)
        return
    if args.command == "evaluate":
        require_test_confirmation(args.splits, args.confirm_test)
        label = Path(args.checkpoint).stem
        split_label = "_".join(args.splits)
        output = run_dir(matrix, args.experiment) / "manual_evaluation" / label / split_label
        command = command_prefix() + [
            str(EVALUATE_SCRIPT), "--config", str(config_path),
            "--checkpoint", args.checkpoint, "--output-dir", str(output),
            "--device", args.device, "--splits", *args.splits,
            "--per-split", str(args.per_split), "--resolution", str(args.resolution),
            "--latent-steps", str(args.latent_steps), "--surface-points", str(args.surface_points),
        ]
        if args.overwrite_meshes:
            command.append("--overwrite-meshes")
        execute_or_print(command, args.execute)
        return
    if args.command == "export":
        require_test_confirmation(args.splits, args.confirm_test)
        output = run_dir(matrix, args.experiment) / "latent_exports" / Path(args.checkpoint).stem
        command = command_prefix() + [
            str(EXPORT_SCRIPT), "--config", str(config_path), "--checkpoint", args.checkpoint,
            "--output-dir", str(output), "--device", args.device,
            "--splits", *args.splits, "--steps", str(args.steps),
        ]
        execute_or_print(command, args.execute)
        return
    if args.command == "audit-export":
        output = run_dir(matrix, args.experiment) / "latent_exports" / Path(args.checkpoint).stem
        command = command_prefix() + [
            str(SCRIPT_DIR / "validate_latent_export.py"),
            "--config", str(config_path), "--export-dir", str(output),
        ]
        if args.require_all_splits:
            command.append("--require-all-splits")
        execute_or_print(command, args.execute)
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
