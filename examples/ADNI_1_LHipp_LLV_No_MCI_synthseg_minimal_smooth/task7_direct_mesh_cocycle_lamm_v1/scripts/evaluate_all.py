#!/usr/bin/env python3
"""Run matched native evaluators and centralize direct/latent flow comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[1]
COHORT_TASK_ROOT = TASK_ROOT.parent
DIRECT_WORKER = TASK_ROOT / "scripts" / "evaluate_direct_first_last.py"
LATENT_VELOCITY_WORKER = TASK_ROOT / "scripts" / "evaluate_latent_velocity.py"
LATENT_SURFACE_WORKER = (
    COHORT_TASK_ROOT / "task3_latent_flow_128_v2_lamm" / "scripts" / "evaluate_surface_forecasts.py"
)
DEFAULT_MANIFEST = TASK_ROOT / "configs" / "central_evaluation.json"
DEFAULT_OUTPUT = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/comparisons"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-velocity-scans", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported central-evaluation manifest")
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Manifest must contain methods")
    names: set[str] = set()
    for method in methods:
        name = str(method.get("name", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Unsafe method name: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate method name: {name}")
        names.add(name)
        family = method.get("family")
        if family not in {"direct_lamm", "direct_spiral", "latent"}:
            raise ValueError(f"Unknown family for {name}: {family}")
        required = "run_dir" if family == "latent" else "checkpoint"
        if not method.get(required):
            raise ValueError(f"{name} is missing {required}")
    return methods


def run_command(command: list[str], dry_run: bool) -> None:
    print("COMMAND:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def direct_command(
    python: str,
    method: dict[str, Any],
    args: argparse.Namespace,
    data_root: str,
    output: Path,
) -> list[str]:
    command = [
        python,
        str(DIRECT_WORKER),
        "--family",
        "lamm" if method["family"] == "direct_lamm" else "spiral",
        "--name",
        method["name"],
        "--checkpoint",
        method["checkpoint"],
        "--split",
        args.split,
        "--device",
        args.device,
        "--data-root",
        data_root,
        "--surface-points",
        str(args.surface_points),
        "--max-velocity-scans",
        str(args.max_velocity_scans),
        "--output-dir",
        str(output),
    ]
    if args.max_subjects is not None:
        command.extend(("--max-subjects", str(args.max_subjects)))
    if args.split == "test":
        command.append("--allow-test")
    return command


def latent_commands(
    python: str,
    method: dict[str, Any],
    args: argparse.Namespace,
    data_root: str,
    output: Path,
) -> list[list[str]]:
    surface = [
        python,
        str(LATENT_SURFACE_WORKER),
        "--run",
        f"{method['name']}={method['run_dir']}",
        "--split",
        args.split,
        "--device",
        args.device,
        "--surface-points",
        str(args.surface_points),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--output-dir",
        str(output / "surface"),
    ]
    velocity = [
        python,
        str(LATENT_VELOCITY_WORKER),
        "--name",
        method["name"],
        "--run-dir",
        method["run_dir"],
        "--split",
        args.split,
        "--device",
        args.device,
        "--reference-root",
        data_root,
        "--max-scans",
        str(args.max_velocity_scans),
        "--output-dir",
        str(output / "velocity"),
    ]
    if args.max_subjects is not None:
        surface.extend(("--max-subjects", str(args.max_subjects)))
    if args.split == "test":
        surface.append("--allow-test")
        velocity.append("--allow-test")
    return [surface, velocity]


def direct_rows(method: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for diagnosis, endpoint in summary["endpoint"].items():
        surface = summary["surface"][diagnosis]
        velocity = summary["instantaneous_velocity"]["groups"].get(diagnosis, {})
        rows.append(
            {
                "method": method["name"],
                "family": method["family"],
                "diagnosis": diagnosis,
                "subjects": endpoint["pairs"],
                "mean_vertex_error_mm": endpoint["mean_vertex_error_mm"],
                "mean_vertex_error_ratio_to_nochange": endpoint["error_to_nochange_ratio"],
                "assd_mm": surface["prediction_assd_mm"],
                "assd_ratio_to_nochange": surface["prediction_assd_mm_ratio_to_nochange"],
                "hd95_mm": surface["prediction_hd95_mm"],
                "chamfer_l2_squared_mm2": surface["prediction_chamfer_l2_squared_mm2"],
                "normal_signed_cosine": surface["prediction_normal_signed_cosine"],
                "flipped_face_fraction": surface[
                    "prediction_flipped_face_fraction_vs_ground_truth"
                ],
                "volume_relative_error": endpoint["volume_relative_error"],
                "observed_log_volume_rate_per_year": endpoint[
                    "observed_log_volume_rate_per_year"
                ],
                "predicted_log_volume_rate_per_year": endpoint[
                    "predicted_log_volume_rate_per_year"
                ],
                "volume_rate_abs_error_per_year": endpoint[
                    "volume_rate_abs_error_per_year"
                ],
                "velocity_normalized_error_ratio": velocity.get("normalized_error_ratio"),
                "velocity_speed_ratio": velocity.get("speed_ratio"),
                "velocity_vector_cosine": velocity.get("vector_cosine"),
                "velocity_normal_pearson": velocity.get("normal_pearson"),
                "relative_cocycle_defect_mean": summary["cocycle"][
                    "relative_cocycle_defect_mean"
                ],
                "relative_inverse_defect_mean": summary["cocycle"][
                    "relative_inverse_defect_mean"
                ],
            }
        )
    return rows


def latent_rows(
    method: dict[str, Any], surface_summary: dict[str, Any], velocity_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    native_path = Path(method["run_dir"]) / "evaluation" / surface_summary["split"] / "summary.json"
    native = read_json(native_path) if native_path.is_file() else {}
    consistency = native.get("consistency_defects", {})
    output = []
    for row in surface_summary["summary_rows"]:
        if row["representation"] != method["name"]:
            continue
        diagnosis = row["diagnosis"]
        prediction_vertex = row["prediction_mean_vertex_euclidean_mm_mean"]
        nochange_vertex = row["nochange_mean_vertex_euclidean_mm_mean"]
        velocity = velocity_summary["groups"].get(diagnosis, {})
        output.append(
            {
                "method": method["name"],
                "family": method["family"],
                "diagnosis": diagnosis,
                "subjects": row["subjects"],
                "mean_vertex_error_mm": prediction_vertex,
                "mean_vertex_error_ratio_to_nochange": prediction_vertex
                / max(nochange_vertex, 1.0e-12),
                "assd_mm": row["prediction_assd_mm_mean"],
                "assd_ratio_to_nochange": row[
                    "prediction_assd_mm_ratio_to_nochange"
                ],
                "hd95_mm": row["prediction_hd95_mm_mean"],
                "chamfer_l2_squared_mm2": row[
                    "prediction_chamfer_l2_squared_mm2_mean"
                ],
                "normal_signed_cosine": row["prediction_normal_signed_cosine_mean"],
                "flipped_face_fraction": row[
                    "prediction_flipped_face_fraction_vs_ground_truth_mean"
                ],
                "volume_relative_error": row["prediction_volume_relative_error_mean"],
                "observed_log_volume_rate_per_year": row[
                    "observed_signed_log_volume_rate_per_year_mean"
                ],
                "predicted_log_volume_rate_per_year": row[
                    "prediction_signed_log_volume_rate_raw_anchor_per_year_mean"
                ],
                "volume_rate_abs_error_per_year": row[
                    "prediction_log_volume_rate_absolute_error_per_year_mean"
                ],
                "velocity_normalized_error_ratio": velocity.get("normalized_error_ratio"),
                "velocity_speed_ratio": velocity.get("speed_ratio"),
                "velocity_vector_cosine": velocity.get("vector_cosine"),
                "velocity_normal_pearson": velocity.get("normal_pearson"),
                "relative_cocycle_defect_mean": consistency.get(
                    "relative_semigroup_defect_mean"
                ),
                "relative_inverse_defect_mean": consistency.get(
                    "relative_inverse_defect_mean"
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No comparable methods were evaluated")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise PermissionError("Test evaluation requires --allow-test")
    manifest = read_json(args.manifest)
    methods = validate_manifest(manifest)
    if args.only:
        requested = set(args.only)
        methods = [method for method in methods if method["name"] in requested]
        missing_names = requested.difference(method["name"] for method in methods)
        if missing_names:
            raise KeyError(f"Unknown --only methods: {sorted(missing_names)}")
    python = str(Path(manifest["worker_python"]).expanduser().resolve())
    data_root = str(Path(manifest["direct_mesh_data_root"]).expanduser().resolve())
    destination = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT / f"all_methods_{args.split}"
    )
    if destination.exists() and not args.dry_run:
        raise FileExistsError(destination)

    active: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for method in methods:
        required = Path(method["run_dir"] if method["family"] == "latent" else method["checkpoint"])
        if not required.exists() and not args.dry_run:
            if args.skip_missing:
                print(f"SKIP MISSING: {method['name']} -> {required}", flush=True)
                continue
            raise FileNotFoundError(required)
        output = destination / method["name"]
        if method["family"].startswith("direct_"):
            commands.append(direct_command(python, method, args, data_root, output))
        else:
            commands.extend(latent_commands(python, method, args, data_root, output))
        active.append(method)

    for command in commands:
        run_command(command, args.dry_run)
    if args.dry_run:
        print(f"DRY RUN: {len(active)} methods, {len(commands)} worker commands")
        return 0

    comparison: list[dict[str, Any]] = []
    for method in active:
        output = destination / method["name"]
        if method["family"].startswith("direct_"):
            comparison.extend(direct_rows(method, read_json(output / "summary.json")))
        else:
            comparison.extend(
                latent_rows(
                    method,
                    read_json(output / "surface" / "summary.json"),
                    read_json(output / "velocity" / "summary.json"),
                )
            )
    overall = sorted(
        [row for row in comparison if row["diagnosis"] == "overall"],
        key=lambda row: float(row["mean_vertex_error_mm"]),
    )
    rank = {row["method"]: index + 1 for index, row in enumerate(overall)}
    for row in comparison:
        row["overall_mean_vertex_rank"] = rank[row["method"]]
    comparison.sort(key=lambda row: (rank[row["method"]], ("CN", "AD", "overall").index(row["diagnosis"])))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "comparison.csv", comparison)
    report = {
        "schema_version": 1,
        "split": args.split,
        "protocol": "same first-to-last held-out subjects; raw aligned meshes; exact selected best.pt",
        "velocity_definition": "zero-horizon model velocity mapped to mm/year surface coordinates",
        "velocity_reference": "reliability-weighted fit to repeated observed visits; not direct physical ground truth",
        "surface_points_per_direction": args.surface_points,
        "methods": active,
        "comparison_rows": comparison,
        "overall_ranking_by_mean_vertex_error": [row["method"] for row in overall],
        "test_was_explicitly_authorized": bool(args.split == "test" and args.allow_test),
    }
    with (destination / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
