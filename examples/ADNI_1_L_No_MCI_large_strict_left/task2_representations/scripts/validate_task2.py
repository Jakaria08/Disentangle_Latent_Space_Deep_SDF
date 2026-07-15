#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from task2_common import (
    TASK_DIR,
    load_config,
    load_json,
    load_manifest,
    resolve_repo_path,
    write_json,
)


STAGES = ("inputs", "pca", "trained", "latents", "complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Task 2 artifacts by stage.")
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "pipeline.json"),
    )
    parser.add_argument("--stage", choices=STAGES, default="complete")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    rows = load_manifest(config["manifest"])
    inr_models = config.get("inr_models", {})
    expected_total = int(config["expected_counts"]["total"])
    expected_by_split = {
        split: int(config["expected_counts"][split])
        for split in ("train", "val", "test")
    }
    errors = []
    warnings = []
    checks = {}
    stage_index = STAGES.index(args.stage)

    audit_path = TASK_DIR / "audit" / "input_audit.json"
    if not audit_path.is_file():
        errors.append(f"Missing input audit: {audit_path}")
    else:
        audit = load_json(audit_path)
        checks["input_audit_status"] = audit.get("status")
        if audit.get("status") != "pass":
            errors.append("Input audit did not pass.")

    if stage_index >= STAGES.index("pca"):
        model_dir = TASK_DIR / "pca" / "model"
        expected_feature_count = int(config["mesh"]["vertex_count"]) * 3
        expected_face_count = int(config["mesh"]["face_count"])
        expected_components = int(config["pca"]["max_components"])
        required = [
            model_dir / "mean.npy",
            model_dir / "components_256.npy",
            model_dir / "faces.npy",
            model_dir / "pca_config.json",
            model_dir / "pca_brainode.pkl",
            TASK_DIR / "pca" / "coefficients" / "pca_coefficients.csv",
        ]
        for path in required:
            if not path.is_file():
                errors.append(f"Missing PCA artifact: {path}")
        if all(path.is_file() for path in required[:3]):
            mean = np.load(required[0])
            components = np.load(required[1])
            faces = np.load(required[2])
            checks["pca_shapes"] = {
                "mean": list(mean.shape),
                "components": list(components.shape),
                "faces": list(faces.shape),
            }
            if mean.shape != (expected_feature_count,):
                errors.append(f"Unexpected PCA mean shape: {mean.shape}")
            if components.shape != (expected_components, expected_feature_count):
                errors.append(f"Unexpected PCA component shape: {components.shape}")
            if faces.shape != (expected_face_count, 3):
                errors.append(f"Unexpected PCA face shape: {faces.shape}")
            if not np.isfinite(mean).all() or not np.isfinite(components).all():
                errors.append("PCA arrays contain non-finite values.")
        coefficient_csv = required[-1]
        if coefficient_csv.is_file():
            coefficient_rows = read_csv(coefficient_csv)
            checks["pca_coefficient_rows"] = len(coefficient_rows)
            if len(coefficient_rows) != expected_total:
                errors.append(
                    f"Expected {expected_total} PCA rows, found {len(coefficient_rows)}."
                )

    if stage_index >= STAGES.index("trained"):
        for name, spec in inr_models.items():
            output = resolve_repo_path(spec["output_dir"])
            checkpoint = resolve_repo_path(spec["checkpoint"])
            checkpoint_kind = spec.get("checkpoint_kind", "task2_trained")
            if checkpoint_kind == "task2_trained":
                best = output / "checkpoints" / "best.pth"
                status = output / "training_status.json"
                if not best.is_file():
                    errors.append(f"Missing validation-selected checkpoint: {best}")
                if not status.is_file():
                    errors.append(f"Missing training status: {status}")
                else:
                    status_data = load_json(status)
                    checks[f"{name}_training_status"] = status_data.get("status")
                    if status_data.get("status") != "complete":
                        errors.append(f"{name} training is not complete.")
            elif checkpoint_kind == "external_pretrained":
                checks[f"{name}_checkpoint_path"] = str(checkpoint)
                if not checkpoint.is_file():
                    errors.append(f"Missing external pretrained checkpoint: {checkpoint}")
            else:
                errors.append(f"Unknown checkpoint kind for {name}: {checkpoint_kind}")

    if stage_index >= STAGES.index("latents"):
        for name, spec in inr_models.items():
            output = resolve_repo_path(spec["output_dir"])
            latent_root = output / "latents"
            csv_path = latent_root / "inr_latents.csv"
            if not csv_path.is_file():
                errors.append(f"Missing latent CSV: {csv_path}")
                continue
            latent_rows = read_csv(csv_path)
            checks[f"{name}_latent_rows"] = len(latent_rows)
            if len(latent_rows) != expected_total:
                errors.append(
                    f"Expected {expected_total} {name} latents, found {len(latent_rows)}."
                )
            for split, expected in expected_by_split.items():
                archive_path = latent_root / f"{split}_latents.npz"
                if not archive_path.is_file():
                    errors.append(f"Missing latent archive: {archive_path}")
                    continue
                with np.load(archive_path) as archive:
                    latents = archive["latents"]
                expected_latent_size = int(spec.get("latent_size", 256))
                if latents.shape != (expected, expected_latent_size):
                    errors.append(
                        f"Unexpected {name} {split} latent shape: {latents.shape}"
                    )
                if not np.isfinite(latents).all():
                    errors.append(f"Non-finite {name} {split} latents.")
            nonconverged = [
                row
                for row in latent_rows
                if row["converged"].strip().lower() != "true"
            ]
            checks[f"{name}_nonconverged_latents"] = len(nonconverged)
            if nonconverged:
                warnings.append(f"{name} has {len(nonconverged)} non-converged fits.")

    if stage_index >= STAGES.index("complete"):
        representation_manifest = (
            TASK_DIR / "metadata" / "representation_manifest.csv"
        )
        if not representation_manifest.is_file():
            errors.append(
                f"Missing combined representation manifest: {representation_manifest}"
            )
        else:
            representation_rows = read_csv(representation_manifest)
            checks["representation_manifest_rows"] = len(representation_rows)
            if len(representation_rows) != expected_total:
                errors.append(
                    f"Expected {expected_total} representation rows, "
                    f"found {len(representation_rows)}."
                )
        for method, relative in config["evaluation"]["methods"].items():
            directory = TASK_DIR / relative
            count = sum(
                (directory / f"{row['scan_id']}.ply").is_file() for row in rows
            )
            checks[f"{method}_mesh_count"] = count
            if count != expected_total:
                errors.append(
                    f"Expected {expected_total} meshes for {method}, found {count}."
                )
        metric_path = (
            TASK_DIR / "evaluation" / "metrics" / "reconstruction_per_scan.csv"
        )
        if not metric_path.is_file():
            errors.append(f"Missing reconstruction metrics: {metric_path}")
        else:
            metric_rows = read_csv(metric_path)
            expected_metric_rows = expected_total * len(
                config["evaluation"]["methods"]
            )
            checks["reconstruction_metric_rows"] = len(metric_rows)
            if len(metric_rows) != expected_metric_rows:
                errors.append(
                    f"Expected {expected_metric_rows} metric rows, found {len(metric_rows)}."
                )
            numeric_columns = (
                "chamfer_l2_squared",
                "assd",
                "hd95",
                "volume_absolute_error",
                "volume_relative_error",
            )
            for row in metric_rows:
                if not all(np.isfinite(float(row[key])) for key in numeric_columns):
                    errors.append(
                        f"Non-finite metric for {row['method']}:{row['scan_id']}"
                    )
                    break
        comparison = (
            TASK_DIR
            / "evaluation"
            / "comparison"
            / "inr_model_comparison.json"
        )
        if not comparison.is_file():
            errors.append(f"Missing INR comparison report: {comparison}")

    report = {
        "stage": args.stage,
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    output_path = TASK_DIR / "metadata" / f"validation_{args.stage}.json"
    write_json(output_path, report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
