#!/usr/bin/env python3
"""Evaluate hash-grid CALSNIC reconstructions, one row per readout variant.

Every metric, the PCA oracle, the subject-clustered bootstrap and the output
schemas are imported from the CALSNIC multires evaluator so MR64/MR128 and these
runs are scored by identical code.  This script adds only the variant loop: a
Compact-SDF checkpoint is decoded as global-only, local-only, hard-band fused and
smooth-gate fused from the *same* fitted latent, which makes the fusion ablation
free of any extra training.

``method`` stays ``inr`` for the configured selection variant and becomes
``inr_<variant>`` for the others, so ``compare_evaluations.py`` and the existing
notebook keep working while every variant remains in the CSV.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hashgrid_common import (
    CALSNIC_SCRIPTS,
    architecture_name,
    choose_device,
    decode_variant_to_mesh,
    fit_single_latent,
    is_two_branch,
    load_config,
    load_decoder_checkpoint,
    mesh_sdf_to_mm,
    read_manifest,
    require_bulk_path,
    stable_seed,
    variants_for,
    write_csv,
    write_json,
)
def _load_calsnic_evaluator():
    """Load the CALSNIC evaluator by path, not by bare module name.

    Both the ADNI and the CALSNIC multires tasks ship a module called
    ``periodic_evaluate_multires``, so a bare import resolves by sys.path order
    and silently picks up whichever happens to come first. Only the CALSNIC one
    defines METRICS and the millimetre PCA baseline.
    """
    import importlib.util

    path = CALSNIC_SCRIPTS / "periodic_evaluate_multires.py"
    spec = importlib.util.spec_from_file_location("calsnic_periodic_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the CALSNIC evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_calsnic = _load_calsnic_evaluator()
METRICS = _calsnic.METRICS
MatchedPCA = _calsnic.MatchedPCA
atomic_export = _calsnic.atomic_export
cluster_bootstrap = _calsnic.cluster_bootstrap
load_mesh = _calsnic.load_mesh
surface_metrics = _calsnic.surface_metrics

SELECTION_DEFAULT = {
    "conditional_instant_ngp_sdf": "single",
    "compact_sdf_two_branch_instant_ngp": "fused_hard_band",
    "grid_free_band_limited_fourier_sdf": "single",
    "deformed_implicit_field": "single",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=None)
    parser.add_argument("--per-split", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--latent-steps", type=int, default=None)
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--overwrite-meshes", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Per split/method group, plus an alias for the selection variant."""
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"split:{row['split']}/method:{row['method']}"].append(row)
    report: dict[str, Any] = {}
    for name, group in groups.items():
        subject_ids = np.asarray([row["subject_id"] for row in group])
        report[name] = {}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group])
            report[name][metric] = {
                "count": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "subject_bootstrap_mean_95ci": cluster_bootstrap(
                    values, subject_ids, stable_seed(name + metric, seed)
                ),
            }
    return report


def paired(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Per-variant paired deltas against the PCA oracle on the same scans."""
    lookup = {(row["scan_id"], row["split"], row["method"]): row for row in rows}
    methods = sorted({row["method"] for row in rows if row["method"] != "pca"})
    report: dict[str, Any] = {}
    for split in sorted({row["split"] for row in rows}):
        scan_ids = sorted({row["scan_id"] for row in rows if row["split"] == split})
        report[split] = {}
        for method in methods:
            pairs = [
                scan_id
                for scan_id in scan_ids
                if (scan_id, split, method) in lookup and (scan_id, split, "pca") in lookup
            ]
            if not pairs:
                continue
            report[split][method] = {}
            for metric in METRICS:
                differences = np.asarray(
                    [
                        float(lookup[(scan_id, split, method)][metric])
                        - float(lookup[(scan_id, split, "pca")][metric])
                        for scan_id in pairs
                    ]
                )
                ids = np.asarray(
                    [lookup[(scan_id, split, method)]["subject_id"] for scan_id in pairs]
                )
                higher_is_better = metric.startswith("fscore") or metric == "normal_absolute_cosine"
                report[split][method][metric] = {
                    "count": len(differences),
                    "mean_inr_minus_pca": float(differences.mean()),
                    "median_inr_minus_pca": float(np.median(differences)),
                    "subject_bootstrap_mean_95ci": cluster_bootstrap(
                        differences, ids, stable_seed(split + method + metric, seed)
                    ),
                    "fraction_inr_better": float(
                        np.mean(differences > 0.0 if higher_is_better else differences < 0.0)
                    ),
                }
    return report


def decode_mm(
    decoder,
    latent: np.ndarray,
    row: dict[str, str],
    output: Path,
    resolution: int,
    max_batch: int,
    device,
    variant: str,
    band_cells: int,
) -> tuple[Any, dict[str, Any]]:
    temporary = output.parent / "_sdf_temporary" / f"{row['scan_id']}.ply"
    report = decode_variant_to_mesh(
        decoder, latent, temporary, resolution, max_batch, device, variant, band_cells
    )
    mm_mesh = mesh_sdf_to_mm(load_mesh(temporary), row)
    atomic_export(mm_mesh, output)
    temporary.unlink()
    return mm_mesh, report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    periodic = config["periodic_evaluation"]
    splits = list(args.splits or periodic.get("splits", ["val"]))
    if "test" in splits and not args.confirm_test:
        raise PermissionError("Test evaluation is locked; add --confirm-test after model selection.")
    device = choose_device(args.device)
    decoder, payload, checkpoint = load_decoder_checkpoint(config, args.checkpoint, device)
    output = require_bulk_path(
        args.output_dir or Path(config["output_dir"]) / "manual_evaluation" / checkpoint.stem
    )

    architecture = architecture_name(config)
    variants = variants_for(config, args.variants or periodic.get("variants"))
    # Resolve the key first: a dict.get default is evaluated eagerly, so an
    # architecture missing from SELECTION_DEFAULT would raise even when the
    # config states the variant explicitly.
    selection_variant = str(
        periodic.get("selection_variant") or SELECTION_DEFAULT[architecture]
    )
    if selection_variant not in variants:
        raise ValueError(
            f"selection_variant {selection_variant!r} is not among the decoded variants {variants}."
        )
    band_cells = int(periodic.get("band_cells", 3))
    per_split = int(args.per_split or periodic.get("per_split", 15))
    resolution = int(args.resolution or periodic.get("resolution", 256))
    latent_steps = int(args.latent_steps or periodic.get("latent_steps", 750))
    surface_points = int(args.surface_points or periodic.get("surface_points", 30000))
    max_batch = int(config["reconstruction"]["max_batch"])

    rows = read_manifest(config["manifest"])
    selected = []
    for split in splits:
        candidates = [row for row in rows if row["split"] == split]
        selected.extend(candidates[: min(per_split, len(candidates))])
    train_ids = list(payload["training_scan_ids"])
    train_table = payload["latent_codes"].detach().cpu().numpy()
    learned = {scan_id: train_table[index] for index, scan_id in enumerate(train_ids)}
    pca = None
    if periodic.get("compare_pca", True):
        pca = MatchedPCA(Path(periodic["pca_model_dir"]), int(periodic.get("pca_components", 172)))

    results: list[dict[str, Any]] = []
    latent_reports: list[dict[str, Any]] = []
    decode_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    for number, row in enumerate(selected, start=1):
        try:
            # One compact 256-D code per subject, shared by every readout variant.
            if row["split"] == "train":
                latent = np.asarray(learned[row["scan_id"]], dtype=np.float32)
                latent_report = {"source": "learned_embedding", "steps_completed": 0}
            else:
                latent, fit_report = fit_single_latent(
                    decoder,
                    row["sdf_npz_path"],
                    int(config["latent_size"]),
                    config["latent_fit"],
                    float(config["clamp_distance"]),
                    device,
                    stable_seed(row["scan_id"], int(config["seed"])),
                    config["network_specs"],
                    steps_override=latent_steps,
                )
                latent_report = {"source": "frozen_decoder_fit", **fit_report}
            latent_reports.append(
                {"scan_id": row["scan_id"], "split": row["split"], **latent_report}
            )
            ground_truth = load_mesh(row["mesh_path_mm"])
            common = {
                "scan_id": row["scan_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "diagnosis": row.get("diagnosis", ""),
            }
            seed = stable_seed(row["scan_id"], int(config["seed"]))
            for variant in variants:
                method = "inr" if variant == selection_variant else f"inr_{variant}"
                mesh_path = output / "meshes" / variant / row["split"] / f"{row['scan_id']}.ply"
                if mesh_path.is_file() and not args.overwrite_meshes:
                    mesh = load_mesh(mesh_path)
                else:
                    mesh, decode_report = decode_mm(
                        decoder, latent, row, mesh_path, resolution, max_batch,
                        device, variant, band_cells,
                    )
                    decode_reports.append(
                        {"scan_id": row["scan_id"], "split": row["split"],
                         "variant": variant, **decode_report}
                    )
                results.append(
                    {
                        **common,
                        "method": method,
                        "variant": variant,
                        **surface_metrics(ground_truth, mesh, surface_points, seed),
                        "mesh_path": str(mesh_path),
                    }
                )
            if pca is not None:
                pca_path = output / "meshes" / "pca" / row["split"] / f"{row['scan_id']}.ply"
                pca_mesh = pca.reconstruct(row)
                if args.overwrite_meshes or not pca_path.is_file():
                    atomic_export(pca_mesh, pca_path)
                results.append(
                    {
                        **common,
                        "method": "pca",
                        "variant": "pca_oracle",
                        **surface_metrics(ground_truth, pca_mesh, surface_points, seed),
                        "mesh_path": str(pca_path),
                    }
                )
        except Exception as error:  # noqa: BLE001 - recorded per scan, not swallowed
            failures.append({"scan_id": row["scan_id"], "split": row["split"], "error": repr(error)})
        print(f"[{number}/{len(selected)}] {row['split']} {row['scan_id']}", flush=True)

    write_csv(output / "per_scan_metrics.csv", results)
    write_csv(output / "latent_fit_metrics.csv", latent_reports)
    write_csv(output / "decode_reports.csv", decode_reports)
    write_json(output / "failures.json", failures)

    minimum = float(periodic.get("minimum_success_fraction", 1.0))
    expected_methods = [
        "inr" if variant == selection_variant else f"inr_{variant}" for variant in variants
    ] + (["pca"] if pca is not None else [])
    for split in splits:
        expected = sum(row["split"] == split for row in selected)
        for method in expected_methods:
            succeeded = sum(
                row["split"] == split and row["method"] == method for row in results
            )
            if succeeded < minimum * expected:
                raise RuntimeError(
                    f"Only {succeeded}/{expected} {split} {method} evaluations succeeded."
                )
    validation = [
        float(row["assd_mm"])
        for row in results
        if row["split"] == "val" and row["method"] == "inr"
    ]
    if "val" in splits and not validation:
        raise RuntimeError("No validation INR mesh succeeded.")

    report = {
        "architecture": architecture,
        "two_branch": is_two_branch(config),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "coordinate_space": "physical millimetres",
        "sdf_to_mm_transform": "per-subject inverse scaled-OBJ similarity transform",
        "splits": splits,
        "variants": variants,
        "selection_variant": selection_variant,
        "band_cells": band_cells,
        "latent_policy": "one 256-D code per subject, shared by every readout variant",
        "test_confirmed": bool(args.confirm_test),
        "resolution": resolution,
        "marching_cubes_voxel_mm": 2.0
        / (resolution - 1)
        * float(config["mm_per_normalized_unit"]),
        "surface_points": surface_points,
        "seconds": time.time() - started,
        "failures": failures,
        "selection_metric": {
            "name": f"validation INR ASSD in millimetres ({selection_variant})",
            "validation_inr_assd_mm": float(np.mean(validation)) if validation else None,
        },
        "summary": summarize(results, int(config["seed"])),
        "paired_inr_minus_pca": paired(results, int(config["seed"])) if pca is not None else {},
        "pca_interpretation": "train-only basis; each target is projected to its own PCA coefficients (oracle reconstruction)",
        "test_policy": "manual confirmation required; test never selects an architecture or checkpoint",
    }
    write_json(output / "summary.json", report)
    print(
        f"Evaluation complete: rows={len(results)}, variants={variants}, "
        f"failures={len(failures)}, output={output}"
    )


if __name__ == "__main__":
    main()
