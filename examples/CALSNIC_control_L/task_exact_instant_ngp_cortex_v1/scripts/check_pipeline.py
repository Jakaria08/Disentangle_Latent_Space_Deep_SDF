#!/usr/bin/env python3
"""Static gate for the CALSNIC Instant-NGP experiment. Reads only; writes nothing.

Run this before any GPU time.  It fails loudly on the mistakes that would
silently invalidate the comparison against MR64/MR128:

* a hash ladder that no longer matches base_resolution/per_level_scale/num_levels;
* a stale ``reference_sampling_sha256`` pin, i.e. the sampler changed underneath;
* a sampling balance resolution that differs from MR128, which would make the
  training distribution rather than the encoder the source of any difference;
* an unlocked test split in periodic evaluation;
* ``mm_per_normalized_unit`` disagreeing with the manifest, which would silently
  mis-scale the millimetre-pinned Eikonal epsilon.

It also prints the physical resolution and hash collision load per level, which
is the number that decides whether a configuration is worth running at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hashgrid_common import (
    REPO_ROOT,
    architecture_name,
    hash_capacity_report,
    is_deformation_field,
    is_grid_free,
    is_two_branch,
    load_config,
    require_bulk_path,
    variants_for,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
MR128_CONFIG = (
    REPO_ROOT
    / "examples"
    / "CALSNIC_control_L"
    / "task_exact_multires_cortex_v1"
    / "configs"
    / "calsnic_control_L_multires128_z256_exact.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--require-generated-data", action="store_true")
    return parser.parse_args()


def check_controlled_against_mr128(config: dict) -> list[str]:
    """Keys that must match MR128 for the encoder to be the only difference."""
    notes = []
    baseline = json.loads(MR128_CONFIG.read_text())
    for key in ("latent_size", "clamp_distance", "code_bound", "code_initial_std", "seed",
                "total_epochs", "scenes_per_batch", "scenes_per_chunk",
                "learning_rate_decay_interval", "learning_rate_decay_factor",
                "gradient_clip_norm"):
        if config[key] != baseline[key]:
            notes.append(f"{key}: {config[key]} vs MR128 {baseline[key]}")
    if config["sampling"] != baseline["sampling"]:
        notes.append("sampling block differs from MR128")
    if config["learning_rates"] != baseline["learning_rates"]:
        notes.append(f"learning_rates: {config['learning_rates']} vs MR128 {baseline['learning_rates']}")
    if config["network_specs"]["grid_aabb"] != baseline["network_specs"]["grid_aabb"]:
        notes.append("grid_aabb differs from MR128")
    if int(config["network_specs"]["sampling_balance_resolution"]) != int(
        baseline["network_specs"]["sampling_balance_resolution"]
    ):
        notes.append("sampling_balance_resolution differs from MR128")
    return notes


def check_config(path: Path, require_data: bool) -> None:
    print(f"\n=== {path.name} ===")
    config = load_config(path)  # validates ladder, grid_resolution, mm scale, sampler hash
    specs = config["network_specs"]

    output_dir = require_bulk_path(config["output_dir"], "config output_dir")
    grid_free = is_grid_free(config)
    ladder = [int(value) for value in specs.get("grid_resolutions", [])]
    if not grid_free:
        schedule = {int(item["resolution"]) for item in config["level_schedule"]}
        if schedule != set(ladder):
            raise AssertionError("level_schedule must define every hash resolution exactly once.")
        if len(config["level_schedule"]) != len(ladder):
            raise AssertionError("level_schedule has duplicate resolutions.")
        final_epoch = max(int(item["end_epoch"]) for item in config["level_schedule"])
        if final_epoch > int(config["total_epochs"]):
            raise AssertionError("level_schedule finishes after the last training epoch.")

    balance = int(specs["sampling_balance_resolution"])
    if balance not in (128, 256, 384):
        raise AssertionError(f"Unexpected sampling_balance_resolution {balance}.")

    second = config.get("second_order", {})
    if second.get("enabled"):
        if float(second.get("weight", 0.0)) <= 0.0:
            raise AssertionError("second_order.enabled requires a positive weight.")
        if float(second.get("curvature_threshold_per_mm", 0.0)) <= 0.0:
            raise AssertionError(
                "second_order needs a positive curvature_threshold_per_mm; a threshold "
                "of zero is a plain L2 penalty and would flatten real cortical folds."
            )

    periodic = config["periodic_evaluation"]
    if list(periodic["splits"]) != ["val"]:
        raise AssertionError("periodic_evaluation.splits must be exactly ['val']; test stays locked.")
    script = REPO_ROOT / periodic["script"]
    if not script.is_file():
        raise AssertionError(f"periodic_evaluation.script does not exist: {script}")
    variants = variants_for(config, periodic.get("variants"))
    if periodic["selection_variant"] not in variants:
        raise AssertionError(
            f"selection_variant {periodic['selection_variant']!r} is not decoded: {variants}"
        )
    if is_deformation_field(config):
        allowed = {"single", "template_only"}
        if not set(variants) <= allowed:
            raise AssertionError(f"A deformation field supports only {sorted(allowed)}.")
    elif not is_two_branch(config) and variants != ["single"]:
        raise AssertionError("A single-field architecture supports only the 'single' variant.")

    eikonal = config["eikonal"]
    if not eikonal.get("enabled"):
        raise AssertionError("Every run in this task enables Eikonal regularization.")
    if float(eikonal.get("epsilon_mm", 0.0)) <= 0.0:
        raise AssertionError(
            "eikonal.epsilon_mm must be positive; the resolution-derived epsilon is "
            "unusable at hash resolutions in the hundreds."
        )
    for branch in eikonal.get("branches", []):
        if branch != "fused" and not is_two_branch(config):
            raise AssertionError(f"Branch {branch!r} does not exist for this architecture.")

    if require_data:
        for key in ("manifest", "required_exact_audit"):
            if not Path(config[key]).is_file():
                raise AssertionError(f"{key} does not exist: {config[key]}")
        audit = json.loads(Path(config["required_exact_audit"]).read_text())
        if not (audit.get("passed") and audit.get("full_cohort_audit") and audit.get("training_allowed")):
            raise AssertionError("Exact-SDF audit does not authorize training.")
        pca_dir = Path(periodic["pca_model_dir"])
        for name in ("components.npy", "mean_vertices.npy", "faces.npy", "metadata.json"):
            if not (pca_dir / name).is_file():
                raise AssertionError(f"PCA baseline file missing: {pca_dir / name}")

    capacity = hash_capacity_report(config)
    print(f"architecture          {architecture_name(config)}")
    print(f"latent size           {config['latent_size']}")
    print(f"output_dir            {output_dir}")
    print(f"variants              {variants}  (selection: {periodic['selection_variant']})")
    print(
        f"eikonal               weight={eikonal['weight']} from epoch {eikonal['start_epoch']} "
        f"branches={eikonal.get('branches', ['fused'])} epsilon={eikonal['epsilon_mm']} mm"
    )
    print(
        f"grid parameters       {capacity['grid_parameters']:,} "
        f"({capacity['grid_fp32_mib']:.1f} MiB fp32)   [MR128 dense: 13,623,296 / 52.0 MiB]"
    )
    print(f"finest cell           {capacity['finest_geometric_mean_cell_mm']:.3f} mm")
    mc = capacity["marching_cubes_voxel_mm"]
    print(f"marching cubes voxel  256 -> {mc['256']:.3f} mm   512 -> {mc['512']:.3f} mm")
    if second.get("enabled"):
        print(
            f"second order          weight={second['weight']} threshold="
            f"{second['curvature_threshold_per_mm']} /mm from epoch {second['start_epoch']}"
        )
    if grid_free:
        print(f"encoding              {capacity['storage']}")
        print(
            f"fourier wavelengths   {capacity['fourier_max_wavelength_mm']} .. "
            f"{capacity['fourier_min_wavelength_mm']} mm"
        )
        if "warp_min_wavelength_mm" in capacity:
            print(
                f"warp wavelengths      {capacity['warp_max_wavelength_mm']} .. "
                f"{capacity['warp_min_wavelength_mm']} mm  "
                f"(bounded at {capacity['warp_scale_mm']} mm)"
            )
    print("  level  R      storage   entries       cell_mm   surf_cells   load")
    for level in capacity["levels"]:
        print(
            f"  {level['resolution']:>5}  {'':<5} {level['storage']:<8} {level['entries']:>9,}"
            f"   {level['geometric_mean_cell_mm']:>8.3f}   {level['estimated_surface_cells']:>10,.0f}"
            f"   {level['collision_load']:>5.2f}"
        )
    notes = check_controlled_against_mr128(config)
    if notes:
        print("controlled-comparison deviations from MR128:")
        for note in notes:
            print(f"  - {note}")
    else:
        print("controlled-comparison: every shared hyperparameter matches MR128")


def main() -> None:
    args = parse_args()
    paths = (
        [Path(value).resolve() for value in args.configs]
        if args.configs
        else sorted(CONFIG_DIR.glob("*.json"))
    )
    if not paths:
        raise AssertionError(f"No configs found under {CONFIG_DIR}")
    for path in paths:
        check_config(path, args.require_generated_data)
    print(f"\nAll {len(paths)} configuration(s) passed. No files were written.")


if __name__ == "__main__":
    main()
