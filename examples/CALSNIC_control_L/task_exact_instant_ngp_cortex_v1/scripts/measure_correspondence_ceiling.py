#!/usr/bin/env python3
"""Measure the spatial scale at which CALSNIC subjects still agree.

A population-shared feature grid stores one vector per location, and that vector
must serve every subject.  This script measures how well that assumption holds
for cortex: it probes a shell of fixed 3D locations with the exact SDF of many
subjects and reports how much they disagree.

The output sets a hard ceiling on useful grid resolution.  Feature cells finer
than the across-subject SDF spread cannot carry shareable population signal --
they can only fit subject-specific detail, which shows up at reconstruction time
as spurious off-surface zero crossings.

Read-only: reads meshes through the manifest and writes one JSON report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from hashgrid_common import (
    exact_signed_distance,
    load_sdf_space_mesh,
    manifest_mm_per_unit,
    read_manifest,
    require_bulk_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None, help="Defaults to the config's manifest.")
    parser.add_argument("--config", default=None, help="Read manifest and ladder from this config.")
    parser.add_argument("--subjects", type=int, default=16)
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument(
        "--shell-sigma",
        type=float,
        default=0.004,
        help="Normalized jitter off the reference surface, so a shell is probed "
        "rather than only the surface itself (0.004 ~ 0.47 mm).",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        from hashgrid_common import load_config

        config = load_config(args.config)
        manifest = args.manifest or config["manifest"]
        ladder = [int(v) for v in config["network_specs"]["grid_resolutions"]]
        aabb = np.asarray(config["network_specs"]["grid_aabb"], dtype=np.float64)
    else:
        manifest = args.manifest
        if not manifest:
            raise ValueError("Pass --config or --manifest.")
        ladder = [8, 16, 32, 64, 96, 128, 163, 211, 353, 767]
        aabb = np.asarray([[-0.45, -0.85, -0.70], [0.45, 0.85, 0.70]], dtype=np.float64)

    mm_per_unit = manifest_mm_per_unit(manifest)
    rows = [row for row in read_manifest(manifest) if row["split"] == "train"][: args.subjects]
    if len(rows) < 2:
        raise ValueError("At least two training subjects are required.")
    meshes = [load_sdf_space_mesh(row) for row in rows]
    print(f"loaded {len(meshes)} training meshes", flush=True)

    # Shape complexity, for context on why cortex differs from ShapeNet objects.
    area = np.asarray([m.area * mm_per_unit**2 for m in meshes])
    volume = np.asarray([abs(m.volume) * mm_per_unit**3 for m in meshes])
    folding = float(np.mean(area / volume ** (2.0 / 3.0)))

    # Probe a shell around one reference surface, at locations shared by all.
    rng = np.random.default_rng(args.seed)
    points, _ = trimesh.sample.sample_surface(meshes[0], int(args.points), seed=args.seed)
    points = np.asarray(points, dtype=np.float64)
    points = points + rng.normal(0.0, float(args.shell_sigma), points.shape)

    values = np.stack(
        [exact_signed_distance(mesh, points) for mesh in meshes]
    )  # [subject, point]
    spread_mm = values.std(axis=0) * mm_per_unit
    positive_fraction = (values > 0).mean(axis=0)
    contested = float(np.mean((positive_fraction > 0.1) & (positive_fraction < 0.9)))
    any_disagreement = float(np.mean((positive_fraction > 0.0) & (positive_fraction < 1.0)))
    median_spread = float(np.median(spread_mm))

    extent = aabb[1] - aabb[0]
    geometric_extent = float(np.prod(extent) ** (1.0 / 3.0))
    levels = []
    for resolution in ladder:
        cell = geometric_extent / max(1, resolution - 1) * mm_per_unit
        levels.append(
            {
                "resolution": int(resolution),
                "cell_mm": cell,
                "above_correspondence_floor": bool(cell >= median_spread),
            }
        )
    ceiling_resolution = int(round(geometric_extent * mm_per_unit / median_spread)) + 1

    report = {
        "manifest": str(manifest),
        "subjects": len(meshes),
        "scan_ids": [row["scan_id"] for row in rows],
        "points": int(len(points)),
        "shell_sigma_normalized": float(args.shell_sigma),
        "shell_sigma_mm": float(args.shell_sigma) * mm_per_unit,
        "mm_per_normalized_unit": mm_per_unit,
        "surface_area_mm2_mean": float(area.mean()),
        "enclosed_volume_mm3_mean": float(volume.mean()),
        "folding_index_area_over_volume_two_thirds": folding,
        "folding_index_sphere_reference": 4.8360,
        "across_subject_sdf_spread_mm": {
            "median": median_spread,
            "mean": float(spread_mm.mean()),
            "p90": float(np.percentile(spread_mm, 90)),
        },
        "fraction_points_contested_inside_outside": contested,
        "fraction_points_any_sign_disagreement": any_disagreement,
        "implied_resolution_ceiling": ceiling_resolution,
        "ladder": levels,
        "interpretation": (
            "Grid levels whose cell size falls below the across-subject SDF spread "
            "cannot encode shareable population signal; they fit subject-specific "
            "detail that a single shared feature grid cannot represent."
        ),
    }
    destination = require_bulk_path(
        args.output_report
        or "/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_instant_ngp_v1/audits/"
        "correspondence_ceiling.json"
    )
    write_json(destination, report)

    print(f"\nfolding index                     {folding:.2f}  (sphere 4.836)")
    print(f"across-subject SDF spread         median {median_spread:.2f} mm  "
          f"p90 {np.percentile(spread_mm, 90):.2f} mm")
    print(f"points contested inside/outside   {100 * contested:.1f}%")
    print(f"implied resolution ceiling        R ~ {ceiling_resolution}")
    print("\n  R      cell_mm   shareable")
    for level in levels:
        print(f"  {level['resolution']:>4}   {level['cell_mm']:>7.3f}   "
              f"{'yes' if level['above_correspondence_floor'] else 'NO'}")
    print(f"\nreport={destination}")


if __name__ == "__main__":
    main()
