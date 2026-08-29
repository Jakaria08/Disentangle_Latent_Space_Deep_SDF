#!/usr/bin/env python3
"""Can an SDF on a lattice represent a cortex at all?  No network involved.

Every arm in this task -- dense grids, three Instant-NGP variants, Compact-SDF,
a grid-free Fourier field, a deformed implicit field -- has landed at 60-65% of
true pial area, while a PCA-172 oracle reaches 75-76%.  Latent size, latent
bound, hash capacity, marching-cubes resolution, sampling density, curvature
penalty and the Euclidean frame have each been tested and found non-binding.

This measures the one thing they all share and nobody has isolated: the
**representation**.  It takes the ground-truth mesh, computes its *exact* signed
distance on the same lattice the decoders are sampled on, and runs the same
marching cubes.  There is no network, no latent, no training and no
approximation error -- this is the best any SDF-on-a-lattice method could ever
do for this shape.

Why it might be the whole story: an SDF is single valued.  Where two pial banks
of a sulcus come closer than the lattice spacing -- and in a real cortex they
routinely touch -- no sample between them is positive, so marching cubes cannot
put a surface between them.  Both banks are merged into one blob and their area
disappears.  No amount of network capacity changes that; it is a property of
representing a surface as the zero set of a function sampled on a grid.

    recovers ~99%  -> the representation is fine; the 65% plateau is the
                      network, and capacity/architecture is the place to work.
    recovers ~65%  -> the ceiling IS the representation. Every arm has been
                      measuring the same wall, and the project must move to a
                      mesh/correspondence formulation to go further.

Writes one JSON report; reads everything else.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import trimesh

from hashgrid_common import (
    CALSNIC_TASK,
    load_config,
    load_manifest,
    require_bulk_path,
    write_json,
)


def _calsnic():
    """Load the CALSNIC helpers by path: two modules share these names."""
    modules = {}
    for name in ("calsnic_common", "periodic_evaluate_multires"):
        spec = importlib.util.spec_from_file_location(
            f"calsnic_{name}", CALSNIC_TASK / "scripts" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["calsnic_common"], modules["periodic_evaluate_multires"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-meshes", action="store_true")
    return parser.parse_args()


def lattice(resolution: int) -> np.ndarray:
    """The decoders' sampling lattice: x = i // R**2, spacing 2/(R-1) over [-1,1]."""
    step = 2.0 / (resolution - 1)
    index = np.arange(resolution**3, dtype=np.int64)
    x = index // (resolution * resolution)
    y = (index // resolution) % resolution
    z = index % resolution
    return np.stack((x, y, z), axis=1).astype(np.float64) * step - 1.0


def contact_fraction(
    mesh: trimesh.Trimesh, thresholds_mm, samples: int = 20000, seed: int = 0
) -> dict[str, float]:
    """Fraction of surface area whose opposing wall sits within each threshold.

    Cast a ray *outward* from each face centre and record the distance to the
    first other piece of surface it meets.  In a sulcus that ray crosses the CSF
    gap and lands on the opposite bank, so the hit distance is the wall-to-wall
    separation; on a gyral crown the ray escapes and the gap is infinite.

    This is the number that bounds any SDF-on-a-lattice method.  An SDF is single
    valued: if the gap is narrower than the lattice spacing, no sample between the
    banks is positive, marching cubes cannot insert a surface there, and both
    banks' area is lost.
    """
    generator = np.random.default_rng(seed)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    count = min(int(samples), len(areas))
    chosen = generator.choice(len(areas), size=count, replace=False)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)[chosen]
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[chosen]
    weights = areas[chosen]

    # Offset off the originating triangle so it is not itself the first hit.
    origins = centroids + normals * 1.0e-4
    locations, ray_index, _tri = mesh.ray.intersects_location(
        origins, normals, multiple_hits=False
    )
    gap = np.full(count, np.inf)
    if len(ray_index):
        gap[ray_index] = np.linalg.norm(locations - origins[ray_index], axis=1)
    return {
        f"{t}": float(weights[gap < t].sum() / weights.sum()) for t in thresholds_mm
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    calsnic, evaluator = _calsnic()
    mm = float(config["mm_per_normalized_unit"])
    surface_points = int(config["periodic_evaluation"]["surface_points"])

    output = require_bulk_path(
        args.output
        or Path(config["output_dir"]).parent.parent / "audits" / "sdf_lattice_ceiling"
    )
    rows = [r for r in load_manifest(config["manifest"]) if r["split"] == "train"]
    rows = rows[: int(args.subjects)]

    results = []
    for number, row in enumerate(rows, start=1):
        scan = row["scan_id"]
        print(f"[{number}/{len(rows)}] {scan}", flush=True)
        truth_mm = evaluator.load_mesh(row["mesh_path_mm"])
        sdf_mesh = calsnic.load_sdf_space_mesh(row)

        gaps = contact_fraction(truth_mm, (0.46, 0.92, 1.5, 2.0))
        print(f"    area with an opposing wall within 0.92 mm: {gaps['0.92'] * 100:.1f}%", flush=True)

        entry = {
            "scan_id": scan,
            "ground_truth_area_mm2": float(truth_mm.area),
            "ground_truth_faces": int(len(truth_mm.faces)),
            "area_fraction_with_opposing_wall_within_mm": gaps,
            "by_resolution": {},
        }
        for resolution in args.resolutions:
            started = time.time()
            print(f"    exact SDF on {resolution}^3 lattice ...", flush=True)
            values = calsnic.exact_signed_distance(sdf_mesh, lattice(resolution))
            volume = values.reshape(resolution, resolution, resolution)
            from skimage.measure import marching_cubes

            step = 2.0 / (resolution - 1)
            vertices, faces, _n, _v = marching_cubes(
                volume, level=0.0, spacing=(step, step, step), method="lewiner"
            )
            vertices = vertices - 1.0
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mm_mesh = evaluator.mesh_sdf_to_mm(mesh, row)
            metrics = evaluator.surface_metrics(truth_mm, mm_mesh, surface_points, 0)
            entry["by_resolution"][str(resolution)] = {
                "voxel_mm": float(step * mm),
                "area_mm2": float(mm_mesh.area),
                "area_fraction_of_ground_truth": float(mm_mesh.area / truth_mm.area),
                "faces": int(len(mm_mesh.faces)),
                "assd_mm": float(metrics["assd_mm"]),
                "hd95_mm": float(metrics["hd95_mm"]),
                "fscore_1mm": float(metrics["fscore_1mm"]),
                "connected_components": int(metrics["predicted_connected_components"]),
                "seconds": round(time.time() - started, 1),
            }
            print(
                f"    MC{resolution} ({step * mm:.3f} mm voxel): "
                f"area={mm_mesh.area / truth_mm.area * 100:.1f}%  "
                f"assd={metrics['assd_mm']:.4f}  f1={metrics['fscore_1mm']:.4f}",
                flush=True,
            )
            if args.save_meshes:
                evaluator.atomic_export(
                    mm_mesh, output / "meshes" / f"{scan}_exactsdf_mc{resolution}.ply"
                )
        results.append(entry)

    report = {
        "question": "upper bound on pial area recoverable by ANY SDF-on-a-lattice method",
        "method": "exact ground-truth signed distance on the decode lattice, same marching cubes",
        "no_network": True,
        "subjects": len(results),
        "per_subject": results,
        "mean_area_fraction": {
            str(r): float(
                np.mean(
                    [e["by_resolution"][str(r)]["area_fraction_of_ground_truth"] for e in results]
                )
            )
            for r in args.resolutions
        },
        "mean_contact_area_fraction": {
            key: float(
                np.mean([e["area_fraction_with_opposing_wall_within_mm"][key] for e in results])
            )
            for key in results[0]["area_fraction_with_opposing_wall_within_mm"]
        },
        "reference": {
            "pca_172_oracle_area_fraction": 0.751,
            "arm_D_single_area_fraction": 0.625,
            "best_hash_grid_area_fraction": 0.651,
        },
    }
    write_json(output / "sdf_lattice_ceiling.json", report)
    print("\nmean area fraction by resolution:",
          {k: round(v, 4) for k, v in report["mean_area_fraction"].items()})
    print("mean area with opposing wall within (mm):",
          {k: round(v, 4) for k, v in report["mean_contact_area_fraction"].items()})
    print("report:", output / "sdf_lattice_ceiling.json")


if __name__ == "__main__":
    main()
