#!/usr/bin/env python3
"""Verify every manifest mesh shares one fixed, genus-0, closed topology.

``fit_matched_pca.py`` (in the sibling multires task) already relies on this
implicitly: it hard-fails if any two subjects' ``faces`` arrays differ. This
script makes that assumption explicit and checks it over the full 203-scan
cohort (not a hand-picked sample), and additionally checks watertightness and
genus (Euler number 2), which the spherical embedding also depends on. No
output is written; this is a read-only gate, run before trusting anything
downstream of it.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import trimesh

from spharm_common import DEFAULT_EXACT_MANIFEST, load_mesh, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_EXACT_MANIFEST))
    parser.add_argument(
        "--mesh-field",
        choices=("mesh_path", "mesh_path_mm"),
        default="mesh_path_mm",
        help="Which manifest column to check (default: the millimetre ground-truth mesh).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    reference_faces: np.ndarray | None = None
    reference_vertex_count: int | None = None
    reference_scan_id: str | None = None
    started = time.time()
    for number, row in enumerate(rows, start=1):
        mesh: trimesh.Trimesh = load_mesh(row[args.mesh_field])
        vertex_count = len(mesh.vertices)
        if not mesh.is_watertight:
            raise ValueError(f"{row['scan_id']}: mesh is not watertight.")
        if not mesh.is_winding_consistent:
            raise ValueError(f"{row['scan_id']}: mesh winding is not consistent.")
        if int(mesh.euler_number) != 2:
            raise ValueError(f"{row['scan_id']}: Euler number {mesh.euler_number} != 2 (not genus 0).")
        if reference_faces is None:
            reference_faces = np.asarray(mesh.faces, dtype=np.int64)
            reference_vertex_count = vertex_count
            reference_scan_id = row["scan_id"]
        else:
            if vertex_count != reference_vertex_count:
                raise ValueError(
                    f"{row['scan_id']}: vertex count {vertex_count} != "
                    f"{reference_vertex_count} ({reference_scan_id})."
                )
            if not np.array_equal(np.asarray(mesh.faces, dtype=np.int64), reference_faces):
                raise ValueError(
                    f"{row['scan_id']}: face connectivity differs from {reference_scan_id}; "
                    "these meshes are not vertex-corresponded, and the shared spherical "
                    "embedding approach in this task does not apply."
                )
        if number % 25 == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] checked, elapsed {time.time() - started:.1f}s", flush=True)

    print(
        f"OK: {len(rows)} meshes share {reference_vertex_count} vertices / "
        f"{len(reference_faces)} faces, watertight, genus 0."
    )


if __name__ == "__main__":
    main()
