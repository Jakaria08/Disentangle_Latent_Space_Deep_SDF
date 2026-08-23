#!/usr/bin/env python3
"""Full 12-metric mm-space evaluation: SPHARM vs PCA (vs INR, if already exported).

Reuses the metric stack unmodified from the sibling multires task
(``periodic_evaluate_multires.py``): the same ``METRICS`` tuple, the same
``surface_metrics`` function (ASSD, Chamfer L1/L2, HD95, F-score@0.5/1/2mm,
normal cosine, high-curvature ASSD, volume error, watertight/genus/manifold
checks), the same ``MatchedPCA`` class, and the same ``cluster_bootstrap``.
This script needs no trained decoder/config — it only reconstructs from PCA
and SPHARM, both of which are closed-form given a manifest row — so unlike
``periodic_evaluate_multires.py`` it takes plain CLI arguments, not a model
config. If a prior ``periodic_evaluate_multires.py`` run's manual-evaluation
directory is passed via ``--inr-evaluation-dir``, its already-exported INR
meshes are loaded and included as a third comparison leg; otherwise the
comparison is SPHARM vs PCA only, which matches this task's approved scope.

Like every test-touching script in this task family, running the ``test``
split requires ``--confirm-test``. Unlike PCA, SPHARM's basis is not fit on
data, so there is no leakage reason for that gate — it exists only so this
script honours the same "test never selects a degree" discipline used
everywhere else here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from spharm_common import (
    DEFAULT_EXACT_MANIFEST,
    DEFAULT_PCA_DIR,
    DEFAULT_SPHARM_DIR,
    load_mesh,
    load_sdf_space_mesh,
    mesh_sdf_to_mm,
    read_manifest,
    require_bulk_path,
)
import spharm_embedding as se

# Reused unmodified from the sibling multires task (see module docstring).
from periodic_evaluate_multires import METRICS, MatchedPCA, cluster_bootstrap, stable_seed, surface_metrics


class MatchedSpharm:
    """Mirrors ``MatchedPCA``'s interface: ``reconstruct(row) -> mm-space mesh``."""

    def __init__(self, directory: Path, degree: int):
        metadata = json.loads((directory / "metadata.json").read_text())
        swept = metadata.get("degrees_swept", [])
        if int(degree) not in swept:
            raise ValueError(f"Degree {degree} was not fit in {directory} (swept degrees: {swept}).")
        self.degree = int(degree)
        angles = np.load(directory / "sphere_angles.npy")
        self.faces = np.load(directory / "faces.npy")
        basis = se.real_sh_basis(angles[:, 0], angles[:, 1], self.degree)
        self.solver = se.SharedBasisSolver(basis)
        self.total_components = 3 * (self.degree + 1) ** 2

    def reconstruct(self, row: dict[str, str]) -> trimesh.Trimesh:
        target = load_sdf_space_mesh(row)
        vertices = np.asarray(target.vertices, dtype=np.float64)
        coefficients = self.solver.fit(vertices)
        reconstructed = self.solver.reconstruct(coefficients)
        sdf_mesh = trimesh.Trimesh(vertices=reconstructed, faces=self.faces, process=False)
        return mesh_sdf_to_mm(sdf_mesh, row)


class MatchedSpharmPCA:
    """SPHARM-then-PCA hybrid: fit_spharm_pca_hybrid.py's spectral pre-smoothing regularizer.

    Same ``reconstruct(row) -> mm-space mesh`` interface as ``MatchedPCA``/``MatchedSpharm``.
    Reconstruction is: fit this scan's own degree-``degree`` SPHARM coefficients (geometric, no
    leakage) -> project onto the train-only PCA basis fit on coefficients (oracle projection,
    identical in kind to how ``MatchedPCA`` projects onto its train-only vertex basis) -> SPHARM
    reconstruct back to vertices -> mm.
    """

    def __init__(self, spharm_dir: Path, hybrid_dir: Path, degree: int, rank: int):
        degree_dir = Path(hybrid_dir) / f"degree_{degree}"
        metadata = json.loads((degree_dir / "metadata.json").read_text())
        self.degree = int(degree)
        self.rank = min(int(rank), int(metadata["achieved_rank"]))
        self.k_per_channel = int(metadata["k_per_channel"])
        angles = np.load(Path(spharm_dir) / "sphere_angles.npy")
        self.faces = np.load(Path(spharm_dir) / "faces.npy")
        basis = se.real_sh_basis(angles[:, 0], angles[:, 1], self.degree)
        self.solver = se.SharedBasisSolver(basis)
        self.mean = np.load(degree_dir / "mean_coefficients.npy")
        self.components = np.asarray(np.load(degree_dir / "components.npy", mmap_mode="r")[: self.rank])
        self.total_components = self.rank

    def reconstruct(self, row: dict[str, str]) -> trimesh.Trimesh:
        target = load_sdf_space_mesh(row)
        vertices = np.asarray(target.vertices, dtype=np.float64)
        coefficients = self.solver.fit(vertices)  # (K, 3)
        flat = coefficients.reshape(-1).astype(np.float32)
        scores = (flat - self.mean) @ self.components.T
        reconstructed_flat = self.mean + scores @ self.components
        reconstructed_coeffs = reconstructed_flat.reshape(self.k_per_channel, 3)
        reconstructed_vertices = self.solver.reconstruct(reconstructed_coeffs)
        sdf_mesh = trimesh.Trimesh(vertices=reconstructed_vertices, faces=self.faces, process=False)
        return mesh_sdf_to_mm(sdf_mesh, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_EXACT_MANIFEST))
    parser.add_argument("--spharm-dir", default=str(DEFAULT_SPHARM_DIR))
    parser.add_argument("--pca-dir", default=str(DEFAULT_PCA_DIR))
    parser.add_argument("--degree", type=int, required=True, help="SPHARM degree to evaluate (must already be fit).")
    parser.add_argument("--pca-components", type=int, default=172)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("val",))
    parser.add_argument("--per-split", type=int, default=None)
    parser.add_argument("--surface-points", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument(
        "--inr-evaluation-dir",
        default=None,
        help="Optional manual_evaluation/<checkpoint> dir from a prior periodic_evaluate_multires.py "
        "run; if given, its exported INR meshes are included as a third comparison leg.",
    )
    parser.add_argument(
        "--hybrid-dir",
        default=str(DEFAULT_SPHARM_DIR.parent / "pca_hybrid"),
        help="Output directory from fit_spharm_pca_hybrid.py.",
    )
    parser.add_argument(
        "--hybrid-degree",
        type=int,
        default=None,
        help="SPHARM smoothing degree of the hybrid leg to evaluate (must already be fit by "
        "fit_spharm_pca_hybrid.py). Omit to skip the hybrid leg entirely.",
    )
    parser.add_argument(
        "--hybrid-rank",
        type=int,
        default=None,
        help="PCA rank of the hybrid leg to evaluate. Required if --hybrid-degree is given.",
    )
    parser.add_argument("--overwrite-meshes", action="store_true")
    return parser.parse_args()


def atomic_export(mesh: trimesh.Trimesh, path: Path) -> None:
    output = require_bulk_path(path, "evaluation mesh")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}")
    mesh.export(temporary)
    os.replace(temporary, output)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def summarize(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"split:{row['split']}/method:{row['method']}"].append(row)
    report = {}
    for name, group in groups.items():
        subject_ids = np.asarray([row["subject_id"] for row in group])
        report[name] = {}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group])
            report[name][metric] = {
                "count": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "subject_bootstrap_mean_95ci": cluster_bootstrap(values, subject_ids, stable_seed(name + metric, seed)),
            }
    return report


def paired_methods(rows: list[dict[str, Any]], seed: int, method_a: str, method_b: str) -> dict[str, Any]:
    """Generalization of periodic_evaluate_multires.paired() to arbitrary method names."""
    lookup = {(row["scan_id"], row["split"], row["method"]): row for row in rows}
    report: dict[str, Any] = {}
    for split in sorted({row["split"] for row in rows}):
        scan_ids = sorted({row["scan_id"] for row in rows if row["split"] == split})
        report[split] = {}
        pairs = [s for s in scan_ids if (s, split, method_a) in lookup and (s, split, method_b) in lookup]
        for metric in METRICS:
            differences = np.asarray(
                [float(lookup[(s, split, method_a)][metric]) - float(lookup[(s, split, method_b)][metric]) for s in pairs]
            )
            ids = np.asarray([lookup[(s, split, method_a)]["subject_id"] for s in pairs])
            higher_is_better = metric.startswith("fscore") or metric == "normal_absolute_cosine"
            report[split][metric] = {
                "count": len(differences),
                f"mean_{method_a}_minus_{method_b}": float(differences.mean()) if len(differences) else float("nan"),
                f"median_{method_a}_minus_{method_b}": float(np.median(differences)) if len(differences) else float("nan"),
                "subject_bootstrap_mean_95ci": cluster_bootstrap(differences, ids, stable_seed(split + metric, seed)),
                f"fraction_{method_a}_better": (
                    float(np.mean(differences > 0.0)) if higher_is_better else float(np.mean(differences < 0.0))
                )
                if len(differences)
                else float("nan"),
            }
    return report


def load_inr_leg(inr_dir: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    """scan_id -> exported INR mesh path, from a prior periodic_evaluate_multires.py run."""
    mapping: dict[str, Path] = {}
    for split in splits:
        directory = inr_dir / "meshes" / "inr" / split
        if not directory.is_dir():
            continue
        for path in directory.glob("*.ply"):
            mapping[path.stem] = path
    return mapping


def main() -> None:
    args = parse_args()
    if "test" in args.splits and not args.confirm_test:
        raise PermissionError("Test evaluation is locked; add --confirm-test after model/degree selection.")
    if args.hybrid_degree is not None and args.hybrid_rank is None:
        raise ValueError("--hybrid-rank is required when --hybrid-degree is given.")
    default_name = f"spharm_degree{args.degree}_vs_pca{args.pca_components}"
    if args.hybrid_degree is not None:
        default_name += f"_vs_hybrid_d{args.hybrid_degree}_r{args.hybrid_rank}"
    output = require_bulk_path(args.output_dir or DEFAULT_SPHARM_DIR.parent.parent / "comparisons" / default_name)
    rows = read_manifest(args.manifest)
    selected = []
    for split in args.splits:
        candidates = [row for row in rows if row["split"] == split]
        selected.extend(candidates[: min(args.per_split or len(candidates), len(candidates))])

    pca = MatchedPCA(Path(args.pca_dir), args.pca_components)
    spharm = MatchedSpharm(Path(args.spharm_dir), args.degree)
    hybrid = (
        MatchedSpharmPCA(Path(args.spharm_dir), Path(args.hybrid_dir), args.hybrid_degree, args.hybrid_rank)
        if args.hybrid_degree is not None
        else None
    )
    inr_meshes = load_inr_leg(Path(args.inr_evaluation_dir), tuple(args.splits)) if args.inr_evaluation_dir else {}

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    for number, row in enumerate(selected, start=1):
        try:
            ground_truth = load_mesh(row["mesh_path_mm"])
            common = {
                "scan_id": row["scan_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "diagnosis": row.get("diagnosis", ""),
            }
            seed = stable_seed(row["scan_id"], args.seed)

            spharm_mesh = spharm.reconstruct(row)
            spharm_path = output / "meshes" / "spharm" / row["split"] / f"{row['scan_id']}.ply"
            if args.overwrite_meshes or not spharm_path.is_file():
                atomic_export(spharm_mesh, spharm_path)
            results.append(
                {**common, "method": "spharm", **surface_metrics(ground_truth, spharm_mesh, args.surface_points, seed), "mesh_path": str(spharm_path)}
            )

            pca_mesh = pca.reconstruct(row)
            pca_path = output / "meshes" / "pca" / row["split"] / f"{row['scan_id']}.ply"
            if args.overwrite_meshes or not pca_path.is_file():
                atomic_export(pca_mesh, pca_path)
            results.append(
                {**common, "method": "pca", **surface_metrics(ground_truth, pca_mesh, args.surface_points, seed), "mesh_path": str(pca_path)}
            )

            if hybrid is not None:
                hybrid_mesh = hybrid.reconstruct(row)
                hybrid_path = output / "meshes" / "hybrid" / row["split"] / f"{row['scan_id']}.ply"
                if args.overwrite_meshes or not hybrid_path.is_file():
                    atomic_export(hybrid_mesh, hybrid_path)
                results.append(
                    {**common, "method": "hybrid", **surface_metrics(ground_truth, hybrid_mesh, args.surface_points, seed), "mesh_path": str(hybrid_path)}
                )

            if row["scan_id"] in inr_meshes:
                inr_mesh = load_mesh(inr_meshes[row["scan_id"]])
                results.append(
                    {**common, "method": "inr", **surface_metrics(ground_truth, inr_mesh, args.surface_points, seed), "mesh_path": str(inr_meshes[row["scan_id"]])}
                )
        except Exception as error:  # noqa: BLE001
            failures.append({"scan_id": row["scan_id"], "split": row["split"], "error": repr(error)})
        print(f"[{number}/{len(selected)}] {row['split']} {row['scan_id']}", flush=True)

    write_csv(output / "per_scan_metrics.csv", results)
    write_json(output / "failures.json", failures)

    have_inr = any(row["method"] == "inr" for row in results)
    have_hybrid = hybrid is not None
    report = {
        "coordinate_space": "physical millimetres",
        "spharm_degree": args.degree,
        "spharm_total_components": spharm.total_components,
        "pca_components": args.pca_components,
        "hybrid_smoothing_degree": args.hybrid_degree,
        "hybrid_pca_rank": hybrid.rank if have_hybrid else None,
        "splits": list(args.splits),
        "test_confirmed": bool(args.confirm_test),
        "surface_points": args.surface_points,
        "seconds": time.time() - started,
        "failures": failures,
        "summary": summarize(results, args.seed),
        "paired_spharm_minus_pca": paired_methods(results, args.seed, "spharm", "pca"),
        "paired_spharm_minus_inr": paired_methods(results, args.seed, "spharm", "inr") if have_inr else {},
        "paired_hybrid_minus_pca": paired_methods(results, args.seed, "hybrid", "pca") if have_hybrid else {},
        "paired_hybrid_minus_spharm": paired_methods(results, args.seed, "hybrid", "spharm") if have_hybrid else {},
        "spharm_interpretation": "geometric (non-data-fitted) basis; every scan fit independently, no train/test leakage",
        "pca_interpretation": "train-only basis; each target is projected to its own PCA coefficients (oracle reconstruction)",
        "hybrid_interpretation": "SPHARM spectral pre-smoothing (geometric) then train-only PCA on coefficients "
        "(oracle projection); tests whether pre-smoothing before PCA regularizes away PCA's rank-172 overfitting",
        "test_policy": "manual confirmation required; test never selects a degree, rank, or architecture",
    }
    write_json(output / "summary.json", report)
    print(f"Evaluation complete: rows={len(results)}, failures={len(failures)}, output={output}")


if __name__ == "__main__":
    main()
