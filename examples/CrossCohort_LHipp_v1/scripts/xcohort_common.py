#!/usr/bin/env python3
"""Shared utilities for the cross-cohort left-hippocampus reconstruction experiments.

Every cohort's meshes live in one shared vertex space: they were registered to the ADNI
template, so vertex order and face connectivity are identical everywhere.  That is what
makes an ADNI-fitted model applicable to AIBL, OASIS, or CALSNIC without retraining, and
this module refuses to proceed if a manifest violates it.

Three protocols are supported, and the distinction that matters is *where the fitted
parameters come from*:

``internal``   fit on the cohort's own train split, evaluate on its own val/test.
               Establishes each cohort's own ceiling.
``external``   apply the reference (ADNI) model unchanged.  Nothing is fitted on the
               target cohort - not the PCA basis, not the template, not the per-vertex
               normalisation.  Recomputing any of those on the target would quietly turn
               an external validation into a partial refit.
``pooled``     fit on several cohorts' train splits combined, evaluate per cohort.
``loco``       fit on every cohort except one, evaluate on the held-out cohort.

The reconstruction metric is imported from the ADNI spiral task rather than reimplemented,
so numbers here are directly comparable with the existing ADNI results: per-scan RMSE over
*coordinates* in mm.  Per-vertex Euclidean distance is sqrt(3) larger, and is reported
alongside because that is the convention BrainODE uses.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = TASK_ROOT / "configs"
BULK_ROOT = Path("/mnt/bulk10tb/Deep3DComp/CrossCohort_LHipp_v1")

ADNI_TASK_ROOT = REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
SPIRAL_SCRIPTS = ADNI_TASK_ROOT / "task_spiral_ae_v1" / "scripts"
LAMM_SCRIPTS = ADNI_TASK_ROOT / "task_lamm_ae_v1" / "scripts"

SPLITS = ("train", "val", "test")
EUCLIDEAN_FACTOR = float(np.sqrt(3.0))

if str(SPIRAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SPIRAL_SCRIPTS))


def spiral_common():
    """Import the ADNI spiral helpers lazily, so config-only use needs no heavy deps."""
    import spiral_common as sc  # noqa: PLC0415 - deliberate lazy import

    return sc


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    name: str
    role: str
    mesh_root: Path
    already_prepared: bool
    cohort_filter: str
    allowed_diagnoses: tuple[str, ...]
    negative_diagnosis: str
    positive_diagnosis: str
    pooled_eligible: bool
    keep_diagnosis_changers: bool = False
    drop_duplicate_visit_months: bool = False
    restrict_diagnoses: tuple[str, ...] | None = None
    keep_manifest_override: Path | None = None
    pca_model_dir_override: Path | None = None
    note: str = ""

    @property
    def output_root(self) -> Path:
        return BULK_ROOT / self.name

    @property
    def cohort_build_root(self) -> Path:
        """Where phase 1 writes this cohort's QC and cohort manifests."""
        return TASK_ROOT / "cohorts" / self.name

    @property
    def keep_manifest(self) -> Path:
        if self.keep_manifest_override is not None:
            return self.keep_manifest_override
        return (
            self.cohort_build_root
            / "cohort"
            / "hippocampus_pca_cocycle_v4"
            / "metadata"
            / "hippocampus_qc_keep_manifest.csv"
        )


@dataclass(frozen=True)
class Config:
    structure: str
    short_name: str
    reference_cohort: str
    topology: dict[str, Any]
    split_ratios: dict[str, float]
    seed: int
    qc_settings: dict[str, Any]
    cohorts: dict[str, CohortSpec] = field(default_factory=dict)

    def targets(self) -> list[CohortSpec]:
        return [c for c in self.cohorts.values() if c.name != self.reference_cohort]

    def reference(self) -> CohortSpec:
        return self.cohorts[self.reference_cohort]

    def pooled_members(self) -> list[CohortSpec]:
        return [c for c in self.cohorts.values() if c.pooled_eligible]


def _resolve(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path | None = None) -> Config:
    payload = json.loads((path or CONFIG_DIR / "cohorts.json").read_text())
    cohorts: dict[str, CohortSpec] = {}
    for name, raw in payload["cohorts"].items():
        cohorts[name] = CohortSpec(
            name=name,
            role=raw["role"],
            mesh_root=Path(raw["mesh_root"]),
            already_prepared=bool(raw.get("already_prepared", False)),
            cohort_filter=raw.get("cohort_filter", "strict_no_mci"),
            allowed_diagnoses=tuple(raw.get("allowed_diagnoses", ["CN", "AD"])),
            negative_diagnosis=raw.get("negative_diagnosis", "CN"),
            positive_diagnosis=raw.get("positive_diagnosis", "AD"),
            pooled_eligible=bool(raw.get("pooled_eligible", True)),
            keep_diagnosis_changers=bool(raw.get("keep_diagnosis_changers", False)),
            drop_duplicate_visit_months=bool(raw.get("drop_duplicate_visit_months", False)),
            restrict_diagnoses=tuple(raw["restrict_diagnoses"]) if raw.get("restrict_diagnoses") else None,
            keep_manifest_override=_resolve(raw.get("keep_manifest")),
            pca_model_dir_override=_resolve(raw.get("pca_model_dir")),
            note=raw.get("note", ""),
        )
    return Config(
        structure=payload["structure"],
        short_name=payload["short_name"],
        reference_cohort=payload["reference_cohort"],
        topology=payload["topology"],
        split_ratios=payload["split_ratios"],
        seed=int(payload["seed"]),
        qc_settings=payload["qc_settings"],
        cohorts=cohorts,
    )


def load_hyperparameters(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or CONFIG_DIR / "hyperparameters_z128.json").read_text())


# --------------------------------------------------------------------------------------
# manifests and vertices
# --------------------------------------------------------------------------------------


def read_manifest(spec: CohortSpec, config: Config) -> list[dict[str, str]]:
    """Read a cohort keep-manifest and verify it really is in the shared vertex space."""
    path = spec.keep_manifest
    if not path.is_file():
        raise FileNotFoundError(f"{spec.name}: keep manifest is missing, run phase 1 first: {path}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{spec.name}: empty manifest {path}")

    expected_hash = config.topology["correspondence_topology_hash"]
    expected_vertices = str(config.topology["vertex_count"])
    hashes = {row["correspondence_topology_hash"] for row in rows}
    counts = {str(row["vertex_count"]) for row in rows}
    if hashes != {expected_hash}:
        raise ValueError(
            f"{spec.name}: manifest topology hash {sorted(hashes)} does not match the shared "
            f"space {expected_hash}. These meshes cannot be compared with the reference cohort."
        )
    if counts != {expected_vertices}:
        raise ValueError(f"{spec.name}: vertex counts {sorted(counts)} != {expected_vertices}")

    if spec.restrict_diagnoses:
        keep = set(spec.restrict_diagnoses)
        rows = [row for row in rows if row.get("diagnosis") in keep]
        if not rows:
            raise ValueError(f"{spec.name}: no rows left after restricting to {sorted(keep)}")
    missing = [s for s in SPLITS if not any(r["split"] == s for r in rows)]
    if missing:
        raise ValueError(f"{spec.name}: manifest has no rows for split(s) {missing}")
    return rows


def split_rows(rows: Sequence[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def load_vertices(spec: CohortSpec, rows: Sequence[dict[str, str]], split: str, refresh: bool = False) -> np.ndarray:
    """(N, V, 3) float32 vertices in mm for one split, cached per cohort under bulk.

    The cache key includes the cohort name: a shared key is how a cross-cohort study
    silently evaluates one cohort's model on another cohort's cached vertices.
    """
    import trimesh

    selected = split_rows(rows, split)
    if not selected:
        raise ValueError(f"{spec.name}: no rows for split={split}")
    cache_dir = spec.output_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = cache_dir / f"{spec.name}_{split}_V_{len(selected)}.npy"
    if cache_fp.exists() and not refresh:
        return np.load(cache_fp)
    stack = np.stack(
        [
            np.asarray(trimesh.load(row["mesh_path_mm"], process=False).vertices, dtype=np.float32)
            for row in selected
        ]
    )
    tmp = cache_fp.with_suffix(".tmp.npy")
    np.save(tmp, stack)
    os.replace(tmp, cache_fp)
    print(f"[cache] {spec.name}/{split}: {stack.shape} -> {cache_fp}", flush=True)
    return stack


def load_faces(rows: Sequence[dict[str, str]]) -> np.ndarray:
    import trimesh

    return np.asarray(trimesh.load(rows[0]["mesh_path_mm"], process=False).faces, dtype=np.int32)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def vertex_rmse_mm(pred_mm: np.ndarray, gt_mm: np.ndarray) -> np.ndarray:
    """Per-scan RMSE over coordinates, delegated to the ADNI implementation."""
    return spiral_common().vertex_rmse_mm(pred_mm, gt_mm)


def reconstruction_metrics(pred_mm: np.ndarray, gt_mm: np.ndarray, faces: np.ndarray | None = None) -> dict[str, float]:
    metrics = dict(spiral_common().reconstruction_metrics(pred_mm, gt_mm, faces=faces))
    rmse = float(metrics.get("vertex_rmse_mm_mean", np.nan))
    # BrainODE and much of the shape literature report per-vertex Euclidean distance, which
    # is sqrt(3) larger than coordinate RMSE. Carry both so no comparison silently mixes them.
    metrics["vertex_euclidean_mm_mean"] = rmse * EUCLIDEAN_FACTOR
    return metrics


# --------------------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PCAModel:
    mean: np.ndarray          # (V*3,)
    components: np.ndarray    # (k, V*3)
    source: str

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    def truncated(self, k: int) -> "PCAModel":
        if k > self.k:
            raise ValueError(f"requested {k} components but the model holds {self.k}")
        return PCAModel(self.mean, self.components[:k], f"{self.source}:k={k}")

    def encode(self, vertices: np.ndarray) -> np.ndarray:
        flat = vertices.reshape(len(vertices), -1).astype(np.float64)
        return (flat - self.mean) @ self.components.T

    def decode(self, coefficients: np.ndarray, vertex_count: int) -> np.ndarray:
        flat = coefficients @ self.components + self.mean
        return flat.reshape(len(coefficients), vertex_count, 3)

    def reconstruct(self, vertices: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(vertices), vertices.shape[1])


def fit_pca(train_vertices: np.ndarray, k: int) -> PCAModel:
    """Deterministic PCA on flattened vertices; k is capped by the sample/feature limit."""
    flat = train_vertices.reshape(len(train_vertices), -1).astype(np.float64)
    mean = flat.mean(axis=0)
    centred = flat - mean
    max_k = min(k, min(centred.shape))
    # full_matrices=False gives the thin SVD; components are rows of Vt, already orthonormal.
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return PCAModel(mean=mean, components=vt[:max_k], source=f"fitted:n={len(flat)}")


def load_reference_pca(spec: CohortSpec) -> PCAModel:
    """Load the stored ADNI PCA basis (mean + components) as fitted on ADNI's train split."""
    model_dir = spec.pca_model_dir_override
    if model_dir is None or not model_dir.is_dir():
        raise FileNotFoundError(f"{spec.name}: no stored PCA model dir configured ({model_dir})")
    mean = np.load(model_dir / "mean.npy").astype(np.float64).ravel()
    candidates = sorted(model_dir.glob("components_*.npy"))
    if not candidates:
        raise FileNotFoundError(f"{spec.name}: no components_*.npy under {model_dir}")
    components = np.load(candidates[-1]).astype(np.float64)
    if components.shape[1] != mean.shape[0]:
        components = components.T
    if components.shape[1] != mean.shape[0]:
        raise ValueError(f"{spec.name}: components {components.shape} incompatible with mean {mean.shape}")
    return PCAModel(mean=mean, components=components, source=f"reference:{model_dir}")


# --------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------

RESULT_FIELDS = [
    "protocol", "model", "latent_k", "fit_cohorts", "eval_cohort", "split", "n_scans",
    "vertex_rmse_mm_mean", "vertex_rmse_mm_median", "vertex_rmse_mm_p95",
    "vertex_euclidean_mm_mean", "volume_abs_relative_error_pct_mean",
    "normalization_source", "seed", "notes",
]


def result_row(**kwargs: Any) -> dict[str, Any]:
    row = {key: kwargs.get(key, "") for key in RESULT_FIELDS}
    unknown = set(kwargs).difference(RESULT_FIELDS)
    if unknown:
        raise KeyError(f"unknown result field(s): {sorted(unknown)}")
    return row


def write_results(path: Path, rows: Iterable[dict[str, Any]], append: bool = False) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    mode = "a" if append and exists else "w"
    with open(path, mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if mode == "w" or not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[results] {len(rows)} row(s) -> {path}", flush=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"[json] {path}", flush=True)
    return path
