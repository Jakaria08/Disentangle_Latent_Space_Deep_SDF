#!/usr/bin/env python3
"""Shared data / template / metric utilities for the hippocampus spiral-AE experiments.

Everything heavy is written below BULK_ROOT; only source and configs live in the repo.
The reconstruction metric here is deliberately identical to the one behind
hippocampus_pca_cocycle_v4/pca/metrics/pca_reconstruction_summary.csv: per-scan RMSE over
*coordinates* (not per-vertex Euclidean distance, which is sqrt(3) larger).
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import uuid
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

BULK_ROOT = Path("/mnt/bulk10tb")
OUTPUT_ROOT = BULK_ROOT / "Deep3DComp" / "ADNI_1_LHipp" / "task_spiral_ae_v1"
CACHE_DIR = OUTPUT_ROOT / "cache"

REPO_TASK_ROOT = _THIS_DIR.parent
COHORT_ROOT = REPO_TASK_ROOT.parent / "hippocampus_pca_cocycle_v4"
MANIFEST_FP = COHORT_ROOT / "metadata" / "hippocampus_qc_keep_manifest.csv"
PCA_REFERENCE_METRICS_FP = COHORT_ROOT / "pca" / "metrics" / "pca_reconstruction_summary.csv"
PCA_REFERENCE_COEFF_FP = COHORT_ROOT / "pca" / "coefficients" / "train_coefficients.npz"

SPLITS = ("train", "val", "test")

# Published PCA-128 validation number this pipeline must reproduce before anything else runs.
PCA128_VAL_REFERENCE = 0.033667
PCA_REPRODUCTION_TOL = 1e-4


# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
def require_bulk_path(value, description: str = "persistent output") -> Path:
    """Refuse to write large artifacts anywhere but the bulk mount (root fs is full)."""
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(BULK_ROOT)
    except ValueError as error:
        raise ValueError(f"{description} must be below {BULK_ROOT}; refusing {path}") from error
    if path == BULK_ROOT:
        raise ValueError(f"{description} cannot be the bulk mount root itself.")
    return path


def makedirs(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path, payload) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def atomic_write_csv(path, rows, fieldnames=None) -> None:
    path = Path(path)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_save_npy(path, array) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# manifest + meshes
# --------------------------------------------------------------------------------------
def read_manifest(manifest_fp=MANIFEST_FP) -> list[dict]:
    with open(manifest_fp, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest {manifest_fp}")
    hashes = {row["correspondence_topology_hash"] for row in rows}
    if len(hashes) != 1:
        raise ValueError(f"Expected one correspondence topology, found {len(hashes)}")
    counts = {row["vertex_count"] for row in rows}
    if len(counts) != 1:
        raise ValueError(f"Expected one vertex count, found {sorted(counts)}")
    return rows


def split_rows(rows, split) -> list[dict]:
    """Manifest order is the canonical scan order for every artifact this task writes."""
    return [row for row in rows if row["split"] == split]


def load_split_vertices(split, rows=None, refresh=False) -> np.ndarray:
    """(N, V, 3) float32 vertices in mm, cached on bulk. Order matches split_rows()."""
    import openmesh as om

    cache_fp = makedirs(CACHE_DIR) / f"adni_{split}_V.npy"
    if cache_fp.exists() and not refresh:
        return np.load(cache_fp)

    rows = rows if rows is not None else read_manifest()
    selected = split_rows(rows, split)
    if not selected:
        raise ValueError(f"No manifest rows for split={split}")
    stack = np.stack(
        [om.read_trimesh(row["mesh_path_mm"]).points().astype(np.float32) for row in selected]
    )
    atomic_save_npy(cache_fp, stack)
    print(f"[cache] {split}: {stack.shape} -> {cache_fp}", flush=True)
    return stack


def load_faces(rows=None) -> np.ndarray:
    import openmesh as om

    cache_fp = makedirs(CACHE_DIR) / "adni_faces.npy"
    if cache_fp.exists():
        return np.load(cache_fp)
    rows = rows if rows is not None else read_manifest()
    faces = om.read_trimesh(rows[0]["mesh_path_mm"]).face_vertex_indices().astype(np.int32)
    atomic_save_npy(cache_fp, faces)
    return faces


def build_template(rows=None, refresh=False):
    """Template = mean training shape on the shared topology (all scans are in correspondence)."""
    v_fp = makedirs(CACHE_DIR) / "template_V.npy"
    f_fp = CACHE_DIR / "template_F.npy"
    if v_fp.exists() and f_fp.exists() and not refresh:
        return np.load(v_fp), np.load(f_fp)

    rows = rows if rows is not None else read_manifest()
    train = load_split_vertices("train", rows=rows)
    vertices = train.mean(axis=0).astype(np.float64)
    faces = load_faces(rows=rows)
    atomic_save_npy(v_fp, vertices)
    atomic_save_npy(f_fp, faces)
    print(f"[template] mean of {len(train)} training shapes: {vertices.shape}", flush=True)
    return vertices, faces


# --------------------------------------------------------------------------------------
# hierarchy / spirals
# --------------------------------------------------------------------------------------
def ds_tag(ds_factors) -> str:
    return "-".join(str(int(f)) for f in ds_factors)


def get_transform(ds_factors, rows=None, refresh=False) -> dict:
    """Quadric-decimation hierarchy, cached per ds_factors so Optuna trials reuse it."""
    tag = ds_tag(ds_factors)
    fp = makedirs(CACHE_DIR) / f"transform_{tag}.pkl"
    if fp.exists() and not refresh:
        with open(fp, "rb") as handle:
            return pickle.load(handle)

    from psbody.mesh import Mesh
    import mesh_sampling

    vertices, faces = build_template(rows=rows)
    _, adj, down, up, hier_f, hier_v = mesh_sampling.generate_transform_matrices(
        Mesh(v=vertices, f=faces), list(int(f) for f in ds_factors)
    )
    payload = {
        "vertices": hier_v,
        "face": hier_f,
        "adj": adj,
        "down_transform": down,
        "up_transform": up,
        "ds_factors": list(int(f) for f in ds_factors),
    }
    tmp = fp.with_name(f".{fp.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle)
    os.replace(tmp, fp)
    print(f"[hierarchy] {tag}: {[v.shape[0] for v in hier_v]} -> {fp}", flush=True)
    return payload


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    import openmesh as om
    import torch
    from generate_spiral_seq import extract_spirals

    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        mesh = om.TriMesh(np.ones([face.max() + 1, 3]), np.array(face))
    return torch.tensor(extract_spirals(mesh, seq_length=int(seq_length), dilation=int(dilation)))


def to_sparse(spmat):
    import torch

    coo = spmat.tocoo()
    return torch.sparse_coo_tensor(
        torch.LongTensor(np.array([coo.row, coo.col])),
        torch.FloatTensor(coo.data),
        torch.Size(coo.shape),
    ).coalesce()


def build_spiral_stack(transform, seq_length, dilation, dynamic_seq_lengths, device):
    """Returns (spiral_indices, dynamic_spiral_indices, down_transforms, up_transforms).

    Sequence lengths are clamped per level: extract_spirals falls back to a KD-tree query for
    seq_length*dilation neighbours, which fails once that exceeds the level's vertex count
    (the coarsest level can be as small as 11). Each conv reads its own seq_length off its
    index tensor, so differing lengths across levels are fine.
    """
    n_levels = len(transform["down_transform"])
    spirals, dynamic = [], []
    for level in range(n_levels):
        face = transform["face"][level]
        verts = transform["vertices"][level]
        n_verts = int(verts.shape[0])

        eff_seq = max(1, min(int(seq_length), n_verts // max(1, int(dilation))))
        spirals.append(preprocess_spiral(face, eff_seq, verts, dilation).to(device))

        dyn_len = int(dynamic_seq_lengths[level]) if dynamic_seq_lengths else 1
        dyn_len = max(1, min(dyn_len, n_verts))
        dynamic.append(preprocess_spiral(face, dyn_len, verts, 1).to(device))
    down = [to_sparse(m).to(device) for m in transform["down_transform"]]
    up = [to_sparse(m).to(device) for m in transform["up_transform"]]
    return spirals, dynamic, down, up


# --------------------------------------------------------------------------------------
# normalization + metrics
# --------------------------------------------------------------------------------------
def train_normalization(train_vertices):
    """Per-vertex mean/std from the training split only."""
    mean = train_vertices.mean(axis=0)
    std = train_vertices.std(axis=0)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _to_numpy(array, dtype=np.float64):
    """Accept numpy or (possibly CUDA) torch tensors."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=dtype)


def vertex_rmse_mm(pred_mm, gt_mm):
    """Per-scan RMSE over coordinates, in mm.

    Matches hippocampus_pca_cocycle_v4 `vertex_rmse_mm`. Reproduces PCA-128 val = 0.033667.
    Accepts numpy or torch; returns a numpy array of length N.
    """
    pred_mm = _to_numpy(pred_mm).reshape(len(pred_mm), -1)
    gt_mm = _to_numpy(gt_mm).reshape(len(gt_mm), -1)
    return np.sqrt(((pred_mm - gt_mm) ** 2).mean(axis=1))


def mesh_volumes(vertices, faces):
    """Signed-volume-of-tetrahedra, |value|, per mesh. vertices (N,V,3), faces (F,3)."""
    vertices = _to_numpy(vertices)
    faces = _to_numpy(faces, dtype=np.int64)
    a = vertices[:, faces[:, 0], :]
    b = vertices[:, faces[:, 1], :]
    c = vertices[:, faces[:, 2], :]
    return np.abs(np.einsum("nfi,nfi->n", a, np.cross(b, c)) / 6.0)


def reconstruction_metrics(pred_mm, gt_mm, faces=None) -> dict:
    rmse = vertex_rmse_mm(pred_mm, gt_mm)
    out = {
        "scans": int(len(rmse)),
        "vertex_rmse_mm_mean": float(rmse.mean()),
        "vertex_rmse_mm_median": float(np.median(rmse)),
        "vertex_rmse_mm_p95": float(np.percentile(rmse, 95)),
    }
    if faces is not None:
        pv = mesh_volumes(pred_mm, faces)
        gv = mesh_volumes(gt_mm, faces)
        rel = np.abs(pv - gv) / np.maximum(gv, 1e-12) * 100.0
        out.update(
            {
                "volume_abs_relative_error_pct_mean": float(rel.mean()),
                "volume_abs_relative_error_pct_median": float(np.median(rel)),
                "volume_abs_relative_error_pct_p95": float(np.percentile(rel, 95)),
            }
        )
    return out


# --------------------------------------------------------------------------------------
# experiment registry
# --------------------------------------------------------------------------------------
EXPERIMENTS = {
    "spiralnet_z128": {"conv_type": "spiral", "latent": 128},
    "spiralnet_z256": {"conv_type": "spiral", "latent": 256},
    "adaptive_z128": {"conv_type": "adaptive", "latent": 128},
    "adaptive_z256": {"conv_type": "adaptive", "latent": 256},
}


def experiment_dirs(exp_name: str, tag: str = "") -> dict:
    """Per-experiment output dirs. `tag` keeps a revised search (e.g. "v2") from overwriting
    an earlier one -- the untagged v1 results stay on disk as a baseline."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {exp_name}; expected one of {sorted(EXPERIMENTS)}")
    key = f"{exp_name}_{tag}" if tag else exp_name
    return {
        "study": makedirs(require_bulk_path(OUTPUT_ROOT / "studies" / key, "study dir")),
        "logs": makedirs(require_bulk_path(OUTPUT_ROOT / "logs" / key, "log dir")),
        "best": makedirs(require_bulk_path(OUTPUT_ROOT / "best" / key, "best dir")),
        "latents": makedirs(require_bulk_path(OUTPUT_ROOT / "latents" / key, "latent dir")),
    }
