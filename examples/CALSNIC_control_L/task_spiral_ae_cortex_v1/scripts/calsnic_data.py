#!/usr/bin/env python3
"""CALSNIC left pial surface: data, template, hierarchy, and metrics.

Mirrors spiral_common.py from the ADNI hippocampus task, but with CALSNIC's conventions,
which differ in two ways that will silently corrupt results if ignored:

  * Metric is PER-VERTEX EUCLIDEAN RMSE, then divided by the scan's `scaled_from_mm_scale`
    to get mm (fit_matched_pca.py:106-111). The ADNI hippocampus pipeline uses PER-COORDINATE
    RMSE, which is sqrt(3) smaller. Do not mix them.
  * PCA is fitted in the bbox-centred *scaled* OBJ space (`mesh_path`), not `mesh_path_mm`.

PCA here is rank-limited: 173 training meshes => at most 172 components, so PCA-256 does not
exist and 3.844 mm (test) is the linear ceiling at any latent budget.
"""

from __future__ import annotations

import csv, hashlib, json, os, pickle, sys, uuid
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reused unchanged from the hippocampus task: conv operators, AE, trainer, mesh decimation.
HIPP_SCRIPTS = (Path("/home/jakaria/INR/Deep3DComp/examples/"
                     "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/"
                     "task_spiral_ae_v1/scripts"))
if str(HIPP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HIPP_SCRIPTS))

BULK_ROOT = Path("/mnt/bulk10tb")
OUTPUT_ROOT = BULK_ROOT / "Deep3DComp" / "CALSNIC" / "spiral_ae_cortex_v1"
CACHE_DIR = OUTPUT_ROOT / "cache"

CALSNIC_ROOT = BULK_ROOT / "Deep3DComp" / "CALSNIC" / "control_L_exact_multires_v1"
MANIFEST_FP = CALSNIC_ROOT / "manifests" / "calsnic_control_L_exact.csv"
PCA_REF_DIR = CALSNIC_ROOT / "pca" / "matched_left_train173"
PCA_REF_SUMMARY = PCA_REF_DIR / "reconstruction_summary.csv"
SPHARM_PER_SCAN = (CALSNIC_ROOT / "comparisons" / "spharm_degree32_vs_pca172"
                   / "per_scan_metrics.csv")

SPLITS = ("train", "val", "test")
EXPECTED_VERTICES = 40962
EXPECTED_FACES = 81920
PCA_MAX_RANK = 172

# Published reference (corresponded_vertex_rmse_mm) this pipeline must reproduce.
PCA_REFERENCE = {
    128: {"train": 1.4276788972695058, "val": 3.9534015460848067, "test": 4.017216211192321},
    172: {"train": 3.400470854012213e-05, "val": 3.8637897176752367, "test": 3.8439483993298946},
}
PCA_TOL = 1e-3


def require_bulk_path(value, description="persistent output") -> Path:
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(BULK_ROOT)
    except ValueError as err:
        raise ValueError(f"{description} must be below {BULK_ROOT}; refusing {path}") from err
    return path


def makedirs(p) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


def atomic_save_npy(path, arr):
    path = Path(path); tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npy")
    np.save(tmp, arr); os.replace(tmp, path)


def atomic_write_json(path, payload):
    path = Path(path); tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def atomic_write_csv(path, rows, fieldnames=None):
    path = Path(path); rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r}) if rows else []
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
def read_manifest(fp=MANIFEST_FP) -> list[dict]:
    with open(fp, newline="") as h:
        rows = list(csv.DictReader(h))
    if not rows:
        raise ValueError(f"empty manifest {fp}")
    return rows


def split_rows(rows, split):
    return [r for r in rows if r["split"] == split]


def row_center(row) -> np.ndarray:
    """Per-scan bbox centre stored in the manifest."""
    return np.asarray([float(row[f"mesh_center_{a}"]) for a in "xyz"], dtype=np.float64)


def _load_obj_vertices(path, center=None):
    """Load a scaled-space mesh, re-centred exactly as calsnic_common.load_sdf_space_mesh does.

    The per-scan `mesh_center_*` subtraction is NOT cosmetic: it removes a per-scan translation
    that PCA would otherwise have to spend components modelling. Omitting it reproduces train
    error but inflates val/test by ~1%.
    """
    import trimesh
    m = trimesh.load(path, process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    if center is not None:
        v = v - np.asarray(center, dtype=np.float64)[None, :]
    return v, np.asarray(m.faces, dtype=np.int32)


def topology_gate(rows, n_check=None) -> np.ndarray:
    """Refuse to proceed unless every scan shares one topology. Returns the faces array."""
    check = rows if n_check is None else rows[:n_check]
    faces_ref, hashes, shapes = None, set(), set()
    for r in check:
        v, f = _load_obj_vertices(r["mesh_path"], row_center(r))
        shapes.add((v.shape[0], f.shape[0]))
        hashes.add(hashlib.sha256(f.tobytes()).hexdigest())
        if faces_ref is None:
            faces_ref = f
    if len(hashes) != 1:
        raise ValueError(f"correspondence broken: {len(hashes)} distinct face topologies")
    if shapes != {(EXPECTED_VERTICES, EXPECTED_FACES)}:
        raise ValueError(f"unexpected mesh size {shapes}")
    return faces_ref


def load_split_vertices(split, rows=None, refresh=False) -> np.ndarray:
    """(N, V, 3) float32 in the bbox-centred SCALED space -- the space PCA was fitted in."""
    fp = makedirs(CACHE_DIR) / f"calsnic_{split}_V.npy"
    if fp.exists() and not refresh:
        return np.load(fp)
    rows = rows if rows is not None else read_manifest()
    sel = split_rows(rows, split)
    stack = np.stack([_load_obj_vertices(r["mesh_path"], row_center(r))[0].astype(np.float32)
                      for r in sel])
    atomic_save_npy(fp, stack)
    print(f"[cache] {split}: {stack.shape} -> {fp}", flush=True)
    return stack


def load_faces(rows=None) -> np.ndarray:
    fp = makedirs(CACHE_DIR) / "calsnic_faces.npy"
    if fp.exists():
        return np.load(fp)
    rows = rows if rows is not None else read_manifest()
    faces = _load_obj_vertices(rows[0]["mesh_path"], row_center(rows[0]))[1]
    atomic_save_npy(fp, faces)
    return faces


def scale_factors(rows, split) -> np.ndarray:
    """Per-scan `scaled_from_mm_scale`; error_scaled / scale = error_mm."""
    return np.array([float(r["scaled_from_mm_scale"]) for r in split_rows(rows, split)])


def build_template(rows=None, refresh=False):
    v_fp = makedirs(CACHE_DIR) / "template_V.npy"
    f_fp = CACHE_DIR / "template_F.npy"
    if v_fp.exists() and f_fp.exists() and not refresh:
        return np.load(v_fp), np.load(f_fp)
    rows = rows if rows is not None else read_manifest()
    train = load_split_vertices("train", rows=rows)
    verts = train.mean(axis=0).astype(np.float64)
    faces = load_faces(rows=rows)
    atomic_save_npy(v_fp, verts); atomic_save_npy(f_fp, faces)
    print(f"[template] mean of {len(train)} training shapes: {verts.shape}", flush=True)
    return verts, faces


def ds_tag(ds): return "-".join(str(int(x)) for x in ds)


def get_transform(ds_factors, rows=None, refresh=False) -> dict:
    """Decimation hierarchy. At 40962 vertices this is slow (minutes) -- cached per ds."""
    fp = makedirs(CACHE_DIR) / f"transform_{ds_tag(ds_factors)}.pkl"
    if fp.exists() and not refresh:
        with open(fp, "rb") as h:
            return pickle.load(h)
    from psbody.mesh import Mesh
    import mesh_sampling
    verts, faces = build_template(rows=rows)
    _, adj, down, up, hf, hv = mesh_sampling.generate_transform_matrices(
        Mesh(v=verts, f=faces), [int(x) for x in ds_factors])
    payload = {"vertices": hv, "face": hf, "adj": adj, "down_transform": down,
               "up_transform": up, "ds_factors": [int(x) for x in ds_factors]}
    tmp = fp.with_name(f".{fp.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as h:
        pickle.dump(payload, h)
    os.replace(tmp, fp)
    print(f"[hierarchy] {ds_tag(ds_factors)}: {[v.shape[0] for v in hv]}", flush=True)
    return payload


# --------------------------------------------------------------------------------------
# metrics -- CALSNIC convention
# --------------------------------------------------------------------------------------
def _np(a, dtype=np.float64):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=dtype)


def corresponded_vertex_rmse_mm(pred_scaled, gt_scaled, scales):
    """Per-vertex EUCLIDEAN RMSE in scaled units, divided by each scan's scale -> mm.

    Reproduces fit_matched_pca.py:106-111 exactly. NOT the same as the hippocampus
    per-coordinate convention (which is sqrt(3) smaller).
    """
    pred, gt = _np(pred_scaled), _np(gt_scaled)
    n = len(pred)
    d2 = ((pred.reshape(n, -1, 3) - gt.reshape(n, -1, 3)) ** 2).sum(axis=2)
    rmse_scaled = np.sqrt(d2.mean(axis=1))
    return rmse_scaled / np.asarray(scales, dtype=np.float64)


def surface_metrics_mm(pred_scaled, gt_scaled, faces, scales, n_samples=100000, seed=0):
    """Symmetric surface distances (ASSD, Chamfer-L1, HD95) in mm.

    Own implementation -- validate against the published `method=pca` rows in
    comparisons/spharm_degree32_vs_pca172/per_scan_metrics.csv before trusting cross-method
    comparisons (validate_metrics.py does this).
    """
    import trimesh
    from scipy.spatial import cKDTree

    pred, gt = _np(pred_scaled), _np(gt_scaled)
    faces = _np(faces, np.int64)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(pred)):
        s = float(scales[i])
        mp = trimesh.Trimesh(vertices=pred[i] / s, faces=faces, process=False)
        mg = trimesh.Trimesh(vertices=gt[i] / s, faces=faces, process=False)
        pp, _ = trimesh.sample.sample_surface(mp, n_samples, seed=int(rng.integers(1 << 30)))
        pg, _ = trimesh.sample.sample_surface(mg, n_samples, seed=int(rng.integers(1 << 30)))
        d_pg = cKDTree(pg).query(pp)[0]      # prediction -> gt
        d_gp = cKDTree(pp).query(pg)[0]      # gt -> prediction
        out.append({
            "assd_mm": float((d_pg.mean() + d_gp.mean()) / 2.0),
            "chamfer_l1_mm": float(d_pg.mean() + d_gp.mean()),
            "hd95_mm": float(max(np.quantile(d_pg, 0.95), np.quantile(d_gp, 0.95))),
            "prediction_to_gt_mm": float(d_pg.mean()),
            "gt_to_prediction_mm": float(d_gp.mean()),
        })
    return out


def paired_bootstrap(candidate, reference, n_boot=10000, seed=0):
    """Per-scan paired difference with bootstrap 95% CI -- the convention already used in
    this repo's comparisons/. A win requires the CI to exclude zero."""
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    d = c - r
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    return {
        "count": int(len(d)),
        "mean_candidate_minus_reference": float(d.mean()),
        "median_candidate_minus_reference": float(np.median(d)),
        "fraction_candidate_better": float((d < 0).mean()),
        "bootstrap_mean_95ci": [float(np.quantile(boots, 0.025)),
                                float(np.quantile(boots, 0.975))],
        "significant": bool(np.quantile(boots, 0.975) < 0 or np.quantile(boots, 0.025) > 0),
    }


# --------------------------------------------------------------------------------------
# icosahedral hierarchy
# --------------------------------------------------------------------------------------
ICO_SIZES = [12, 42, 162, 642, 2562, 10242, 40962]   # 10*4^k + 2


def _coarse_faces(fine_faces, n_coarse):
    """Recover the parent triangulation of a 1-to-4 subdivided icosphere.

    Each parent triangle (a,b,c) subdivides into (a,ab,ca), (ab,b,bc), (ca,bc,c) and the
    central (ab,bc,ca). Only the central triangle has all three vertices above n_coarse, so
    the centrals enumerate the parents exactly; the parent's corners are the low-index
    neighbours shared by pairs of its midpoints.
    """
    import collections
    adj = collections.defaultdict(set)
    for a, b, c in fine_faces:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    low_nb = {v: sorted(u for u in adj[v] if u < n_coarse)
              for v in range(n_coarse, len(adj))}
    out = []
    for a, b, c in fine_faces:
        if a >= n_coarse and b >= n_coarse and c >= n_coarse:
            corners = set()
            for m1, m2 in ((a, b), (b, c), (c, a)):
                shared = set(low_nb[m1]) & set(low_nb[m2])
                if len(shared) != 1:
                    raise ValueError("icosphere structure violated: ambiguous parent corner")
                corners |= shared
            if len(corners) != 3:
                raise ValueError("icosphere structure violated: parent is not a triangle")
            out.append(sorted(corners))
    return np.asarray(out, dtype=np.int32)


def icosphere_hierarchy(faces, n_vertices, n_levels):
    """Exact nested hierarchy for an icosphere-ordered mesh.

    Replaces psbody quadric decimation, which took ~4-5 h at 40,962 vertices and does not
    respect the icosahedral structure. Down-sampling is exact vertex selection (the first
    N_coarse vertices ARE the coarse mesh); up-sampling puts each midpoint vertex at the
    mean of its two parents. This is the standard hierarchy for cortical surface CNNs.
    """
    import scipy.sparse as sp
    if n_vertices not in ICO_SIZES:
        raise ValueError(f"{n_vertices} is not an icosphere size {ICO_SIZES}")
    idx = ICO_SIZES.index(n_vertices)
    sizes = [ICO_SIZES[idx - i] for i in range(n_levels + 1)]
    if sizes[-1] < 12:
        raise ValueError(f"{n_levels} levels is too deep for {n_vertices} vertices")

    face_levels, down, up = [np.asarray(faces, dtype=np.int32)], [], []
    import collections
    for lvl in range(n_levels):
        n_f, n_c = sizes[lvl], sizes[lvl + 1]
        f_fine = face_levels[-1]
        face_levels.append(_coarse_faces(f_fine, n_c))

        # down: (n_c, n_f) select the first n_c vertices
        d = sp.coo_matrix((np.ones(n_c), (np.arange(n_c), np.arange(n_c))), shape=(n_c, n_f))
        down.append(d.tocsr())

        # up: (n_f, n_c) identity on parents, 0.5/0.5 on midpoints
        adj = collections.defaultdict(set)
        for a, b, c in f_fine:
            adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
        rows, cols, vals = list(range(n_c)), list(range(n_c)), [1.0] * n_c
        for v in range(n_c, n_f):
            parents = sorted(u for u in adj[v] if u < n_c)
            if len(parents) != 2:
                raise ValueError(f"vertex {v} has {len(parents)} parents, expected 2")
            rows += [v, v]; cols += parents; vals += [0.5, 0.5]
        up.append(sp.coo_matrix((vals, (rows, cols)), shape=(n_f, n_c)).tocsr())
    return sizes, face_levels, down, up


def get_transform_ico(n_levels, rows=None, refresh=False) -> dict:
    """Cached icosahedral hierarchy in the same dict shape mesh_sampling produces."""
    fp = makedirs(CACHE_DIR) / f"transform_ico{n_levels}.pkl"
    if fp.exists() and not refresh:
        with open(fp, "rb") as h:
            return pickle.load(h)
    verts, faces = build_template(rows=rows)
    sizes, face_levels, down, up = icosphere_hierarchy(faces, verts.shape[0], n_levels)
    payload = {"vertices": [verts[:n] for n in sizes], "face": face_levels,
               "down_transform": down, "up_transform": up,
               "ds_factors": [4] * n_levels, "level_sizes": sizes, "hierarchy": "icosphere"}
    tmp = fp.with_name(f".{fp.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as h:
        pickle.dump(payload, h)
    os.replace(tmp, fp)
    print(f"[hierarchy] icosphere {n_levels} levels: {sizes}", flush=True)
    return payload
