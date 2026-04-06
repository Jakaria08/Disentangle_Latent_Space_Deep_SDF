import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree
from deep_sdf.metrics.chamfer import compute_chamfer
from deep_sdf.metrics.mesh_normal_consistency import compute_mesh_normal_consistency
from deep_sdf.utils import as_mesh
import point_cloud_utils as pcu


def _kabsch_rigid(src_pts, dst_pts):
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    h = src0.T @ dst0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_mean - src_mean @ r.T
    return r.astype(np.float32), t.astype(np.float32)


def _icp_rigid(
    src_pts,
    dst_pts,
    max_iterations=20,
    trim_quantile=0.90,
    tol=1e-6,
):
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if src.shape[0] < 8 or dst.shape[0] < 8:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    tree = KDTree(dst)
    r_tot = np.eye(3, dtype=np.float32)
    t_tot = np.zeros(3, dtype=np.float32)

    for _ in range(int(max_iterations)):
        src_cur = src @ r_tot.T + t_tot
        distances, idx = tree.query(src_cur, k=1)
        nn = dst[idx]

        if trim_quantile is not None and float(trim_quantile) < 1.0:
            thr = np.quantile(distances, float(trim_quantile))
            keep = distances <= thr
            if int(keep.sum()) >= 16:
                a = src_cur[keep]
                b = nn[keep]
            else:
                a = src_cur
                b = nn
        else:
            a = src_cur
            b = nn

        d_r, d_t = _kabsch_rigid(a, b)
        r_tot = d_r @ r_tot
        t_tot = t_tot @ d_r.T + d_t

        if np.linalg.norm(d_t) < tol and np.linalg.norm(d_r - np.eye(3)) < 1e-4:
            break

    return r_tot, t_tot


def _align_points_for_starmen(
    gen_points,
    gt_points,
    align_mode="centroid",
    max_iterations=20,
    trim_quantile=0.90,
):
    gen = np.asarray(gen_points, dtype=np.float32)
    gt = np.asarray(gt_points, dtype=np.float32)

    mode = str(align_mode).lower()
    if mode in ("none", "off"):
        return gen

    if mode in ("centroid", "translation", "translate"):
        shift = gt.mean(axis=0) - gen.mean(axis=0)
        return gen + shift

    if mode == "rigid":
        r, t = _icp_rigid(
            gen,
            gt,
            max_iterations=int(max_iterations),
            trim_quantile=trim_quantile,
        )
        return gen @ r.T + t

    raise ValueError(
        f"Unknown Starmen alignment mode '{align_mode}'. "
        "Use one of: none, centroid, rigid."
    )


def compute_metric(
    gt_mesh=None,
    gen_mesh=None,
    num_mesh_samples=30000,
    metric="chamfer",
    **kwargs,
):
    if gt_mesh is not None and isinstance(gt_mesh, str):
        gt_mesh = as_mesh(trimesh.load_mesh(gt_mesh))
    if gen_mesh is not None and isinstance(gen_mesh, str):
        gen_mesh = as_mesh(trimesh.load_mesh(gen_mesh))
        
    if gt_mesh is not None and gen_mesh is not None:
        gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
        gt_points_sampled = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
        if metric == "chamfer": 
            return compute_chamfer(gen_points_sampled, gt_points_sampled)
        elif metric in ("chamfer_starmen_aligned", "chamfer_aligned_starmen"):
            align_mode = kwargs.get("align_mode", "centroid")
            align_max_iterations = kwargs.get("align_max_iterations", 20)
            align_trim_quantile = kwargs.get("align_trim_quantile", 0.90)
            gen_points_aligned = _align_points_for_starmen(
                gen_points_sampled,
                gt_points_sampled,
                align_mode=align_mode,
                max_iterations=align_max_iterations,
                trim_quantile=align_trim_quantile,
            )
            return compute_chamfer(gen_points_aligned, gt_points_sampled)
        elif metric == "hausdorff":
            return pcu.hausdorff_distance(gen_points_sampled, gt_points_sampled)
    elif metric == "normal_consistency":
        return compute_mesh_normal_consistency(gen_mesh)
    else:
        return NotImplementedError(f"Chosen metric '{metric}' does not exist.")
