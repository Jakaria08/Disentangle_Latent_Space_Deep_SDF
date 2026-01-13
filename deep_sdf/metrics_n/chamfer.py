#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
import scipy
from scipy.spatial import cKDTree as KDTree
import trimesh
from deep_sdf.utils import scale_to_unit_sphere


def _cotangent_laplacian(verts, faces):
    verts = np.asarray(verts)
    faces = np.asarray(faces)
    if faces.size == 0 or verts.size == 0:
        raise ValueError("Empty mesh provided for Laplacian computation")

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    def cotangent(a, b):
        cross = np.cross(a, b)
        denom = np.linalg.norm(cross, axis=1)
        denom = np.maximum(denom, 1e-12)
        return (a * b).sum(axis=1) / denom

    cot0 = cotangent(v1 - v0, v2 - v0)
    cot1 = cotangent(v0 - v1, v2 - v1)
    cot2 = cotangent(v0 - v2, v1 - v2)

    i = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 2], faces[:, 0], faces[:, 1]])
    j = np.concatenate([faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 1], faces[:, 0]])
    w = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    w_mat = scipy.sparse.coo_matrix((w, (i, j)), shape=(verts.shape[0], verts.shape[0])).tocsr()
    laplacian = scipy.sparse.diags(np.array(w_mat.sum(axis=1)).ravel()) - w_mat

    face_areas = trimesh.triangles.area(verts[faces])
    mass = np.zeros(verts.shape[0], dtype=np.float64)
    for k in range(3):
        np.add.at(mass, faces[:, k], face_areas / 3.0)
    minv = scipy.sparse.diags(1.0 / np.maximum(mass, 1e-12))
    return minv.dot(laplacian)


def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000, curvature_sampling=0.):
    """This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just points, sampled from the surface (see
               compute_metrics.py for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """
    try:
        # compute cotangent laplacian
        Lap = _cotangent_laplacian(np.array(gen_mesh.vertices), np.array(gen_mesh.faces))
        
        # compute mean curvature for vertices. Clip at median
        curvatures = np.linalg.norm(Lap.dot(gen_mesh.vertices), axis=1)
        curvatures = np.clip(curvatures, np.percentile(curvatures, 0.00), np.percentile(curvatures, 50))

        # create face weights proportional to area and mean face curvature
        face_curvatures = curvatures[gen_mesh.faces].mean(axis=1)
        face_areas = trimesh.triangles.area(gen_mesh.triangles)
        face_curvatures = np.interp(face_curvatures,
                                    (face_curvatures.min(), face_curvatures.max()),
                                    (0, 1))
        face_areas = np.interp(face_areas,
                            (face_areas.min(), face_areas.max()),
                            (0, 1))
        weights = curvature_sampling * face_curvatures + (1 - curvature_sampling) * face_areas
        
        # sample points with appropriate weighting
        gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples, face_weight=weights)[0]
    except IndexError as e:
        raise IndexError

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    return compute_chamfer(gen_points_sampled, gt_points_np)


def compute_chamfer(gen_points_sampled, gt_points_sampled) -> float:
    """This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gen_points_sampled: np.array of points sampled from the generated mesh surface.
    gt_points_sampled: np.array of points sampled from the GT mesh surface.
    """
    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_sampled)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_sampled)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return float(gt_to_gen_chamfer + gen_to_gt_chamfer), np.concatenate((one_distances, two_distances), axis=0)


def compute_trimesh_iou():
    # TODO
    pass

def compute_trimesh_emd():
    # TODO
    pass
