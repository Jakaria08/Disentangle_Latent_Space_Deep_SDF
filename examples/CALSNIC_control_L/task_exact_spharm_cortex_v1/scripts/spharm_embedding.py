#!/usr/bin/env python3
"""Topology-only spherical embedding and real-spherical-harmonic basis utilities.

The CALSNIC left-pial meshes share one fixed 40962-vertex / 81920-face topology
across all 203 subjects (verified by ``check_shared_topology.py``): they are
already vertex-corresponded, the same way the ADNI hippocampus/lateral-ventricle
meshes are. That means the mesh-to-sphere parameterization SPHARM needs only has
to be solved once, on the shared face connectivity, and reused unchanged for
every subject: vertex index ``i`` always names the same anatomical location, so
its ``(theta, phi)`` never has to change from one subject to the next.

Embedding method: anchored harmonic (uniform-weight Laplace) map, deliberately
independent of any one subject's folded geometry — everything here is derived
purely from the shared face connectivity.

``40962 = 10*4**6 + 2`` is exactly the vertex count of an order-6 subdivided
icosahedron, and this is confirmed combinatorially, not assumed: there are
exactly 12 valence-5 vertices (every other vertex has valence 6, the signature
of a geodesic icosphere subdivision), and the graph distance between them takes
only the values {0, 64, 128, 192} in a pattern that exactly reproduces the
icosahedron's own adjacency (an "antipodal" vertex at distance 192, five
"edge-adjacent" vertices at distance 64, five "second-ring" vertices at
distance 128 — see ``check_shared_topology.py``, which asserts this). That
match is exact, not approximate, so these 12 vertices can be pinned at the
regular icosahedron's exact closed-form coordinates (colatitude
``arccos(1/sqrt(5))`` for the two five-vertex rings) with the correct
correspondence recovered purely from the graph distances and one-ring cycle
order — no external mesh or reference file needed.

With those 12 well-separated anchors fixed, every other vertex's position is
found by *one* sparse linear solve of the uniform-weight (combinatorial)
Laplace equation in 3D, then normalized onto the unit sphere. This was tried
first with the more commonly cited recipe (Tutte-embed the mesh minus one
point, inverse-stereographic-project the plane) and rejected: with only a
single vertex's 1-ring (6 vertices) as the Tutte boundary against 40950
"interior" vertices, the harmonic solution suffers a severe boundary-layer
effect — nearly the entire mesh collapses to a tiny region near the boundary's
centroid, which came out as an 8-orders-of-magnitude triangle-area distortion
ratio (1.6e8) when measured directly. Spreading the Dirichlet anchors across
12 points that are already evenly distributed over the whole sphere (instead
of clustered at one location) removes that pathology outright: the same
measurement on the anchored solve gives a ratio of 37, with zero flipped or
degenerate triangles, out of the box, no further relaxation needed.

Real spherical harmonics: built from ``scipy.special.sph_harm`` combined into
the standard real form. No SPHARM-PDM/Slicer/pyshtools dependency.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import cho_factor, cho_solve
from scipy.special import sph_harm


# ---------------------------------------------------------------------------
# Combinatorial discovery and placement of the 12 icosahedral anchor vertices.
# ---------------------------------------------------------------------------


def vertex_degrees(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    degree = np.zeros(vertex_count, dtype=np.int64)
    np.add.at(degree, faces.reshape(-1), 1)
    return degree


def directed_edges(faces: np.ndarray) -> np.ndarray:
    """Every directed edge implied by each face's CCW vertex order."""
    faces = np.asarray(faces, dtype=np.int64)
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)


def find_icosahedral_anchors(faces: np.ndarray, vertex_count: int) -> dict[str, np.ndarray]:
    """Locate the 12 valence-5 vertices and recover their icosahedral roles.

    Returns a dict with ``vertex_ids`` (length-12 mesh vertex indices, ordered
    ``[pole, *top_cycle, *bottom_cycle, antipode]``) and ``xyz`` (their exact
    regular-icosahedron unit-sphere coordinates in that same order). Raises if
    the mesh's 12 valence-5 vertices do not exactly reproduce the icosahedron's
    combinatorial structure (see module docstring) — that structure is
    verified here, not assumed.
    """
    degree = vertex_degrees(faces, vertex_count)
    val5 = np.flatnonzero(degree == 5)
    if len(val5) != 12:
        raise ValueError(f"Expected exactly 12 valence-5 vertices, found {len(val5)}.")

    edges = directed_edges(faces)
    adjacency = sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(vertex_count, vertex_count)
    ).tocsr()
    adjacency.data[:] = 1.0
    distance = shortest_path(adjacency, method="D", unweighted=True, indices=val5)
    sub = distance[:, val5]
    if not np.array_equal(np.unique(sub.astype(int)), np.array([0, 64, 128, 192])):
        raise ValueError(
            "Valence-5 vertices do not have the expected icosahedral distance "
            f"pattern {{0, 64, 128, 192}}; found {sorted(set(sub.astype(int).ravel().tolist()))}."
        )
    sub = sub.astype(int)

    pole_local = 0
    top_local = np.flatnonzero(sub[pole_local] == 64)
    antipode_local = int(np.flatnonzero(sub[pole_local] == 192)[0])
    bottom_local = np.flatnonzero(sub[antipode_local] == 64)
    if len(top_local) != 5 or len(bottom_local) != 5:
        raise ValueError("Pole/antipode do not each have exactly five ring neighbours.")

    top_adjacent = sub[np.ix_(top_local, top_local)] == 64
    if not np.array_equal(top_adjacent.sum(axis=1), np.full(5, 2)):
        raise ValueError("Top ring is not a 5-cycle.")
    top_order_local = _walk_five_cycle(top_adjacent)
    top_cycle = top_local[top_order_local]

    cross = sub[np.ix_(top_cycle, bottom_local)] == 64
    if not np.array_equal(cross.sum(axis=1), np.full(5, 2)) or not np.array_equal(
        cross.sum(axis=0), np.full(5, 2)
    ):
        raise ValueError("Top/bottom ring cross-adjacency is not the expected 2-regular pattern.")
    bottom_cycle = []
    for i in range(5):
        shared = set(bottom_local[cross[i]].tolist()) & set(bottom_local[cross[(i + 1) % 5]].tolist())
        if len(shared) != 1:
            raise ValueError("Top/bottom rings do not share a unique vertex between consecutive top vertices.")
        bottom_cycle.append(shared.pop())
    bottom_cycle = np.asarray(bottom_cycle, dtype=np.int64)

    ring_z = 1.0 / np.sqrt(5.0)
    ring_r = np.sqrt(1.0 - ring_z * ring_z)
    xyz = np.empty((12, 3), dtype=np.float64)
    order = np.concatenate([[pole_local], top_cycle, bottom_cycle, [antipode_local]])
    xyz[0] = (0.0, 0.0, 1.0)
    for i in range(5):
        lon = 2.0 * np.pi * i / 5.0
        xyz[1 + i] = (ring_r * np.cos(lon), ring_r * np.sin(lon), ring_z)
    for i in range(5):
        lon = 2.0 * np.pi * i / 5.0 + np.pi / 5.0
        xyz[6 + i] = (ring_r * np.cos(lon), ring_r * np.sin(lon), -ring_z)
    xyz[11] = (0.0, 0.0, -1.0)

    return {"vertex_ids": val5[order], "xyz": xyz}


def _walk_five_cycle(adjacent: np.ndarray) -> np.ndarray:
    """Order 0..4 into a cycle given a 5x5 boolean adjacency (each row: 2 True)."""
    order = [0]
    visited = {0}
    current = 0
    while len(order) < 5:
        candidates = [j for j in range(5) if adjacent[current, j] and j not in visited]
        if not candidates:
            raise ValueError("Ring adjacency did not close into a single 5-cycle.")
        current = candidates[0]
        order.append(current)
        visited.add(current)
    return np.asarray(order, dtype=np.int64)


# ---------------------------------------------------------------------------
# Anchored harmonic solve for every other vertex, then project to the sphere.
# ---------------------------------------------------------------------------


def anchored_harmonic_embedding(faces: np.ndarray, vertex_count: int, anchors: dict[str, np.ndarray]) -> np.ndarray:
    """Solve the uniform-weight Laplace equation in 3D with 12 fixed anchors.

    Returns unit-sphere ``(x, y, z)`` for every vertex, shape
    ``(vertex_count, 3)``. The 12 anchor rows are returned exactly as given
    (already unit-length); every other row is the harmonic solution
    normalized onto the sphere.
    """
    edges = directed_edges(faces)
    a, b = edges[:, 0], edges[:, 1]

    anchor_ids = np.asarray(anchors["vertex_ids"], dtype=np.int64)
    anchor_xyz = np.asarray(anchors["xyz"], dtype=np.float64)
    is_anchor = np.zeros(vertex_count, dtype=bool)
    is_anchor[anchor_ids] = True
    free_mask = ~is_anchor
    free_index = -np.ones(vertex_count, dtype=np.int64)
    free_index[free_mask] = np.arange(int(free_mask.sum()))
    n_free = int(free_mask.sum())

    fixed_xyz = np.zeros((vertex_count, 3), dtype=np.float64)
    fixed_xyz[anchor_ids] = anchor_xyz

    from_free = free_mask[a]
    a_free, b_of_a = a[from_free], b[from_free]
    ia = free_index[a_free]
    diagonal = np.zeros(n_free, dtype=np.float64)
    np.add.at(diagonal, ia, 1.0)  # exactly one entry per distinct neighbour

    to_free = free_mask[b_of_a]
    rows_off, cols_off = ia[to_free], free_index[b_of_a[to_free]]

    rhs = np.zeros((n_free, 3), dtype=np.float64)
    to_anchor = ~to_free
    np.add.at(rhs, ia[to_anchor], fixed_xyz[b_of_a[to_anchor]])

    rows = np.concatenate([np.arange(n_free), rows_off])
    cols = np.concatenate([np.arange(n_free), cols_off])
    vals = np.concatenate([diagonal, -np.ones(len(rows_off))])
    matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(n_free, n_free)).tocsc()
    solved = splu(matrix).solve(rhs)

    xyz = np.zeros((vertex_count, 3), dtype=np.float64)
    xyz[anchor_ids] = anchor_xyz
    xyz[free_mask] = solved
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    if not np.all(norms > 1.0e-6):
        raise ValueError("Harmonic solve produced a near-zero-norm vertex; embedding degenerate.")
    return xyz / norms


def xyz_to_angles(xyz: np.ndarray) -> np.ndarray:
    """Convert unit-sphere ``(x, y, z)`` to ``(theta, phi)``: polar, azimuth."""
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack([theta, phi], axis=1)


def build_spherical_embedding(faces: np.ndarray, vertex_count: int):
    """Run the full anchor-discovery -> harmonic-solve -> angles pipeline once.

    Returns ``(angles, sphere_xyz, anchors)`` where ``angles`` has shape
    ``(vertex_count, 2)`` as ``(theta, phi)`` and ``sphere_xyz`` has shape
    ``(vertex_count, 3)``.
    """
    anchors = find_icosahedral_anchors(faces, vertex_count)
    sphere_xyz = anchored_harmonic_embedding(faces, vertex_count, anchors)
    angles = xyz_to_angles(sphere_xyz)
    return angles, sphere_xyz, anchors


def triangle_signed_areas(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Signed spherical-triangle indicator; consistent sign = no flips."""
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    return np.einsum("ij,ij->i", np.cross(v0, v1), v2)


def triangle_chord_areas(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


# ---------------------------------------------------------------------------
# Real spherical harmonic basis, fit, and reconstruction.
# ---------------------------------------------------------------------------


def degree_order_pairs(degree: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(degree + 1) for m in range(-l, l + 1)]


def real_sh_basis(theta: np.ndarray, phi: np.ndarray, degree: int) -> np.ndarray:
    """Real spherical harmonic basis matrix, shape ``(len(theta), (degree+1)**2)``.

    ``scipy.special.sph_harm(m, l, azimuth, polar)`` uses the physics
    convention (its ``theta`` argument is azimuth, its ``phi`` argument is
    polar/colatitude) — the opposite of the ``(theta, phi)`` = (polar, azimuth)
    convention used everywhere else in this module, so the arguments are
    swapped explicitly below.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    pairs = degree_order_pairs(degree)
    columns = []
    for l, m in pairs:
        complex_y = sph_harm(abs(m), l, phi, theta)  # (azimuth=phi, polar=theta)
        if m == 0:
            columns.append(complex_y.real)
        elif m > 0:
            columns.append(np.sqrt(2.0) * ((-1.0) ** m) * complex_y.real)
        else:
            complex_y_pos = sph_harm(-m, l, phi, theta)
            columns.append(np.sqrt(2.0) * ((-1.0) ** m) * complex_y_pos.imag)
    return np.stack(columns, axis=1)


def fit_coefficients(basis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Ordinary least squares fit; ``values`` shape ``(V, C)``, any ``C``."""
    coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return coefficients


class SharedBasisSolver:
    """Precomputed normal-equations solver for one fixed basis matrix.

    Every subject shares the same ``(theta, phi)`` and therefore the same
    basis matrix ``Phi`` for a given degree; factorizing ``Phi^T Phi`` once and
    reusing it for all 203 subjects turns each subject's fit into one
    ``K x K`` triangular solve instead of a fresh ``V x K`` least squares.
    """

    def __init__(self, basis: np.ndarray):
        self.basis = basis
        gram = basis.T @ basis
        self._cho = cho_factor(gram, lower=True)

    def fit(self, values: np.ndarray) -> np.ndarray:
        rhs = self.basis.T @ values
        return cho_solve(self._cho, rhs)

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.basis @ coefficients
