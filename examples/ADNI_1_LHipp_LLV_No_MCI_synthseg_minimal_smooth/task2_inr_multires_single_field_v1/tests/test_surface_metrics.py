#!/usr/bin/env python3
"""Metric tests for the ADNI hippocampus evaluator.

Run from the repository root:

    pytest examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_multires_single_field_v1/tests/ -q

CPU only; touches no bulk data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

TASK_DIR = Path(__file__).resolve().parents[1]
for path in (TASK_DIR / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from periodic_evaluate_multires import (  # noqa: E402
    axis_aligned_face_fraction,
    curvature_tail_per_mm,
    discrete_curvature_per_mm,
    surface_metrics,
)


def sphere(radius: float, subdivisions: int) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def test_curvature_is_tessellation_invariant():
    """The bug this metric replaces: the same shape at two mesh densities.

    A sphere of radius r has mean curvature 1/r everywhere, whatever the
    triangulation.  The raw dihedral angle does not know that -- it halves when
    the edges halve -- which is how a 162k-face reconstruction was reported as
    smoother than the 5.5k-face ground truth it was four times rougher than.
    """
    coarse, fine = sphere(20.0, 2), sphere(20.0, 4)
    assert len(fine.faces) > 10 * len(coarse.faces)

    coarse_curvature = discrete_curvature_per_mm(coarse)
    fine_curvature = discrete_curvature_per_mm(fine)
    assert coarse_curvature == pytest.approx(fine_curvature, rel=0.10)

    # ...whereas the raw angle changes by roughly the edge-length ratio.
    coarse_angle = np.degrees(coarse.face_adjacency_angles).mean()
    fine_angle = np.degrees(fine.face_adjacency_angles).mean()
    assert fine_angle < 0.5 * coarse_angle


def test_curvature_scales_inversely_with_radius():
    """Doubling the radius must halve the metric: it is a curvature.

    The integral form returns exactly 1/r for a sphere of radius r, with no
    triangulation-dependent constant.
    """
    small = discrete_curvature_per_mm(sphere(10.0, 3))
    large = discrete_curvature_per_mm(sphere(20.0, 3))
    assert large == pytest.approx(0.5 * small, rel=0.05)
    assert small == pytest.approx(1.0 / 10.0, rel=0.05)


def test_curvature_rises_with_added_noise():
    """A jittered sphere must read as rougher than the sphere it came from."""
    clean = sphere(20.0, 4)
    noisy = clean.copy()
    generator = np.random.default_rng(0)
    noisy.vertices = np.asarray(noisy.vertices) + generator.normal(
        0.0, 0.15, size=np.asarray(noisy.vertices).shape
    )
    assert discrete_curvature_per_mm(noisy) > 3.0 * discrete_curvature_per_mm(clean)


def test_axis_aligned_fraction_separates_a_cube_from_a_sphere():
    """A cube is entirely lattice-aligned; a sphere is almost not at all."""
    assert axis_aligned_face_fraction(trimesh.creation.box((10, 10, 10))) == pytest.approx(1.0)
    assert axis_aligned_face_fraction(sphere(20.0, 4)) < 0.05


def test_surface_metrics_reports_the_new_keys_and_is_self_consistent():
    truth = sphere(20.0, 4)
    metrics = surface_metrics(truth, truth, 2000, 0)
    for key in (
        "curvature_per_mm_mean",
        "curvature_ratio_to_ground_truth",
        "surface_area_ratio",
        "axis_aligned_face_fraction",
        "predicted_face_count",
        "predicted_median_edge_mm",
    ):
        assert key in metrics
    # Reconstructing a mesh as itself must be exact on every derived ratio.
    assert metrics["curvature_ratio_to_ground_truth"] == pytest.approx(1.0)
    assert metrics["surface_area_ratio"] == pytest.approx(1.0)
    assert metrics["assd_mm"] == pytest.approx(0.0, abs=1.0e-6)


def test_curvature_ratio_flags_an_oversmoothed_prediction():
    """Ratio < 1 means over-smoothed, > 1 means jagged; both must be detectable."""
    truth = sphere(20.0, 4)
    jagged = truth.copy()
    generator = np.random.default_rng(1)
    jagged.vertices = np.asarray(jagged.vertices) + generator.normal(
        0.0, 0.1, size=np.asarray(jagged.vertices).shape
    )
    assert surface_metrics(truth, jagged, 2000, 0)["curvature_ratio_to_ground_truth"] > 1.5
    # A larger sphere is locally flatter, i.e. under-curved relative to truth.
    assert surface_metrics(truth, sphere(40.0, 4), 2000, 0)["curvature_ratio_to_ground_truth"] < 0.75


def test_curvature_ignores_marching_cubes_degenerate_edges():
    """Regression: dividing by near-zero edge lengths gave 4.5e6 /mm.

    Marching cubes emits slivers wherever the isosurface passes close to a
    lattice vertex.  Splitting an edge to create one must not move the metric.
    """
    clean = sphere(20.0, 3)
    before = discrete_curvature_per_mm(clean)

    slivered = clean.copy()
    vertices = np.asarray(slivered.vertices).copy()
    faces = np.asarray(slivered.faces).copy()
    # Duplicate a vertex a nanometre away and rewire one face to it: the new
    # edge has ~zero length, and theta/|e| on it is astronomically large.
    target = int(faces[0][0])
    vertices = np.vstack([vertices, vertices[target] + 1.0e-9])
    faces[0][0] = len(vertices) - 1
    slivered = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    assert discrete_curvature_per_mm(slivered) == pytest.approx(before, rel=0.05)
    assert curvature_tail_per_mm(slivered) < 10.0 * before


# ---------------------------------------------------------------------------
# Finite-difference stencil used by the curvature hinge
# ---------------------------------------------------------------------------
def test_gradient_and_laplacian_match_an_analytic_field():
    """f(x) = x^2 + y^2 + z^2 has grad = 2x and laplacian = 6, exactly.

    The trainer reports |laplacian| as zero for an untrained decoder, which is
    correct -- the output head is initialised near zero, so the field is flat.
    This pins the stencil itself so a genuine zero cannot be confused with a
    broken one.
    """
    import torch

    from multires_common import numerical_gradient_and_laplacian

    latent_size = 4

    class Quadratic(torch.nn.Module):
        def forward(self, value, level_weights=None):
            xyz = value[:, latent_size:]
            return xyz.square().sum(dim=1, keepdim=True)

    torch.manual_seed(0)
    xyz = torch.randn(64, 3, dtype=torch.float64)
    codes = torch.zeros(64, latent_size, dtype=torch.float64)
    epsilon = torch.full((3,), 1.0e-3, dtype=torch.float64)

    gradient, laplacian = numerical_gradient_and_laplacian(
        Quadratic(), codes, xyz, epsilon, []
    )
    assert torch.allclose(gradient, 2.0 * xyz, rtol=1.0e-6, atol=1.0e-6)
    assert torch.allclose(laplacian, torch.full((64,), 6.0, dtype=torch.float64), atol=1.0e-5)


def test_gradient_matches_the_existing_six_point_helper():
    """The 7-point stencil must not change the Eikonal term's gradient."""
    import torch

    from multires_common import numerical_gradient_and_laplacian, numerical_spatial_gradient

    latent_size = 4

    class Wavy(torch.nn.Module):
        def forward(self, value, level_weights=None):
            xyz = value[:, latent_size:]
            return torch.sin(3.0 * xyz).sum(dim=1, keepdim=True)

    torch.manual_seed(1)
    xyz = torch.randn(32, 3, dtype=torch.float64) * 0.3
    codes = torch.zeros(32, latent_size, dtype=torch.float64)
    epsilon = torch.full((3,), 1.0e-3, dtype=torch.float64)

    only = numerical_spatial_gradient(Wavy(), codes, xyz, epsilon, [])
    both, _laplacian = numerical_gradient_and_laplacian(Wavy(), codes, xyz, epsilon, [])
    assert torch.allclose(only, both, rtol=1.0e-10, atol=1.0e-12)


def test_restore_rng_accepts_states_that_came_back_on_the_gpu():
    """Regression: torch.load(map_location=cuda) yields CUDA ByteTensors, but
    set_rng_state requires CPU uint8, so every --resume onto a GPU failed with
    'RNG state must be a torch.ByteTensor'."""
    import random as _random

    import torch

    from train_multires_sdf import restore_rng

    state = {
        "python": _random.getstate(),
        "numpy": np.random.get_state(),
        # Simulate a state that returned with the wrong dtype.
        "torch": torch.get_rng_state().to(torch.int32),
    }
    restore_rng({"rng_state": state})  # must not raise
    if torch.cuda.is_available():
        restore_rng({"rng_state": {"torch": torch.get_rng_state().cuda()}})
        # A checkpoint saved with more devices visible than we have now: the
        # extra states must be dropped, not indexed off the end.
        many = [torch.cuda.get_rng_state(0)] * (torch.cuda.device_count() + 3)
        restore_rng({"rng_state": {"cuda": many}})
        # ...and fewer saved states than devices must also be accepted.
        restore_rng({"rng_state": {"cuda": [torch.cuda.get_rng_state(0)]}})
