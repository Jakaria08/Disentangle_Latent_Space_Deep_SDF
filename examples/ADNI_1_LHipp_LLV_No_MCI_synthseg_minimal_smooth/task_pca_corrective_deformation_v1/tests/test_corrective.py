#!/usr/bin/env python3
"""CPU unit tests for the PCA corrective model; no cohort files are loaded."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import trimesh

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))

import common  # noqa: E402
from model import PCACorrectiveModel, orthogonal_project, sparse_pool  # noqa: E402
from surface_metrics import metrics as surface_metrics  # noqa: E402


class ConstantDecoder(nn.Module):
    def __init__(self, n_vertices: int, value: float):
        super().__init__()
        self.n_vertices = n_vertices
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, coefficients, pca_offset):
        return torch.ones(
            coefficients.shape[0], self.n_vertices, 3, device=coefficients.device
        ) * self.value


def synthetic_contract() -> common.PCAContract:
    generator = np.random.default_rng(1)
    features, latent = 12, 3
    q, _r = np.linalg.qr(generator.normal(size=(features, latent)))
    components = q.T.astype(np.float32)
    return common.PCAContract(
        mean=np.linspace(-1, 1, features, dtype=np.float32),
        components=components,
        coefficient_mean=np.zeros(latent, dtype=np.float32),
        coefficient_std=np.ones(latent, dtype=np.float32),
        residual_rms_mm=0.1,
        coordinate_scale_mm=1.0,
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        rows=[],
        topology_hash="synthetic",
        mean_sha256="synthetic",
        components_sha256="synthetic",
    )


def test_orthogonal_projection_removes_every_retained_pca_component():
    contract = synthetic_contract()
    basis = torch.from_numpy(contract.components)
    delta = torch.randn(5, contract.mean.size)
    projected = orthogonal_project(delta, basis)
    assert torch.max(torch.abs(projected @ basis.T)) < 2e-6


def test_corrected_mesh_reencodes_to_the_original_pca_latent():
    contract = synthetic_contract()
    model = PCACorrectiveModel(contract, ConstantDecoder(contract.n_vertices, 1.0), True)
    coefficients = torch.randn(4, contract.latent_dim)
    details = model.decode_with_details(coefficients)
    recovered = model.pca_encode(details["prediction"])
    assert torch.allclose(recovered, coefficients, atol=2e-6)
    assert torch.max(torch.abs(details["delta"])) > 0


def test_zero_correction_is_exact_pca():
    contract = synthetic_contract()
    model = PCACorrectiveModel(contract, ConstantDecoder(contract.n_vertices, 0.0), True)
    coefficients = torch.randn(2, contract.latent_dim)
    details = model.decode_with_details(coefficients)
    assert torch.equal(details["prediction"], details["pca_mesh"])
    assert torch.count_nonzero(details["delta"]) == 0


def test_sparse_pool_matches_dense_matrix_multiplication():
    indices = torch.tensor([[0, 0, 1], [0, 1, 2]])
    values = torch.tensor([0.5, 0.5, 1.0])
    transform = torch.sparse_coo_tensor(indices, values, (2, 3)).coalesce()
    features = torch.randn(3, 3, 4)
    expected = torch.einsum("ij,bjc->bic", transform.to_dense(), features)
    assert torch.allclose(sparse_pool(features, transform), expected)


def test_surface_metric_is_zero_for_identical_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    result = surface_metrics(mesh.vertices, mesh.vertices, mesh.faces, 1000, 4)
    assert result["assd_mm"] == pytest.approx(0.0, abs=1e-6)
    assert result["hd95_mm"] == pytest.approx(0.0, abs=1e-6)
    assert result["curvature_ratio_to_ground_truth"] == pytest.approx(1.0)


def test_runtime_outputs_outside_bulk_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be below"):
        common.require_bulk_path(tmp_path / "forbidden")
