from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from networks.shared_multires_grid_sdf import Decoder, DenseMultiResolutionEncoding


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = (
    REPO_ROOT
    / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task2_inr_multires_single_field_v1/scripts/multires_common.py"
)
SPEC = importlib.util.spec_from_file_location("multires_common_test", COMMON_PATH)
assert SPEC is not None and SPEC.loader is not None
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)
if str(COMMON_PATH.parent) not in sys.path:
    sys.path.insert(0, str(COMMON_PATH.parent))
EVALUATION_PATH = COMMON_PATH.parent / "periodic_evaluate_multires.py"
EVALUATION_SPEC = importlib.util.spec_from_file_location(
    "multires_evaluation_test", EVALUATION_PATH
)
assert EVALUATION_SPEC is not None and EVALUATION_SPEC.loader is not None
EVALUATION = importlib.util.module_from_spec(EVALUATION_SPEC)
EVALUATION_SPEC.loader.exec_module(EVALUATION)


def small_decoder() -> Decoder:
    return Decoder(
        latent_size=8,
        grid_resolutions=(4, 8),
        grid_channels_per_level=2,
        grid_aabb=((-0.8, -0.9, -0.7), (0.8, 0.9, 0.7)),
        hidden_dims=(16, 16, 16),
        latent_skip_layer=2,
        activation="softplus",
    )


def test_dense_encoding_shape_and_boundary_taper() -> None:
    encoding = DenseMultiResolutionEncoding(
        resolutions=(4, 8),
        channels=3,
        grid_aabb=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        taper_width=0.1,
    )
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    value, levels, taper, inside, weights = encoding(xyz, return_levels=True)
    assert value.shape == (3, 6)
    assert len(levels) == 2
    assert taper[0].item() == pytest.approx(1.0)
    assert taper[1].item() == pytest.approx(0.0)
    assert taper[2].item() == pytest.approx(0.0)
    assert inside[:, 0].tolist() == [True, True, False]
    assert torch.allclose(value[1:], torch.zeros_like(value[1:]))
    assert weights.tolist() == [1.0, 1.0]


def test_inactive_levels_are_exactly_zero() -> None:
    encoding = DenseMultiResolutionEncoding(
        resolutions=(4, 8),
        channels=2,
        grid_aabb=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    )
    xyz = torch.rand(11, 3) - 0.5
    _value, levels, _taper, _inside, _weights = encoding(
        xyz, level_weights=[1.0, 0.0], return_levels=True
    )
    assert torch.count_nonzero(levels[1]) == 0
    assert torch.count_nonzero(levels[0]) > 0


def test_decoder_uses_one_field_and_backpropagates_to_latent_and_grids() -> None:
    model = small_decoder()
    latent = torch.randn(19, 8, requires_grad=True)
    xyz = torch.rand(19, 3) - 0.5
    result = model(torch.cat((latent, xyz), dim=1), level_weights=[1.0, 1.0], return_parts=True)
    assert set(result).issuperset({"sdf", "features", "level_features", "roi_weight"})
    assert result["sdf"].shape == (19, 1)
    result["sdf"].square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert all(grid.grad is not None and torch.isfinite(grid.grad).all() for grid in model.grids)


class AnalyticSphere(torch.nn.Module):
    grid_resolutions = (8,)

    def forward(self, input_x, level_weights=None):
        xyz = input_x[:, -3:]
        return torch.linalg.vector_norm(xyz, dim=1, keepdim=True) - 0.5


def test_numerical_gradient_matches_analytic_sphere() -> None:
    model = AnalyticSphere()
    xyz = torch.tensor([[0.4, 0.3, 0.2], [-0.2, 0.5, 0.1]], dtype=torch.float32)
    codes = torch.zeros(len(xyz), 2)
    epsilon = torch.tensor([1.0e-3, 1.0e-3, 1.0e-3])
    actual = COMMON.numerical_spatial_gradient(model, codes, xyz, epsilon, [1.0])
    expected = xyz / torch.linalg.vector_norm(xyz, dim=1, keepdim=True)
    assert torch.allclose(actual, expected, atol=2.0e-4, rtol=2.0e-4)


def test_eikonal_finite_difference_backward_avoids_spatial_double_backward() -> None:
    model = small_decoder()
    xyz = torch.rand(13, 3) - 0.5
    codes = torch.randn(13, 8)
    gradient = COMMON.numerical_spatial_gradient(
        model, codes, xyz, torch.tensor([0.02, 0.02, 0.02]), [1.0, 1.0]
    )
    loss = torch.square(torch.linalg.vector_norm(gradient, dim=1) - 1.0).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_progressive_schedule_and_anisotropic_epsilon() -> None:
    config = {
        "total_epochs": 100,
        "network_specs": {
            "grid_resolutions": [8, 16],
            "grid_aabb": [[-1.0, -2.0, -0.5], [1.0, 2.0, 0.5]],
        },
        "level_schedule": [
            {"resolution": 8, "start_epoch": 1, "end_epoch": 1},
            {"resolution": 16, "start_epoch": 11, "end_epoch": 21},
        ],
        "eikonal": {
            "enabled": True,
            "weight": 0.01,
            "start_epoch": 11,
            "warmup_epochs": 5,
            "epsilon_cell_scale": 1.0,
            "final_epsilon_start_epoch": 91,
            "final_epsilon_cell_scale": 0.5,
        },
    }
    assert COMMON.level_weights_for_epoch(1, config) == [1.0, 0.0]
    assert COMMON.level_weights_for_epoch(16, config)[1] == pytest.approx(0.5)
    assert COMMON.level_weights_for_epoch(21, config) == [1.0, 1.0]
    assert COMMON.effective_eikonal_weight(10, config) == 0.0
    assert COMMON.effective_eikonal_weight(15, config) == pytest.approx(0.01)
    epsilon = COMMON.finite_difference_epsilon(21, config, [1.0, 1.0])
    assert np.allclose(epsilon, np.asarray([2.0, 4.0, 1.0]) / 15.0)


def test_output_guard_rejects_ssd_and_accepts_bulk() -> None:
    with pytest.raises(ValueError, match="refusing SSD path"):
        COMMON.require_bulk_path(REPO_ROOT / "forbidden_training_output")
    accepted = COMMON.require_bulk_path("/mnt/bulk10tb/unit_test_destination")
    assert str(accepted).startswith("/mnt/bulk10tb/")


def test_primary_config_constructs_expected_parameter_budget() -> None:
    config_path = (
        COMMON.TASK_DIR
        / "configs/hippocampus_multires_z256_c2f_noeik.json"
    )
    config = COMMON.load_config(config_path)
    model = COMMON.build_decoder(config, torch.device("cpu"))
    assert model.latent_size == 256
    assert model.grid_resolutions == (8, 16, 24, 32, 48, 64)
    assert sum(parameter.numel() for parameter in model.grids) == 1_695_744
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
    assert config["run_role"] == "primary"
    assert not config["eikonal"]["enabled"]
    assert config["scenes_per_batch"] == 16
    assert config["scenes_per_chunk"] == 16


def test_primary_and_eikonal_configs_are_a_controlled_pair() -> None:
    primary = COMMON.load_config(
        COMMON.TASK_DIR / "configs/hippocampus_multires_z256_c2f_noeik.json"
    )
    eikonal = COMMON.load_config(
        COMMON.TASK_DIR / "configs/hippocampus_multires_z256_c2f_eik001.json"
    )
    ignored = {
        "name",
        "description",
        "run_role",
        "output_dir",
        "eikonal",
        "_config_path",
        "_output_dir",
    }
    for key in set(primary).union(eikonal).difference(ignored):
        assert primary.get(key) == eikonal.get(key), key
    assert eikonal["eikonal"]["enabled"]
    assert eikonal["eikonal"]["weight"] == pytest.approx(0.01)


def test_confidence_interval_resamples_subject_clusters() -> None:
    values = np.asarray([1.0, 1.0, 9.0, 9.0])
    subjects = np.asarray(["a", "a", "b", "b"])
    first = EVALUATION.cluster_bootstrap(values, subjects, seed=42, repeats=500)
    second = EVALUATION.cluster_bootstrap(values, subjects, seed=42, repeats=500)
    assert first == second
    assert first == pytest.approx([1.0, 9.0])
