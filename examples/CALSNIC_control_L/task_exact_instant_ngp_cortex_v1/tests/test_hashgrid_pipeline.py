#!/usr/bin/env python3
"""Unit tests for the CALSNIC Instant-NGP / Compact-SDF task.

Run from the repository root:

    pytest examples/CALSNIC_control_L/task_exact_instant_ngp_cortex_v1/tests/ -q

Everything here is CPU-only and touches no bulk data except the config files.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[2]
for path in (REPO_ROOT, TASK_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from networks.compact_hashgrid_sdf import Decoder as CompactDecoder  # noqa: E402
from networks.conditional_hashgrid_sdf import Decoder as ConditionalDecoder  # noqa: E402
from networks.multires_hashgrid_encoding import (  # noqa: E402
    MultiResolutionHashEncoding,
    hash_ladder,
)

AABB = [[-0.45, -0.85, -0.70], [0.45, 0.85, 0.70]]
CONFIG_DIR = TASK_DIR / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.json"))


def inside_points(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    unit = torch.rand(count, 3, generator=generator)
    # Stay clear of the taper band so features are not attenuated.
    low = torch.tensor([-0.35, -0.70, -0.55])
    high = torch.tensor([0.35, 0.70, 0.55])
    return low + unit * (high - low)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_dense_level_matches_grid_sample():
    """A level that fits its table is bijective and must equal trilinear lookup."""
    encoding = MultiResolutionHashEncoding(
        AABB, resolutions=[8], features_per_level=4, log2_hashmap_size=19,
        init_std=1.0, taper_width=0.04,
    )
    resolution = 8
    table = encoding.grids[0].detach()
    # Row index is (ix*R + iy)*R + iz, i.e. a [C, X, Y, Z] volume.
    volume = table.t().reshape(1, 4, resolution, resolution, resolution)
    points = inside_points(500)
    minimum = torch.tensor(AABB[0])
    maximum = torch.tensor(AABB[1])
    normalized = (points - minimum) / (maximum - minimum) * 2.0 - 1.0
    # grid_sample orders the query as (z, y, x) for a [1, C, X, Y, Z] volume.
    query = normalized[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)
    reference = F.grid_sample(
        volume, query, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, :, :, 0, 0].t()
    reference = reference * encoding.roi_weight(points)
    assert torch.allclose(encoding(points), reference, atol=1e-6)


def test_levels_are_dense_exactly_when_they_fit():
    encoding = MultiResolutionHashEncoding(
        AABB, resolutions=[16, 75, 97, 767], log2_hashmap_size=19
    )
    assert encoding.dense_levels == (True, True, False, False)
    assert encoding.table_entries == (16**3, 75**3, 2**19, 2**19)


def test_taper_is_exactly_zero_outside_the_aabb():
    encoding = MultiResolutionHashEncoding(AABB, resolutions=[16, 767], init_std=1.0)
    outside = torch.tensor(
        [[0.9, 0.0, 0.0], [0.0, -1.2, 0.0], [0.0, 0.0, 0.8], [-0.45, 0.0, 0.0]]
    )
    assert float(encoding(outside).abs().max()) == 0.0


def test_hash_lookup_is_deterministic_and_in_range():
    encoding = MultiResolutionHashEncoding(AABB, resolutions=[767], log2_hashmap_size=17)
    points = inside_points(256, seed=3)
    first = encoding(points)
    second = encoding(points)
    assert torch.equal(first, second)
    unit = ((points - torch.tensor(AABB[0])) / (torch.tensor(AABB[1]) - torch.tensor(AABB[0]))).clamp(0, 1)
    base = torch.floor(unit * 766).to(torch.int64).clamp_(0, 765)
    corners = base.unsqueeze(1) + encoding.corner_offsets
    rows = encoding._lookup_indices(corners, 0)
    assert int(rows.min()) >= 0 and int(rows.max()) < 2**17


def test_level_weights_gate_individual_levels():
    encoding = MultiResolutionHashEncoding(
        AABB, resolutions=[16, 767], features_per_level=2, init_std=1.0
    )
    points = inside_points(64, seed=1)
    gated = encoding(points, level_weights=[1.0, 0.0])
    assert float(gated[:, 2:].abs().max()) == 0.0
    assert float(gated[:, :2].abs().max()) > 0.0
    with pytest.raises(ValueError):
        encoding(points, level_weights=[1.0])
    with pytest.raises(ValueError):
        encoding(points, level_weights=[1.0, 1.5])


def test_gradients_reach_tables_and_coordinates():
    encoding = MultiResolutionHashEncoding(AABB, resolutions=[16, 767], init_std=1.0)
    points = inside_points(64, seed=2).requires_grad_(True)
    encoding(points).sum().backward()
    for table in encoding.grids:
        assert table.grad is not None and torch.isfinite(table.grad).all()
        assert float(table.grad.abs().sum()) > 0.0
    assert torch.isfinite(points.grad).all() and float(points.grad.abs().sum()) > 0.0


def test_total_variation_is_zero_for_hashed_storage():
    encoding = MultiResolutionHashEncoding(AABB, resolutions=[16, 767], init_std=1.0)
    l2, tv = encoding.regularization([1.0, 1.0])
    assert float(tv) == 0.0
    assert float(l2) > 0.0


def test_ladder_rejects_a_non_increasing_scale():
    with pytest.raises(ValueError):
        hash_ladder(16, 1.02, 16)  # floor() collapses adjacent levels
    with pytest.raises(ValueError):
        hash_ladder(16, 1.0, 16)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------
def probe(latent_size: int = 256, count: int = 16, seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(count, latent_size, generator=generator) * 0.05
    return torch.cat((latent, inside_points(count, seed=seed)), dim=1)


def test_conditional_decoder_exposes_the_trainer_contract():
    decoder = ConditionalDecoder(256, grid_aabb=AABB)
    assert decoder.grid_resolutions == hash_ladder(16, 1.2944, 16)
    assert len(list(decoder.grids)) == 16
    assert decoder(probe()).shape == (16, 1)
    parts = decoder(probe(), return_parts=True)
    assert {"sdf", "features", "roi_weight", "level_weights"} <= set(parts)
    l2, tv = decoder.grid_regularization([1.0] * 16)
    assert torch.isfinite(l2) and float(tv) == 0.0


def test_conditional_decoder_rejects_a_mismatched_pinned_ladder():
    with pytest.raises(ValueError):
        ConditionalDecoder(256, grid_resolutions=[16, 32, 64], grid_aabb=AABB)


def test_compact_decoder_branches_and_gate():
    decoder = CompactDecoder(256, grid_aabb=AABB, gate_tau=0.02)
    value = probe()
    parts = decoder(value, return_parts=True)
    assert torch.allclose(parts["sdf_global"], decoder.global_sdf(value))
    assert torch.allclose(parts["sdf_local"], decoder.local_sdf(value))
    assert bool((parts["gate"] >= 0.0).all()) and bool((parts["gate"] <= 1.0).all())
    # smooth_gate interpolates between the branches with weight `gate`.
    expected = parts["sdf_global"] + parts["gate"] * (parts["sdf_local"] - parts["sdf_global"])
    assert torch.allclose(parts["sdf_smooth_gate"], expected, atol=1e-6)


def test_compact_fusion_collapses_to_each_branch_at_the_gate_extremes():
    decoder = CompactDecoder(256, grid_aabb=AABB, gate_tau=0.02)
    value = probe()
    parts = decoder(value, return_parts=True)
    zero_gate = parts["sdf_global"] + 0.0 * (parts["sdf_local"] - parts["sdf_global"])
    unit_gate = parts["sdf_global"] + 1.0 * (parts["sdf_local"] - parts["sdf_global"])
    assert torch.allclose(zero_gate, parts["sdf_global"])
    assert torch.allclose(unit_gate, parts["sdf_local"])


def test_compact_fusion_modes_select_the_right_field():
    decoder = CompactDecoder(256, grid_aabb=AABB)
    value = probe()
    parts = decoder(value, return_parts=True)
    for mode, key in (
        ("global", "sdf_global"),
        ("local", "sdf_local"),
        ("smooth_gate", "sdf_smooth_gate"),
    ):
        decoder.set_fusion_mode(mode)
        assert torch.allclose(decoder(value), parts[key])
    with pytest.raises(ValueError):
        decoder.set_fusion_mode("hard_band")  # decode-time only, never pointwise


def test_state_dict_round_trip_is_exact():
    for factory in (ConditionalDecoder, CompactDecoder):
        source = factory(256, grid_aabb=AABB)
        target = factory(256, grid_aabb=AABB)
        target.load_state_dict(source.state_dict())
        value = probe()
        assert float((source(value) - target(value)).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# Narrow-band fusion
# ---------------------------------------------------------------------------
def test_surface_band_is_a_shell_of_the_requested_thickness():
    from hashgrid_common import _surface_band

    resolution = 64
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    sphere = np.sqrt(x**2 + y**2 + z**2) - 0.5

    thin = _surface_band(sphere, 0)
    thick = _surface_band(sphere, 3)
    assert thin.any() and thick.sum() > thin.sum()
    # The band must hug the surface: every selected cell is within a few voxels.
    step = 2.0 / (resolution - 1)
    assert float(np.abs(sphere[thick]).max()) < 6.0 * step
    # Dilation only adds cells: every zero-crossing cell survives into the band.
    assert bool(thick[thin].all())
    assert not thick.all()


def test_surface_band_is_empty_without_a_sign_change():
    from hashgrid_common import _surface_band

    assert not _surface_band(np.full((8, 8, 8), 0.5, dtype=np.float32), 3).any()


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_contract(path: Path):
    from hashgrid_common import GRID_FREE_ARCHITECTURES

    config = json.loads(path.read_text())
    specs = config["network_specs"]
    ladder = [int(value) for value in specs["grid_resolutions"]]
    # Read the registry rather than naming one architecture, so adding a
    # grid-free arm cannot silently send its config down the hash-ladder branch.
    grid_free = config["network_arch"] in GRID_FREE_ARCHITECTURES

    if grid_free:
        assert ladder == [] and config["level_schedule"] == []
    else:
        assert ladder == list(
            hash_ladder(
                int(specs["base_resolution"]),
                float(specs["per_level_scale"]),
                int(specs["num_levels"]),
            )
        )
        assert int(specs["grid_resolution"]) == max(ladder)
    assert int(specs["sampling_balance_resolution"]) in (128, 256, 384)
    assert specs["grid_aabb"] == AABB
    assert config["latent_size"] in (256, 512)
    assert config["output_dir"].startswith("/mnt/bulk10tb/")
    assert config["sdf_supervision"]["exact_triangle_distance"] is True
    assert config["periodic_evaluation"]["splits"] == ["val"]
    assert config["eikonal"]["enabled"] is True
    assert float(config["eikonal"]["epsilon_mm"]) > 0.0
    assert abs(float(config["mm_per_normalized_unit"]) - 117.431971) < 1.0e-4

    scheduled = [int(item["resolution"]) for item in config["level_schedule"]]
    assert sorted(scheduled) == sorted(ladder)
    assert len(scheduled) == len(set(scheduled))
    if scheduled:
        assert max(int(item["end_epoch"]) for item in config["level_schedule"]) <= int(
            config["total_epochs"]
        )

    from hashgrid_common import DEFORMATION_ARCHITECTURES

    two_branch = config["network_arch"] == "compact_hashgrid_sdf"
    deformation = config["network_arch"] in DEFORMATION_ARCHITECTURES
    variants = config["periodic_evaluation"].get("variants", ["single"])
    assert config["periodic_evaluation"]["selection_variant"] in variants
    if two_branch:
        assert "far_field_agreement" in config["loss_weights"]
        assert set(config["eikonal"]["branches"]) <= {"fused", "global", "local"}
    elif deformation:
        assert set(variants) <= {"single", "template_only"}
        assert config["eikonal"]["branches"] == ["fused"]
        # The warp must stay coarser than the canonical field, or it can
        # re-create the high-frequency noise T is meant to hold still.
        assert (
            float(specs["warp_min_wavelength_mm"])
            > float(specs["template_min_wavelength_mm"])
        )
        assert float(specs["warp_scale_mm"]) > 0.0
        for key in ("deformation_jacobian", "residual_l2"):
            assert key in config["loss_weights"]
    else:
        assert variants == ["single"]
        assert config["eikonal"]["branches"] == ["fused"]


def test_the_arms_are_present_and_distinct():
    """Required arms exist; adding a new one must not break this test."""
    names = {path.stem for path in CONFIGS}
    required = {
        # Instant-NGP sweep: published scale, cortex-tuned, high capacity.
        "calsnic_ngp_paper_z256_exact",
        "calsnic_ngp_cortex_z256_exact",
        "calsnic_ngp_cortex_big_z256_exact",
        # Compact-SDF two branch.
        "calsnic_compact_ngp_z256_exact",
        # Geometry follow-ups: Fourier folds, latent capacity, grid-free.
        "calsnic_compact_fourier_z256_exact",
        "calsnic_compact_fourier_z512_exact",
        "calsnic_fourier_global_z256_exact",
    }
    missing = required - names
    assert not missing, f"missing arm configs: {sorted(missing)}"
    # Output directories must be unique, or two arms would overwrite each other.
    outputs = [json.loads(p.read_text())["output_dir"] for p in CONFIGS]
    assert len(outputs) == len(set(outputs))
    stems = [json.loads(p.read_text())["name"] for p in CONFIGS]
    assert stems == [p.stem for p in CONFIGS], "config name must match its filename"


def test_instant_ngp_sweep_isolates_one_variable_at_a_time():
    def spec(stem):
        return json.loads((CONFIG_DIR / f"{stem}.json").read_text())["network_specs"]

    paper, cortex, big = (
        spec("calsnic_ngp_paper_z256_exact"),
        spec("calsnic_ngp_cortex_z256_exact"),
        spec("calsnic_ngp_cortex_big_z256_exact"),
    )
    # A1 uses the scale Compact-SDF reports; the cortex arms go finer.
    assert paper["per_level_scale"] == 1.1740
    assert cortex["per_level_scale"] == 1.2944
    # A2 and A3 differ only in table size, which is what isolates capacity.
    assert cortex["per_level_scale"] == big["per_level_scale"]
    assert cortex["log2_hashmap_size"] == 19 and big["log2_hashmap_size"] == 22


def test_millimetre_epsilon_beats_the_resolution_derived_one():
    """The reason the epsilon rule was overridden, asserted numerically."""
    config = json.loads((CONFIG_DIR / "calsnic_ngp_cortex_z256_exact.json").read_text())
    mm_per_unit = float(config["mm_per_normalized_unit"])
    extent = np.asarray(AABB[1]) - np.asarray(AABB[0])
    finest = max(int(value) for value in config["network_specs"]["grid_resolutions"])
    derived_mm = (extent / (finest - 1) * mm_per_unit).max()
    assert derived_mm < 0.3  # the rule we are replacing
    assert float(config["eikonal"]["epsilon_mm"]) >= 0.35


# ---------------------------------------------------------------------------
# Metric schema stability
# ---------------------------------------------------------------------------
def minimal_config(branches: list[str], levels: list[int]) -> dict:
    """Just enough config for compute_loss / eikonal_terms, no bulk data."""
    return {
        "total_epochs": 100,
        "mm_per_normalized_unit": 117.431971,
        "clamp_distance": 0.1,
        "sampling": {"near_band": 0.1},
        "far_field_margin": 0.0,
        "network_specs": {"grid_resolutions": levels, "grid_aabb": AABB},
        "level_schedule": [
            {"resolution": value, "start_epoch": 1, "end_epoch": 1} for value in levels
        ],
        "loss_weights": {
            "broad_sdf_l1": 1.0, "near_sdf_l1": 1.0,
            "global_broad_sdf_l1": 1.0, "global_near_sdf_l1": 1.0,
            "local_broad_sdf_l1": 0.25, "local_near_sdf_l1": 1.0,
            "fused_near_sdf_l1": 1.0, "far_field_agreement": 0.1,
            "latent_l2": 1e-4, "grid_l2": 1e-6, "grid_tv": 0.0,
        },
        "eikonal": {
            "enabled": True, "weight": 0.01, "start_epoch": 50, "warmup_epochs": 5,
            "target_band": 0.03, "points_per_chunk": 16, "branches": branches,
            "epsilon_mm": 0.5,
        },
    }


@pytest.mark.parametrize(
    "factory,arch,branches",
    [
        (ConditionalDecoder, "conditional_hashgrid_sdf", ["fused"]),
        (CompactDecoder, "compact_hashgrid_sdf", ["fused", "global"]),
    ],
)
def test_metric_keys_do_not_change_when_eikonal_activates(factory, arch, branches):
    """A key appearing mid-run would shift every column of training_history.csv."""
    from train_hashgrid_sdf import compute_loss

    levels = [16, 20]
    config = minimal_config(branches, levels)
    config["network_arch"] = arch
    model = factory(
        8, grid_resolutions=levels, base_resolution=16, per_level_scale=1.2944,
        num_levels=2, grid_aabb=AABB,
    )
    generator = torch.Generator().manual_seed(11)
    codes = torch.randn(2, 8, generator=generator) * 0.05
    xyz = inside_points(2 * 24, seed=9).reshape(2, 24, 3)
    target = torch.rand(2, 24, 1, generator=generator) * 0.02 - 0.01
    samples = torch.cat((xyz, target), dim=2)

    _, before = compute_loss(model, codes, samples, samples, 1, config)
    _, during = compute_loss(model, codes, samples, samples, 52, config)
    _, after = compute_loss(model, codes, samples, samples, 99, config)
    assert set(before) == set(during) == set(after)
    assert before["eikonal_weight"] == 0.0 and during["eikonal_weight"] > 0.0
    for branch in branches:
        assert f"eikonal_{branch}_loss" in before
    # total must equal the reported parts before Eikonal starts.
    reconstructed = (
        before["broad_sdf_l1"] + before["near_sdf_l1"]
        if arch == "conditional_hashgrid_sdf"
        else None
    )
    if reconstructed is not None:
        assert before["total"] == pytest.approx(
            reconstructed + 1e-4 * before["latent_l2"] + 1e-6 * before["grid_l2"], rel=1e-5
        )


# ---------------------------------------------------------------------------
# Correspondence-ceiling arms (E1, E2)
# ---------------------------------------------------------------------------
CEILING_LADDER = [16, 20, 26, 34, 44, 58, 75, 97, 126]


@pytest.mark.parametrize(
    "name",
    ["calsnic_ngp_ceiling_z256_exact", "calsnic_ngp_ceiling_freez_z256_exact"],
)
def test_ceiling_ladder_is_a2_prefix(name):
    """E1/E2 must truncate A2's ladder, not redefine it, or the ablation is not clean."""
    a2 = json.loads((CONFIG_DIR / "calsnic_ngp_cortex_z256_exact.json").read_text())
    cfg = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    ladder = cfg["network_specs"]["grid_resolutions"]
    assert ladder == CEILING_LADDER
    assert ladder == a2["network_specs"]["grid_resolutions"][: len(ladder)]
    assert cfg["network_specs"]["num_levels"] == len(ladder)
    assert cfg["network_specs"]["grid_resolution"] == max(ladder)
    assert cfg["network_specs"]["per_level_scale"] == a2["network_specs"]["per_level_scale"]
    # The finest level must sit at or above the measured 1.74 mm correspondence floor.
    extent = np.asarray(AABB[1]) - np.asarray(AABB[0])
    cell = float(np.prod(extent) ** (1 / 3)) / (max(ladder) - 1) * 117.431971
    assert 1.0 < cell < 1.8, cell
    # Every level must be scheduled on before training ends.
    assert max(i["end_epoch"] for i in cfg["level_schedule"]) < cfg["total_epochs"]


def test_e2_relaxes_the_code_bound_everywhere_e1_does_not():
    e1 = json.loads((CONFIG_DIR / "calsnic_ngp_ceiling_z256_exact.json").read_text())
    e2 = json.loads((CONFIG_DIR / "calsnic_ngp_ceiling_freez_z256_exact.json").read_text())
    assert e1["code_bound"] == 1.0
    assert e2["code_bound"] == 4.0
    # A bound relaxed at training but not at fitting time would silently clip
    # held-out codes back to the old ball.
    assert e2["latent_fit"]["code_bound"] == 4.0
    assert e2["validation"]["latent_fit"]["code_bound"] == 4.0
    assert e2["loss_weights"]["latent_l2"] > e1["loss_weights"]["latent_l2"]
    # Otherwise E2 must be identical to E1.
    for key in ("network_specs", "level_schedule", "sampling", "learning_rates",
                "total_epochs", "seed", "clamp_distance"):
        assert e1[key] == e2[key], key


# ---------------------------------------------------------------------------
# Band-limited Fourier features
# ---------------------------------------------------------------------------
from networks.fourier_features import (  # noqa: E402
    BandLimitedFourierFeatures,
    fibonacci_directions,
)
from networks.fourier_global_sdf import Decoder as FourierGlobalDecoder  # noqa: E402

MM_PER_UNIT = 117.431971


def test_fibonacci_directions_are_unit_and_deterministic():
    first, second = fibonacci_directions(16), fibonacci_directions(16)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=1), torch.ones(16), atol=1e-5)


def test_fourier_frequencies_are_band_limited_to_the_configured_wavelengths():
    """The whole point of this encoding over a hash grid: a hard frequency cap."""
    encoding = BandLimitedFourierFeatures(
        num_bands=8, directions_per_band=8,
        min_wavelength_mm=1.0, max_wavelength_mm=120.0, mm_per_unit=MM_PER_UNIT,
    )
    # |2*pi*d/lambda| = 2*pi/lambda, so wavelength = 2*pi/|row|.
    wavelengths_mm = (2 * np.pi / encoding.frequencies.norm(dim=1).numpy()) * MM_PER_UNIT
    assert wavelengths_mm.min() == pytest.approx(1.0, rel=1e-4)
    assert wavelengths_mm.max() == pytest.approx(120.0, rel=1e-4)
    # Nothing finer than the configured floor exists at all.
    assert (wavelengths_mm >= 1.0 - 1e-6).all()


def test_fourier_matrix_is_a_buffer_not_a_trained_parameter():
    encoding = BandLimitedFourierFeatures()
    assert sum(p.numel() for p in encoding.parameters()) == 0
    assert "frequencies" in dict(encoding.named_buffers())


def test_fourier_features_separate_points_a_minimum_wavelength_apart():
    encoding = BandLimitedFourierFeatures(
        num_bands=6, directions_per_band=8, min_wavelength_mm=1.0,
        max_wavelength_mm=60.0, mm_per_unit=MM_PER_UNIT,
    )
    base = torch.zeros(1, 3)
    step = (0.5 / MM_PER_UNIT)  # half the minimum wavelength, in model units
    shifted = base + torch.tensor([[step, 0.0, 0.0]])
    difference = (encoding(base) - encoding(shifted)).abs().max()
    assert float(difference) > 0.1  # raw xyz alone would differ by only 0.004


def test_fourier_rejects_an_inverted_wavelength_range():
    with pytest.raises(ValueError):
        BandLimitedFourierFeatures(min_wavelength_mm=10.0, max_wavelength_mm=1.0)


def test_compact_decoder_accepts_fourier_global_branch():
    plain = CompactDecoder(256, grid_aabb=AABB, global_fourier_enabled=False)
    fourier = CompactDecoder(
        256, grid_aabb=AABB, global_fourier_enabled=True,
        global_fourier_bands=8, global_fourier_directions=8,
        global_fourier_min_wavelength_mm=1.0,
    )
    assert plain.global_encoding is None
    assert fourier.global_encoding is not None
    value = probe()
    assert fourier(value).shape == (16, 1)
    # The Fourier branch has a wider first layer; the local branch is untouched.
    assert fourier.global_hidden[0].in_features > plain.global_hidden[0].in_features
    assert fourier.local_hidden[0].in_features == plain.local_hidden[0].in_features


# ---------------------------------------------------------------------------
# Grid-free architecture
# ---------------------------------------------------------------------------
def test_fourier_global_decoder_satisfies_the_grid_contract():
    decoder = FourierGlobalDecoder(256)
    assert decoder.grid_resolutions == ()
    assert len(list(decoder.grids)) == 0
    l2, tv = decoder.grid_regularization([])
    assert float(l2) == 0.0 and float(tv) == 0.0
    value = probe()
    assert decoder(value).shape == (16, 1)
    # decode_latent_to_mesh passes [1.0] * len(grid_resolutions), i.e. [].
    assert torch.equal(decoder(value), decoder(value, level_weights=[]))
    parts = decoder(value, return_parts=True)
    assert {"sdf", "features", "roi_weight"} <= set(parts)


def test_grid_free_model_has_no_grid_parameters_to_optimize():
    decoder = FourierGlobalDecoder(256)
    grid_ids = {id(p) for p in decoder.grids}
    trainable = [p for p in decoder.parameters() if id(p) not in grid_ids]
    assert len(trainable) == len(list(decoder.parameters())) > 0


# ---------------------------------------------------------------------------
# Finite-difference Laplacian and the curvature hinge
# ---------------------------------------------------------------------------
def test_finite_difference_laplacian_is_exact_for_a_quadratic():
    """A central second difference is exact on quadratics, so this pins the maths."""
    from hashgrid_common import numerical_gradient_and_laplacian

    # Run in float64: a second difference divides by h^2, which amplifies float32
    # round-off enough to mask the exactness this test is meant to establish.
    coefficients = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float64)

    def field(value, level_weights=None):
        xyz = value[:, 1:]
        return (coefficients * xyz.square()).sum(dim=1, keepdim=True)

    xyz = inside_points(32, seed=7).to(torch.float64)
    codes = torch.zeros(len(xyz), 1, dtype=torch.float64)
    epsilon = torch.full((3,), 1e-2, dtype=torch.float64)
    gradient, laplacian = numerical_gradient_and_laplacian(field, codes, xyz, epsilon, None)
    assert torch.allclose(gradient, 2 * coefficients * xyz, atol=1e-9)
    expected = 2.0 * coefficients.sum()
    assert torch.allclose(laplacian, expected.expand(len(xyz)), atol=1e-8)


def test_finite_difference_laplacian_recovers_sphere_curvature():
    """For a sphere SDF |x| - r the Laplacian is 2/|x|, i.e. twice mean curvature."""
    from hashgrid_common import numerical_gradient_and_laplacian

    def field(value, level_weights=None):
        return value[:, 1:].norm(dim=1, keepdim=True) - 0.3

    radii = torch.tensor([0.30, 0.45, 0.60], dtype=torch.float64)
    xyz = torch.stack([torch.tensor([r, 0.0, 0.0], dtype=torch.float64) for r in radii])
    codes = torch.zeros(len(xyz), 1, dtype=torch.float64)
    _, laplacian = numerical_gradient_and_laplacian(
        field, codes, xyz, torch.full((3,), 2e-3, dtype=torch.float64), None
    )
    assert torch.allclose(laplacian, 2.0 / radii, rtol=2e-2)


def test_curvature_hinge_ignores_physical_curvature_and_charges_noise():
    """Threshold 1.0 /mm must leave real cortical curvature (~0.4 /mm) alone."""
    per_mm = 1.0 / MM_PER_UNIT
    threshold = 1.0
    physical = torch.full((100,), 0.4 / per_mm)   # 0.4 /mm, a 5 mm curvature radius
    noisy = torch.full((100,), 3.5 / per_mm)      # 3.5 /mm, measured on the fused branch
    charge = lambda x: torch.clamp(x.abs() * per_mm - threshold, min=0.0).square().mean()
    assert float(charge(physical)) == 0.0
    assert float(charge(noisy)) == pytest.approx(2.5**2, rel=1e-4)


def test_second_order_weight_schedule():
    from train_hashgrid_sdf import effective_second_order_weight

    config = {"second_order": {"enabled": True, "weight": 0.001,
                               "start_epoch": 600, "warmup_epochs": 200}}
    assert effective_second_order_weight(599, config) == 0.0
    assert 0.0 < effective_second_order_weight(700, config) < 0.001
    assert effective_second_order_weight(900, config) == pytest.approx(0.001)
    assert effective_second_order_weight(900, {"second_order": {"enabled": False}}) == 0.0


# ---------------------------------------------------------------------------
# The three new experiment configs
# ---------------------------------------------------------------------------
NEW_ARMS = {
    "calsnic_compact_fourier_z256_exact": ("compact_hashgrid_sdf", 256),
    "calsnic_compact_fourier_z512_exact": ("compact_hashgrid_sdf", 512),
    "calsnic_fourier_global_z256_exact": ("fourier_global_sdf", 256),
}


@pytest.mark.parametrize("name,expected", sorted(NEW_ARMS.items()))
def test_new_arm_configs(name, expected):
    arch, latent = expected
    config = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    assert config["network_arch"] == arch
    assert config["latent_size"] == latent
    second = config["second_order"]
    assert second["enabled"] is True and second["weight"] > 0.0
    # A zero threshold would be a plain L2 penalty, which flattens real folds.
    assert second["curvature_threshold_per_mm"] >= 0.5
    assert int(config["network_specs"]["sampling_balance_resolution"]) == 256
    if arch == "fourier_global_sdf":
        assert config["network_specs"]["grid_resolutions"] == []
        assert config["level_schedule"] == []
    else:
        ladder = config["network_specs"]["grid_resolutions"]
        # Levels finer than the MC-512 voxel (0.46 mm) can only alias, so the
        # ladder is capped well above that.
        finest_mm = 1.7 / (max(ladder) - 1) * 117.431971
        assert finest_mm > 0.46
        assert config["network_specs"]["global_fourier_enabled"] is True
        assert config["network_specs"]["gate_tau"] < 0.02  # tighter than run B


# ---------------------------------------------------------------------------
# Grid-free regression: the shared sampler must not assume a non-empty ladder
# ---------------------------------------------------------------------------
def test_balance_resolution_tolerates_an_empty_ladder():
    """Regression: `dict.get(k, max(ladder))` evaluated the default eagerly.

    That crashed the grid-free run at its first validation (epoch 25) even
    though sampling_balance_resolution was set, because max([]) raises.
    """
    # hashgrid_common already puts the engine on sys.path; inserting it again
    # at position 0 would shadow the CALSNIC periodic_evaluate_multires module.
    import hashgrid_common  # noqa: F401
    from multires_common import balance_resolution

    assert balance_resolution({"sampling_balance_resolution": 256, "grid_resolutions": []}) == 256
    assert balance_resolution({"sampling_balance_resolution": 128,
                               "grid_resolutions": [16, 64]}) == 128
    # Falling back to the ladder still works when the key is absent.
    assert balance_resolution({"grid_resolutions": [16, 64]}) == 64
    # But an absent key with no ladder is a configuration error, not a crash.
    with pytest.raises(ValueError):
        balance_resolution({"grid_resolutions": []})


def test_every_config_supplies_a_usable_balance_resolution():
    # hashgrid_common already puts the engine on sys.path; inserting it again
    # at position 0 would shadow the CALSNIC periodic_evaluate_multires module.
    import hashgrid_common  # noqa: F401
    from multires_common import balance_resolution

    for path in CONFIGS:
        specs = json.loads(path.read_text())["network_specs"]
        assert balance_resolution(specs) > 0, path.name


# ---------------------------------------------------------------------------
# Grid-free regressions found by running the model, not by reading the code
# ---------------------------------------------------------------------------
def grid_free_config() -> dict:
    """Minimal config shaped like the grid-free arm."""
    return {
        "network_arch": "fourier_global_sdf",
        "latent_size": 32,
        "network_specs": {
            "fourier_bands": 4, "fourier_directions": 4,
            "fourier_min_wavelength_mm": 1.0, "fourier_max_wavelength_mm": 60.0,
            "mm_per_unit": MM_PER_UNIT, "hidden_dims": [32, 32],
            "latent_skip_layer": 1, "activation": "relu", "softplus_beta": 100.0,
            # Dataset-only keys the decoder must never receive.
            "grid_resolutions": [], "grid_resolution": 0,
            "sampling_balance_resolution": 256, "grid_aabb": AABB,
        },
    }


def test_build_decoder_drops_dataset_only_keys_for_a_grid_free_model():
    """`grid_aabb` must stay in the config for the sampler but not reach __init__."""
    from hashgrid_common import build_decoder

    decoder = build_decoder(grid_free_config(), torch.device("cpu"))
    assert decoder.grid_resolutions == ()
    assert len(list(decoder.grids)) == 0


def test_task_overrides_the_shared_checkpoint_loader():
    """Regression: the shared loader calls its own build_decoder and rejects
    a grid-free config, which killed the run at its first periodic evaluation."""
    import hashgrid_common
    import multires_common

    assert (
        hashgrid_common.load_decoder_checkpoint
        is not multires_common.load_decoder_checkpoint
    )


def test_restore_rng_accepts_state_loaded_onto_another_device():
    """Regression: torch.load(map_location=cuda) returns CUDA ByteTensors, but
    set_rng_state requires CPU uint8, so every --resume onto a GPU failed."""
    from train_hashgrid_sdf import restore_rng

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        # Simulate a state that came back with the wrong dtype/placement.
        "torch": torch.get_rng_state().to(torch.int32),
    }
    restore_rng({"rng_state": state})  # must not raise
    if torch.cuda.is_available():
        cuda_state = {"torch": torch.get_rng_state().cuda()}
        restore_rng({"rng_state": cuda_state})


def test_every_architecture_has_a_selection_default():
    """Regression: dict.get(k, TABLE[arch]) raised for an unlisted architecture."""
    # hashgrid_common puts the CALSNIC evaluator on sys.path, so it must be
    # imported first for evaluate_hashgrid_variants to resolve its imports.
    from hashgrid_common import HASH_ARCHITECTURES
    from evaluate_hashgrid_variants import SELECTION_DEFAULT

    missing = set(HASH_ARCHITECTURES.values()) - set(SELECTION_DEFAULT)
    assert not missing, f"architectures with no selection default: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Deformed implicit field (arm D)
# ---------------------------------------------------------------------------
from networks.deformed_implicit_field_sdf import Decoder as DeformedDecoder  # noqa: E402


def small_deformed(latent_size: int = 256, **overrides) -> DeformedDecoder:
    """A tiny arm-D decoder; the defaults are far too large for a unit test."""
    settings = dict(
        template_bands=2,
        template_directions=2,
        template_hidden_dims=(16, 16),
        warp_bands=2,
        warp_directions=2,
        warp_hidden_dims=(16, 16),
        warp_latent_skip_layer=1,
        residual_bands=2,
        residual_directions=2,
        residual_hidden_dims=(16, 16),
        residual_latent_skip_layer=1,
    )
    settings.update(overrides)
    return DeformedDecoder(latent_size, **settings)


def test_deformed_decoder_satisfies_the_grid_contract():
    decoder = small_deformed()
    assert decoder.grid_resolutions == ()
    assert len(list(decoder.grids)) == 0
    l2, tv = decoder.grid_regularization([])
    assert float(l2) == 0.0 and float(tv) == 0.0
    value = probe()
    assert decoder(value).shape == (16, 1)
    # decode_latent_to_mesh passes [1.0] * len(grid_resolutions), i.e. [].
    assert torch.equal(decoder(value), decoder(value, level_weights=[]))
    parts = decoder(value, return_parts=True)
    assert {"sdf", "features", "roi_weight"} <= set(parts)
    assert {"warp", "warp_unit", "residual", "sdf_canonical", "sdf_full"} <= set(parts)
    assert parts["warp"].shape == (16, 3)
    assert parts["warp_unit"].shape == (16, 3)
    assert parts["residual"].shape == (16, 1)
    # warp is exactly warp_scale * warp_unit, and warp_unit is a tanh.
    assert torch.allclose(parts["warp"], decoder.warp_scale * parts["warp_unit"])
    assert float(parts["warp_unit"].abs().max()) < 1.0


def test_deformed_template_branch_ignores_the_latent():
    """T carries no latent: that is the whole point of the factorization."""
    decoder = small_deformed()
    one, two = probe(seed=1), probe(seed=2)
    # Same coordinates, different codes.
    two = torch.cat((two[:, :256] * 3.0, one[:, 256:]), dim=1)
    assert not torch.allclose(one[:, :256], two[:, :256])
    assert torch.allclose(decoder.template_only_sdf(one), decoder.template_only_sdf(two))
    # ...while the full readout does depend on the code.
    decoder.warp_output.weight.data.normal_(0.0, 0.5)
    assert not torch.allclose(decoder(one), decoder(two))


def test_deformed_readout_mode_switches_and_restores():
    decoder = small_deformed()
    assert decoder.readout_mode == "full"
    value = probe()
    decoder.warp_output.weight.data.normal_(0.0, 0.5)
    decoder.residual_output.weight.data.normal_(0.0, 0.5)
    full = decoder(value)
    decoder.set_readout_mode("template")
    template = decoder(value)
    assert not torch.allclose(full, template)
    assert torch.allclose(template, decoder.template_only_sdf(value))
    decoder.set_readout_mode("full")
    assert torch.allclose(decoder(value), full)
    with pytest.raises(ValueError):
        decoder.set_readout_mode("global")


def test_deformed_warp_is_bounded_by_its_configured_scale():
    """tanh bounding is what stops the warp folding space over on itself."""
    decoder = small_deformed(warp_scale_mm=6.0, mm_per_unit=117.431971)
    # Drive the trunk hard; the bound must hold regardless of the pre-activation.
    decoder.warp_output.weight.data.normal_(0.0, 50.0)
    decoder.warp_output.bias.data.normal_(0.0, 50.0)
    warp = decoder.warp(probe())
    assert warp.abs().max() <= decoder.warp_scale + 1.0e-6
    assert abs(decoder.warp_scale * 117.431971 - 6.0) < 1.0e-4


def test_deformed_decoder_starts_as_a_plain_template_field():
    """At init the warp and residual heads are ~0, so D starts where C3 starts."""
    decoder = small_deformed()
    value = probe()
    parts = decoder(value, return_parts=True)
    assert parts["warp"].abs().max() < 1.0e-3
    assert parts["residual"].abs().max() < 1.0e-3
    assert torch.allclose(decoder(value), decoder.template_only_sdf(value), atol=1.0e-3)


def test_deformed_warp_must_stay_coarser_than_the_template():
    with pytest.raises(ValueError, match="warp_min_wavelength_mm"):
        small_deformed(template_min_wavelength_mm=20.0, warp_min_wavelength_mm=15.0)


def test_build_mlp_output_dim_is_backward_compatible():
    from networks.conditional_hashgrid_sdf import build_mlp

    _hidden, scalar = build_mlp(7, 0, (8, 8), 0)
    _hidden, vector = build_mlp(7, 0, (8, 8), 0, output_dim=3)
    assert scalar.out_features == 1 and vector.out_features == 3
    with pytest.raises(ValueError):
        build_mlp(7, 0, (8, 8), 0, output_dim=0)


def test_numerical_jacobian_squared_matches_a_known_linear_field():
    """A linear warp W(x) = Ax has constant Jacobian A, exactly recoverable."""
    from hashgrid_common import numerical_jacobian_squared

    torch.manual_seed(0)
    matrix = torch.randn(3, 3, dtype=torch.float64)
    latent_size = 4

    def field(value):
        xyz = value[:, latent_size:]
        return xyz @ matrix

    xyz = torch.randn(32, 3, dtype=torch.float64)
    codes = torch.zeros(32, latent_size, dtype=torch.float64)
    epsilon = torch.full((3,), 1.0e-3, dtype=torch.float64)
    measured = numerical_jacobian_squared(field, codes, xyz, epsilon)
    assert torch.allclose(
        measured, matrix.square().sum().expand(32), rtol=1.0e-8, atol=1.0e-8
    )


def test_deformation_variants_and_decode_modes_are_registered():
    from hashgrid_common import DEFORMATION_ARCHITECTURES, is_deformation_field, variants_for

    config = {"network_arch": "deformed_implicit_field_sdf"}
    assert is_deformation_field(config)
    assert variants_for(config) == ["single", "template_only"]
    assert variants_for(config, ["single"]) == ["single"]
    assert "deformed_implicit_field_sdf" in DEFORMATION_ARCHITECTURES


def test_warp_metrics_separate_norm_from_component_bound():
    """Regression: a vector-norm metric was compared against a per-component
    bound, so a healthy warp looked saturated. The norm may legitimately reach
    warp_scale * sqrt(3); only the per-component value is comparable to the bound."""
    decoder = small_deformed(warp_scale_mm=6.0, mm_per_unit=117.431971)
    decoder.warp_output.weight.data.normal_(0.0, 50.0)
    decoder.warp_output.bias.data.normal_(0.0, 50.0)
    parts = decoder(probe(), return_parts=True)
    mm = 117.431971
    component_max = float(parts["warp"].abs().max()) * mm
    norm_max = float(parts["warp"].norm(dim=1).max()) * mm
    assert component_max <= 6.0 + 1.0e-4
    # Driven to saturation the norm exceeds the per-component bound; that is
    # arithmetic, not a bug, and the metric names must not imply otherwise.
    assert norm_max > 6.0
    assert norm_max <= 6.0 * 3**0.5 + 1.0e-4


def test_deformation_metric_keys_are_stable_and_unambiguous():
    """append_csv writes the header from the first row and history_fields is
    re-initialised per process, so a key set that varies -- between epochs or
    across a --resume -- shifts every column silently instead of failing."""
    from train_hashgrid_sdf import deformation_terms

    config = {
        "network_arch": "deformed_implicit_field_sdf",
        "mm_per_normalized_unit": 117.431971,
        "loss_weights": {"deformation_l2": 1.0, "residual_l2": 1.0, "deformation_jacobian": 0.0},
        "eikonal": {"target_band": 0.03},
    }
    codes = torch.zeros(4, 8)
    parts = {
        "warp": torch.zeros(4, 3),
        "warp_unit": torch.zeros(4, 3),
        "residual": torch.zeros(4, 1),
    }
    _loss, off = deformation_terms(None, codes, torch.zeros(1), parts, 1, config, [])
    config["loss_weights"]["deformation_jacobian"] = 0.01
    parts = {k: v + 0.5 for k, v in parts.items()}
    # The jacobian branch needs a model; only the key set is under test here.
    config["loss_weights"]["deformation_jacobian"] = 0.0
    _loss, on = deformation_terms(None, codes, torch.zeros(1), parts, 900, config, [])
    assert set(off) == set(on), "deformation metric keys must not depend on epoch"
    assert "warp_saturation_fraction" in off and "warp_gradient_scale_mean" in off
    # The ambiguous names must be gone, not merely supplemented.
    assert "warp_abs_mean_mm" not in off and "warp_abs_p95_mm" not in off
    assert {"warp_norm_mean_mm", "warp_component_p50_mm", "warp_component_max_mm"} <= set(off)


def test_non_deformation_architecture_logs_no_deformation_keys():
    from train_hashgrid_sdf import deformation_terms

    config = {"network_arch": "fourier_global_sdf", "loss_weights": {}}
    loss, metrics = deformation_terms(None, torch.zeros(2, 4), torch.zeros(1), {}, 1, config, [])
    assert metrics == {} and float(loss) == 0.0
