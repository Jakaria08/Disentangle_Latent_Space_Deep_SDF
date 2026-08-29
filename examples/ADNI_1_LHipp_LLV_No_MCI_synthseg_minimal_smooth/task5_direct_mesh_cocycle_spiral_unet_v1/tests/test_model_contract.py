from __future__ import annotations

import copy

import torch

import common as C
import data as D
from conditional_spiral_unet import ConditionalSpiralUNet
from mesh_hierarchy import load_hierarchy
from mesh_layers import adaptive_modules
from train import validate_config


def build(operator: str):
    filename = "adaptive_direct_c4_s42.json" if operator == "adaptive_spiral" else "spiral_direct_c4_s42.json"
    config = C.read_json(C.TASK_ROOT / "configs" / filename)
    validate_config(config)
    return ConditionalSpiralUNet(load_hierarchy(), config, D.load_statistics()), config


def test_configs_prohibit_ode_and_global_latent():
    for name in (
        "spiral_direct_c4_s42.json",
        "adaptive_direct_c4_s42.json",
        "spiral_direct_c4_velocity_v2_s42.json",
        "adaptive_direct_c4_velocity_v2_s42.json",
    ):
        config = C.read_json(C.TASK_ROOT / "configs" / name)
        validate_config(config)
        assert config["scientific_contract"]["ode_used"] is False
        assert config["scientific_contract"]["global_latent_bottleneck"] is False


def test_velocity_v2_operator_configs_are_matched():
    spiral = C.read_json(C.TASK_ROOT / "configs" / "spiral_direct_c4_velocity_v2_s42.json")
    adaptive = C.read_json(C.TASK_ROOT / "configs" / "adaptive_direct_c4_velocity_v2_s42.json")
    left = copy.deepcopy(spiral["model"])
    right = copy.deepcopy(adaptive["model"])
    left.pop("operator")
    right.pop("operator")
    assert left == right
    assert spiral["loss"] == adaptive["loss"]
    left_selection = copy.deepcopy(spiral["selection"])
    right_selection = copy.deepcopy(adaptive["selection"])
    left_selection.pop("velocity_validation_batch_size")
    right_selection.pop("velocity_validation_batch_size")
    assert left_selection == right_selection
    assert spiral["selection"]["velocity_aware"] is True
    assert spiral["selection"]["max_relative_cocycle_defect"] == 0.02
    assert spiral["selection"]["max_relative_inverse_defect"] == 0.02


def test_identity_is_exact_for_both_operators():
    split = D.load_split("val")
    for operator in ("spiral", "adaptive_spiral"):
        model, _config = build(operator)
        output = model.transport(split.vertices[:1], split.ages[:1], split.ages[:1], split.labels[:1])
        assert torch.equal(output, split.vertices[:1])


def test_operator_ablation_has_identical_nonoperator_contract():
    spiral = C.read_json(C.TASK_ROOT / "configs" / "spiral_direct_c4_s42.json")
    adaptive = C.read_json(C.TASK_ROOT / "configs" / "adaptive_direct_c4_s42.json")
    left = copy.deepcopy(spiral["model"])
    right = copy.deepcopy(adaptive["model"])
    left.pop("operator")
    right.pop("operator")
    assert left == right
    assert spiral["loss"] == adaptive["loss"]
    assert spiral["selection"] == adaptive["selection"]


def test_adaptive_is_only_at_coarse_resolutions():
    spiral, _ = build("spiral")
    adaptive, _ = build("adaptive_spiral")
    assert len(adaptive_modules(spiral)) == 0
    assert len(adaptive_modules(adaptive)) == 3
