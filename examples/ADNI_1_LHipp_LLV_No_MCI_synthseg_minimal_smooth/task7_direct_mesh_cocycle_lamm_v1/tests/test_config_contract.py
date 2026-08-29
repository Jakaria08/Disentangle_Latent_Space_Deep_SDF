from __future__ import annotations

import common as C
from train import validate_config


def test_all_public_configs_are_valid_and_end_to_end() -> None:
    for path in sorted((C.TASK_ROOT / "configs").glob("*.json")):
        config = C.read_json(path)
        validate_config(config)
        contract = config["scientific_contract"]
        assert contract["end_to_end"] is True
        assert contract["frozen_components"] is False
        assert contract["ode_used"] is False
        assert config["model"]["latent_dim"] in {128, 256}


def test_width_comparison_has_controlled_layouts() -> None:
    z128 = C.read_json(C.TASK_ROOT / "configs" / "lamm_direct_c4_z128_s42.json")
    equal = C.read_json(C.TASK_ROOT / "configs" / "lamm_direct_c4_z256_equal_s42.json")
    fine = C.read_json(C.TASK_ROOT / "configs" / "lamm_direct_c4_z256_fine_s42.json")
    assert z128["model"]["latent_split"] == [64, 64]
    assert equal["model"]["latent_split"] == [128, 128]
    assert fine["model"]["latent_split"] == [96, 160]
    for key in (
        "region_scales",
        "token_dim",
        "encoder_depth",
        "decoder_depth",
        "latent_width",
        "latent_residual_blocks",
        "condition_dim",
        "dropout",
    ):
        assert z128["model"][key] == equal["model"][key] == fine["model"][key]
