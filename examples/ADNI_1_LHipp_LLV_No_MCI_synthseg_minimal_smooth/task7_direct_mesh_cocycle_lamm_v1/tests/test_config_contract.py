from __future__ import annotations

import common as C
from train import validate_config


def test_all_public_configs_are_valid_and_end_to_end() -> None:
    for path in sorted((C.TASK_ROOT / "configs").glob("*.json")):
        config = C.read_json(path)
        if config.get("method") != "direct_surface_cocycle":
            continue
        validate_config(config)
        contract = config["scientific_contract"]
        assert contract["end_to_end"] is True
        assert contract["frozen_components"] is False
        assert contract["ode_used"] is False
        mode = config["model"].get("bottleneck_mode", "global")
        assert contract["global_latent_bottleneck"] is (mode == "global")
        if mode == "global":
            assert config["model"]["latent_dim"] > 0
        else:
            assert "latent_dim" not in config["model"]


def test_three_final_experiments_are_scientifically_distinct() -> None:
    global256 = C.read_json(C.TASK_ROOT / "configs" / "lamm_global_c4_z256_s42.json")
    global384 = C.read_json(C.TASK_ROOT / "configs" / "lamm_global_c4_z384_s42.json")
    tokens = C.read_json(C.TASK_ROOT / "configs" / "lamm_token_c4_s42.json")
    assert global256["model"]["bottleneck_mode"] == "global"
    assert global256["model"]["latent_dim"] == 256
    assert global256["model"]["latent_split"] == [96, 160]
    assert global384["model"]["bottleneck_mode"] == "global"
    assert global384["model"]["latent_dim"] == 384
    assert global384["model"]["latent_split"] == [128, 256]
    assert tokens["model"]["bottleneck_mode"] == "regional_tokens"
    assert tokens["model"]["token_flow_depth"] == 2
    assert "latent_dim" not in tokens["model"]


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
