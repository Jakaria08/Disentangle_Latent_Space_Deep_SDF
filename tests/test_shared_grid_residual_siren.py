from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task2_inr_representations_v1"
)
SCRIPT_DIR = TASK_DIR / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from networks.shared_grid_residual_siren import Decoder  # noqa: E402
from shared_grid_common import (  # noqa: E402
    ContinuousSDFDataset,
    build_decoder,
    fit_single_latent,
    load_manifest,
    sample_continuous_sdf_pair,
    select_stratified_rows,
    warm_start_global_decoder,
)
from train_shared_grid_residual import make_optimizer, save_checkpoint  # noqa: E402


def tiny_decoder() -> Decoder:
    return Decoder(
        latent_size=4,
        global_network_specs={
            "dims": [16, 20],
            "latent_in": [2],
            "xyz_in": [],
            "xyz_in_all": False,
            "encoding_features": 1,
            "nonlinearity": "sine",
            "final_layer_init": "siren",
        },
        grid_resolution=8,
        grid_channels=3,
        grid_aabb=[[-0.5, -0.6, -0.7], [0.5, 0.6, 0.7]],
        local_hidden_dims=[12, 12, 12],
        local_skip_layer=2,
        gate_tau=0.05,
        roi_taper_width=0.04,
    )


def sampling_config(scale: int = 1) -> dict:
    return {
        "global_near_samples_per_scene": 8 * scale,
        "global_positive_samples_per_scene": 4 * scale,
        "global_negative_samples_per_scene": 4 * scale,
        "local_ultra_near_samples_per_scene": 8 * scale,
        "local_positive_samples_per_scene": 4 * scale,
        "local_negative_samples_per_scene": 4 * scale,
        "near_band": 0.1,
        "ultra_near_band": 0.03,
    }


def test_fusion_alpha_zero_preserves_global_and_local_is_direct_sdf() -> None:
    torch.manual_seed(7)
    decoder = tiny_decoder()
    latent = torch.randn(10, 4)
    xyz = torch.rand(10, 3) - 0.5
    input_x = torch.cat((latent, xyz), dim=1)
    parts = decoder(input_x, return_parts=True, fusion_alpha=0.0)
    torch.testing.assert_close(parts["sdf"], parts["global_sdf"], rtol=0.0, atol=0.0)
    assert not torch.equal(parts["local_sdf"], parts["global_sdf"])
    assert decoder.local_decoder.hidden[2].in_features == 12 + 4 + 3 + 3


def test_grid_is_shared_and_roi_fusion_tapers_continuously() -> None:
    decoder = tiny_decoder()
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    features, inside = decoder.sample_shared_grid(xyz)
    assert tuple(decoder.shared_grid.shape) == (1, 3, 8, 8, 8)
    assert inside[:, 0].tolist() == [True, False]
    torch.testing.assert_close(features[1], torch.zeros(3), rtol=0.0, atol=0.0)

    boundary = torch.tensor([[0.5 - 1.0e-5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5 + 1.0e-5, 0.0, 0.0]])
    weights = decoder.roi_weight(boundary)[:, 0]
    assert float(weights[0]) < 1.0e-6
    assert float(weights[1]) == 0.0
    assert float(weights[2]) == 0.0


def test_training_stage_freezes_intended_model_groups() -> None:
    decoder = tiny_decoder()
    decoder.set_train_stage("global_adapt")
    assert all(parameter.requires_grad for parameter in decoder.global_decoder.parameters())
    assert not any(parameter.requires_grad for parameter in decoder.local_decoder.parameters())
    assert not decoder.shared_grid.requires_grad

    decoder.set_train_stage("local_warmup")
    assert not any(parameter.requires_grad for parameter in decoder.global_decoder.parameters())
    assert all(parameter.requires_grad for parameter in decoder.local_decoder.parameters())
    assert decoder.shared_grid.requires_grad

    decoder.set_train_stage("joint")
    assert all(parameter.requires_grad for parameter in decoder.parameters())


def test_continuous_sampler_returns_distinct_global_local_contract() -> None:
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-0.4, 0.4, (200, 3))
    positive = np.concatenate((xyz, rng.uniform(0.001, 0.12, (200, 1))), axis=1).astype(np.float32)
    negative = np.concatenate((xyz, rng.uniform(-0.12, -0.001, (200, 1))), axis=1).astype(np.float32)
    global_samples, local_samples = sample_continuous_sdf_pair(
        positive,
        negative,
        sampling_config(),
        rng,
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        8,
    )
    assert global_samples.shape == (16, 4)
    assert local_samples.shape == (16, 4)
    assert np.all(np.abs(local_samples[:, 3]) <= 0.1)
    assert np.any(np.abs(global_samples[:, 3]) > 0.1)


def test_dataset_reads_continuous_npz_without_synthetic_offsets(tmp_path: Path) -> None:
    sdf = tmp_path / "samples.npz"
    rng = np.random.default_rng(4)
    xyz = rng.uniform(-0.4, 0.4, (200, 3))
    positive = np.concatenate((xyz, rng.uniform(0.001, 0.12, (200, 1))), axis=1).astype(np.float32)
    negative = np.concatenate((xyz, rng.uniform(-0.12, -0.001, (200, 1))), axis=1).astype(np.float32)
    np.savez(sdf, pos=positive, neg=negative)
    config = {
        "sampling": sampling_config(),
        "network_specs": {
            "grid_aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "grid_resolution": 8,
        },
        "load_dataset_into_ram": False,
    }
    dataset = ContinuousSDFDataset([{"sdf_npz_path": str(sdf)}], config)
    global_samples, local_samples, index = dataset[0]
    assert index == 0
    assert tuple(global_samples.shape) == (16, 4)
    assert tuple(local_samples.shape) == (16, 4)
    archive_values = np.concatenate((positive[:, 3], negative[:, 3]))
    assert all(float(value) in archive_values for value in global_samples[:, 3])
    assert all(float(value) in archive_values for value in local_samples[:, 3])
    repeated_global, repeated_local, _ = dataset[0]
    torch.testing.assert_close(global_samples, repeated_global)
    torch.testing.assert_close(local_samples, repeated_local)
    dataset.set_epoch(1)
    next_global, next_local, _ = dataset[0]
    assert not torch.equal(global_samples, next_global)
    assert not torch.equal(local_samples, next_local)


def test_heldout_latent_fit_reports_all_three_branches(tmp_path: Path) -> None:
    sdf = tmp_path / "fit_samples.npz"
    rng = np.random.default_rng(9)
    xyz = rng.uniform(-0.4, 0.4, (300, 3))
    positive = np.concatenate((xyz, rng.uniform(0.001, 0.12, (300, 1))), axis=1).astype(np.float32)
    negative = np.concatenate((xyz, rng.uniform(-0.12, -0.001, (300, 1))), axis=1).astype(np.float32)
    np.savez(sdf, pos=positive, neg=negative)
    fit_config = {
        "steps": 2,
        "sampling": sampling_config(),
        "branch_weights": {"global": 1.0, "local": 1.0, "fused": 1.0},
        "learning_rate": 0.001,
        "code_regularization_lambda": 0.0001,
        "code_bound": 1.0,
        "initial_std": 0.01,
        "holdout_fraction": 0.2,
        "early_stop_patience": 2,
        "early_stop_min_delta": 0.0,
    }
    latent, metrics = fit_single_latent(
        tiny_decoder(),
        sdf,
        4,
        fit_config,
        0.1,
        torch.device("cpu"),
        42,
        network_specs={
            "grid_aabb": [[-0.5, -0.6, -0.7], [0.5, 0.6, 0.7]],
            "grid_resolution": 8,
        },
    )
    assert latent.shape == (4,)
    assert metrics["steps_completed"] == 2
    assert all(
        np.isfinite(metrics[name])
        for name in ("heldout_global_l1", "heldout_local_l1", "heldout_fused_l1")
    )


def test_hippocampus_default_contract_and_schedule() -> None:
    config_path = TASK_DIR / "configs" / "hippocampus_shared_grid_32x16_z256.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = REPO_ROOT / config["global_checkpoint"]
    source = torch.load(source_path, map_location="cpu")["config"]
    assert config["latent_size"] == source["latent_size"] == 256
    assert config["network_specs"]["global_network_specs"] == source["network_specs"]
    assert config["network_specs"]["grid_resolution"] == 32
    assert config["network_specs"]["grid_channels"] == 16
    assert config["network_specs"]["local_hidden_dims"] == [128, 128, 128]
    assert config["network_specs"]["local_skip_layer"] == 2
    assert config["schedule"] == {
        "global_adapt_epochs": 100,
        "local_warmup_epochs": 100,
        "joint_fusion_ramp_epochs": 25,
        "total_epochs": 1700,
    }
    assert config["reuse_matching_source_latents"] is False


def test_warm_start_loads_global_weights_without_old_qc_latents() -> None:
    config_path = TASK_DIR / "configs" / "hippocampus_shared_grid_32x16_z256.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = build_decoder(config, torch.device("cpu"))
    report = warm_start_global_decoder(
        decoder,
        config["global_checkpoint"],
        torch.device("cpu"),
        source_scan_id_suffix=config["global_checkpoint_scan_id_suffix"],
    )
    source_payload = torch.load(REPO_ROOT / config["global_checkpoint"], map_location="cpu")
    first_key = next(iter(source_payload["model_state_dict"]))
    torch.testing.assert_close(
        decoder.global_decoder.state_dict()[first_key],
        source_payload["model_state_dict"][first_key],
    )
    assert report["transfer"] == "strict_global_decoder"


def test_stratified_selection_is_fixed_and_subject_diverse() -> None:
    rows = [
        {"scan_id": f"{subject}_{visit}", "subject_id": str(subject), "diagnosis": diagnosis}
        for diagnosis in ("CN", "AD")
        for subject in range(5)
        for visit in ("bl", "m12")
    ]
    first = select_stratified_rows(rows, 8, 42)
    second = select_stratified_rows(rows, 8, 42)
    assert [row["scan_id"] for row in first] == [row["scan_id"] for row in second]
    assert {row["diagnosis"] for row in first} == {"CN", "AD"}
    assert len({row["subject_id"] for row in first}) >= 4


def test_checkpoint_contains_resume_state_and_compatibility_latents(tmp_path: Path) -> None:
    decoder = tiny_decoder()
    embedding = torch.nn.Embedding(2, 4)
    config = {
        "output_dir": str(tmp_path),
        "stage_learning_rates": {
            "global_adapt": {"global": 1e-4, "local": 0.0, "grid": 0.0, "latent": 1e-3}
        },
    }
    optimizer = make_optimizer(decoder, embedding, config)
    checkpoint = tmp_path / "checkpoints" / "epoch_0001.pth"
    save_checkpoint(
        checkpoint,
        1,
        decoder,
        embedding,
        optimizer,
        0.1,
        0.2,
        config,
        [{"scan_id": "a"}, {"scan_id": "b"}],
        {"transfer": "test"},
        compatibility_label="epoch_0001",
    )
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["format_version"] == 2
    assert payload["training_scan_ids"] == ["a", "b"]
    assert "optimizer_state_dict" in payload and "rng_state" in payload
    latent_payload = torch.load(
        tmp_path / "LatentCodes" / "epoch_0001.pth", map_location="cpu"
    )
    assert latent_payload["train_scan_ids"] == ["a", "b"]
    assert latent_payload["latent_codes"]["weight"].shape == (2, 4)
    assert not list(tmp_path.rglob("*.tmp"))


def test_manifest_is_the_expected_qc_cohort() -> None:
    config = json.loads(
        (TASK_DIR / "configs" / "hippocampus_shared_grid_32x16_z256.json").read_text()
    )
    rows = load_manifest(config["manifest"])
    assert len([row for row in rows if row["split"] == "train"]) == 2037
    assert len(rows) == 2583
