from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_adni_synthseg_pca_cocycle_v5 import (  # noqa: E402
    DirectDiagnosisResidualCocycleFlow,
    TimeEmbeddedDirectDiagnosisResidualCocycleFlow,
    build_flow,
    validate_config,
)


def make_time_flow() -> TimeEmbeddedDirectDiagnosisResidualCocycleFlow:
    return TimeEmbeddedDirectDiagnosisResidualCocycleFlow(
        latent_dim=150,
        width=256,
        residual_blocks=2,
        time_embedding_dim=32,
        time_frequencies=[1.0, 2.0, 4.0],
        modulation_max_scale=0.25,
    )


def test_time_conditioned_c4_keeps_backbone_size_and_adds_only_small_adapters() -> None:
    torch.manual_seed(17)
    baseline = DirectDiagnosisResidualCocycleFlow(latent_dim=150, width=256, residual_blocks=2)
    torch.manual_seed(17)
    time_flow = make_time_flow()

    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    time_parameters = sum(parameter.numel() for parameter in time_flow.parameters())
    assert baseline_parameters == 381_228
    assert time_parameters == 416_716
    assert time_parameters - baseline_parameters == 35_488

    time_state = time_flow.state_dict()
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(time_state[name], tensor, rtol=0.0, atol=0.0)


def test_zero_initialized_time_adapters_match_c4_and_are_checkpoint_compatible() -> None:
    torch.manual_seed(29)
    baseline = DirectDiagnosisResidualCocycleFlow(latent_dim=150, width=256, residual_blocks=2)
    torch.manual_seed(29)
    time_flow = make_time_flow()
    result = time_flow.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == {
        "time_embedding.frequencies",
        "time_embedding.fc1.weight",
        "time_embedding.fc1.bias",
        "time_embedding.fc2.weight",
        "time_embedding.fc2.bias",
        "time_embedding.norm.weight",
        "time_embedding.norm.bias",
        "time_modulations.0.weight",
        "time_modulations.0.bias",
        "time_modulations.1.weight",
        "time_modulations.1.bias",
    }

    latent = torch.randn(5, 150)
    source = torch.tensor([0.10, 0.25, 0.40, 0.55, 0.70])
    target = torch.tensor([0.16, 0.30, 0.48, 0.52, 0.88])
    label = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    torch.testing.assert_close(
        time_flow.transport(latent, source, target, label),
        baseline.transport(latent, source, target, label),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_time_flow_has_exact_identity_and_direction_sensitive_encoding() -> None:
    torch.manual_seed(5)
    flow = make_time_flow()
    latent = torch.randn(4, 150)
    source = torch.tensor([0.12, 0.31, 0.49, 0.77])
    target = torch.tensor([0.18, 0.47, 0.43, 0.95])
    label = torch.tensor([0.0, 1.0, 0.0, 1.0])

    torch.testing.assert_close(flow.transport(latent, source, source, label), latent, rtol=0.0, atol=0.0)
    forward_embedding = flow.time_embedding(source, target)
    backward_embedding = flow.time_embedding(target, source)
    assert bool(torch.isfinite(forward_embedding).all())
    assert bool(torch.isfinite(backward_embedding).all())
    assert not torch.allclose(forward_embedding, backward_embedding)
    assert bool(torch.isfinite(flow.transport(latent, target, source, label)).all())


def test_time_encoder_receives_gradients_after_the_zero_velocity_warm_start() -> None:
    torch.manual_seed(11)
    flow = make_time_flow()
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1.0e-3)
    latent = torch.randn(6, 150)
    source = torch.tensor([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    target = torch.tensor([0.18, 0.29, 0.46, 0.37, 0.72, 0.81])
    label = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    desired = torch.randn(6, 150)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((flow.transport(latent, source, target, label) - desired).square())
        loss.backward()
        optimizer.step()

    gradient = flow.time_embedding.fc1.weight.grad
    assert gradient is not None
    assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_time_embedded_configs_are_valid_and_build_independent_models() -> None:
    configuration_paths = {
        "hippocampus": PROJECT_ROOT
        / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/hippocampus_pca_cocycle_v4/cocycle_v5/configs/c4_time_embedded_cocycle_v5.json",
        "lateral_ventricle": PROJECT_ROOT
        / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/lateral_ventricle_pca_cocycle_v4/cocycle_v5/configs/c4_time_embedded_cocycle_v5.json",
    }
    run_names: set[str] = set()
    for structure, path in configuration_paths.items():
        config = json.loads(path.read_text())
        validate_config(path, config, structure, "c4_time_embedded")
        flow = build_flow(config["model"])
        assert isinstance(flow, TimeEmbeddedDirectDiagnosisResidualCocycleFlow)
        assert sum(parameter.numel() for parameter in flow.parameters()) == 416_716
        run_names.add(config["training"]["run_name"])
        assert config["scientific_contract"]["strict_no_mci"] is True
        assert config["scientific_contract"]["test_loaded_during_training"] is False
        assert config["scientific_contract"]["source_meshes_modified"] is False
    assert len(run_names) == 2


def test_existing_c4_configuration_remains_the_unchanged_direct_baseline() -> None:
    path = (
        PROJECT_ROOT
        / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/hippocampus_pca_cocycle_v4/cocycle_v5/configs/c4_cocycle_v5.json"
    )
    config = json.loads(path.read_text())
    validate_config(path, config, "hippocampus", "c4")
    flow = build_flow(config["model"])
    assert isinstance(flow, DirectDiagnosisResidualCocycleFlow)
    assert sum(parameter.numel() for parameter in flow.parameters()) == 381_228
