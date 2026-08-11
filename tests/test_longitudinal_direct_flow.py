from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from adni_no_mci_direct_flow_data import build_forward_pairs
from longitudinal_direct_flow import (
    DirectAgeFlow,
    MinimalDirectFlowLoss,
    MinimalLossConfig,
    direct_and_composed,
    per_row_latent_mse,
)


class FirstLatentDecoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1]


def test_transport_is_identity_when_source_equals_target() -> None:
    torch.manual_seed(3)
    flow = DirectAgeFlow(
        latent_size=4,
        hidden_dims=[8],
        condition_dim=1,
        zero_initialize_output=False,
    )
    latent = torch.randn(3, 4)
    time = torch.tensor([0.1, 0.4, 0.9])
    condition = torch.tensor([[0.0], [1.0], [0.0]])

    transported = flow.transport(latent, time, time, condition)

    torch.testing.assert_close(transported, latent)


def test_constant_velocity_satisfies_exact_cocycle() -> None:
    flow = DirectAgeFlow(
        latent_size=3,
        hidden_dims=[5],
        condition_dim=1,
        zero_initialize_output=True,
    )
    final_layer = [layer for layer in flow.net if isinstance(layer, nn.Linear)][-1]
    with torch.no_grad():
        final_layer.bias.copy_(torch.tensor([0.4, -0.2, 0.1]))

    latent = torch.tensor([[0.2, 0.3, -0.1], [-0.4, 0.1, 0.5]])
    source = torch.tensor([0.2, 0.3])
    intermediate = torch.tensor([0.4, 0.6])
    target = torch.tensor([0.8, 0.9])
    condition = torch.tensor([[0.0], [1.0]])
    direct, _, composed = direct_and_composed(
        flow,
        latent,
        source,
        intermediate,
        target,
        condition,
    )

    torch.testing.assert_close(direct, composed, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        per_row_latent_mse(direct, composed),
        torch.zeros(2),
        rtol=0.0,
        atol=1e-12,
    )


def test_minimal_loss_reaches_flow_but_not_decoder_parameters() -> None:
    torch.manual_seed(7)
    flow = DirectAgeFlow(
        latent_size=2,
        hidden_dims=[6],
        condition_dim=1,
        zero_initialize_output=True,
    )
    decoder = FirstLatentDecoder()
    loss_module = MinimalDirectFlowLoss(
        decoder=decoder,
        flow=flow,
        clamp_distance=0.1,
        config=MinimalLossConfig(),
    )
    source_latent = torch.tensor([[0.05, 0.01], [0.04, -0.02]])
    target_samples = torch.zeros(2, 8, 4)
    target_samples[:, :, 3] = -0.02

    output = loss_module(
        source_latent=source_latent,
        source_time=torch.tensor([0.2, 0.3]),
        target_time=torch.tensor([0.7, 0.8]),
        condition=torch.tensor([[0.0], [1.0]]),
        target_samples=target_samples,
        observed_intermediate_time=torch.tensor([0.4, 0.55]),
        observed_intermediate_mask=torch.tensor([True, False]),
        virtual_ratio=torch.tensor([[0.25], [0.75]]),
    )
    output.total.backward()

    final_layer = [layer for layer in flow.net if isinstance(layer, nn.Linear)][-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad).item() > 0
    assert list(decoder.parameters()) == []
    assert output.real_prediction.item() > 0
    assert output.observed_consistency.item() >= 0
    assert output.virtual_consistency.item() >= 0


def test_three_scans_create_all_forward_pairs_and_one_real_midpoint() -> None:
    ages = [70.0, 70.5, 71.0]
    frame = pd.DataFrame(
        {
            "scan_id": ["scan_70", "scan_70_5", "scan_71"],
            "subject_id": ["subject"] * 3,
            "diagnosis": ["CN"] * 3,
            "label_ad": [0] * 3,
            "visit_order": [0, 1, 2],
            "continuous_age_norm": [(age - 57.0) / 34.0 for age in ages],
            "sdf_npz_path": ["unused"] * 3,
        }
    )

    pairs = build_forward_pairs(frame)

    assert [
        (pair.source_scan_id, pair.target_scan_id) for pair in pairs
    ] == [
        ("scan_70", "scan_70_5"),
        ("scan_70", "scan_71"),
        ("scan_70_5", "scan_71"),
    ]
    midpoint_pairs = [pair for pair in pairs if pair.has_observed_intermediate]
    assert len(midpoint_pairs) == 1
    assert midpoint_pairs[0].source_scan_id == "scan_70"
    assert midpoint_pairs[0].target_scan_id == "scan_71"
    assert midpoint_pairs[0].observed_intermediate_times == (
        (70.5 - 57.0) / 34.0,
    )


def test_first_only_pairs_use_subject_anchor_as_source() -> None:
    ages = [70.0, 70.5, 71.0, 73.0]
    frame = pd.DataFrame(
        {
            "scan_id": ["scan_70", "scan_70_5", "scan_71", "scan_73"],
            "subject_id": ["subject"] * 4,
            "diagnosis": ["AD"] * 4,
            "label_ad": [1] * 4,
            "visit_order": [0, 1, 2, 3],
            "continuous_age_norm": [(age - 57.0) / 34.0 for age in ages],
            "sdf_npz_path": ["unused"] * 4,
        }
    )

    pairs = build_forward_pairs(frame, source_mode="first_only")

    assert [
        (pair.source_scan_id, pair.target_scan_id) for pair in pairs
    ] == [
        ("scan_70", "scan_70_5"),
        ("scan_70", "scan_71"),
        ("scan_70", "scan_73"),
    ]
    assert pairs[-1].observed_intermediate_times == (
        (70.5 - 57.0) / 34.0,
        (71.0 - 57.0) / 34.0,
    )
