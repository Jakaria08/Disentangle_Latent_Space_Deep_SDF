import torch

from networks.longitudinal_additive_flow import build_temporal_flow, gradient_reverse


def _spec():
    return {
        "FlowModelType": "additive_velocity",
        "AdditiveDiseaseUsesLatent": False,
        "AdditiveResidualUsesDiagnosis": False,
        "AdditiveProbeHiddenDims": [16],
    }


def test_additive_velocity_components_and_sum():
    flow = build_temporal_flow(_spec(), 256, [32], age_condition_dim=1)
    z = torch.randn(5, 256)
    s = torch.zeros(5, 1)
    t = torch.linspace(0.1, 0.5, 5).unsqueeze(1)
    disease = torch.tensor([[0.0], [1.0], [0.0], [1.0], [1.0]])

    components = flow.velocity_components(z, s, t, age_cond=disease)
    assert components["age"].shape == (5, 256)
    assert components["disease"].shape == (5, 256)
    assert components["residual"].shape == (5, 256)
    assert components["velocity"].shape == (5, 256)

    total = components["age"] + components["disease"] + components["residual"]
    assert torch.allclose(components["velocity"], total)


def test_disease_branch_zero_reference():
    flow = build_temporal_flow(_spec(), 256, [32], age_condition_dim=1)
    z = torch.randn(3, 256)
    s = torch.zeros(3, 1)
    t = torch.ones(3, 1) * 0.25
    disease = torch.zeros(3, 1)

    components = flow.velocity_components(z, s, t, age_cond=disease)
    assert torch.allclose(components["disease"], torch.zeros_like(components["disease"]))


def test_gradient_reverse_changes_gradient_sign():
    x = torch.tensor([[1.0, -2.0]], requires_grad=True)
    y = gradient_reverse(x, 0.5).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))
