import torch

from networks.longitudinal_disentangled_flow import build_temporal_flow


def _spec():
    return {
        "FlowModelType": "disentangled_velocity",
        "VelocityAgeDim": 16,
        "VelocityDiseaseDim": 16,
        "VelocityResidualDim": 224,
        "VelocityAgeEmbeddingScale": 0.02,
        "VelocityResidualUsesDiagnosis": False,
    }


def test_disentangled_velocity_shapes_and_gate():
    flow = build_temporal_flow(_spec(), 256, [32], age_condition_dim=1)
    z = torch.randn(4, 256)
    s = torch.zeros(4, 1)
    t = torch.linspace(0.1, 0.4, 4).unsqueeze(1)
    diagnosis = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    components = flow.velocity_components(z, s, t, age_cond=diagnosis)
    assert components["velocity"].shape == (4, 256)
    assert components["age"].shape == (4, 16)
    assert components["disease_raw"].shape == (4, 16)
    assert components["disease"].shape == (4, 16)
    assert components["residual"].shape == (4, 224)
    assert torch.allclose(components["disease"][diagnosis.view(-1) == 0], torch.zeros(2, 16))


def test_age_embedding_composes():
    flow = build_temporal_flow(_spec(), 256, [32], age_condition_dim=1)
    z = torch.randn(3, 256)
    s = torch.full((3, 1), 0.1)
    r = torch.full((3, 1), 0.4)
    t = torch.full((3, 1), 0.7)

    disp_sr = (r - s) * flow.age_velocity(s, r)
    disp_rt = (t - r) * flow.age_velocity(r, t)
    disp_st = (t - s) * flow.age_velocity(s, t)

    assert torch.allclose(disp_sr + disp_rt, disp_st, atol=1e-6)

    diagonal = flow.age_velocity(t, t)
    assert torch.isfinite(diagonal).all()
