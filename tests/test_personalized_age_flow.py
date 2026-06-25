import torch

from networks.longitudinal_flow_64_160_32_personal_age import (
    build_temporal_flow as build_personal_flow,
)
from networks.longitudinal_flow_64_160_32_pred_dx import (
    build_temporal_flow as build_e2_flow,
)


def _spec(mode):
    return {
        "FlowModelType": "disentangled_velocity",
        "VelocityAgeDim": 64,
        "VelocityDiseaseDim": 160,
        "VelocityResidualDim": 32,
        "VelocityAgeEmbeddingScale": 0.02,
        "VelocityDiseaseUsesLatent": True,
        "VelocityResidualUsesDiagnosis": False,
        "VelocityAdvProbeHiddenDims": [16],
        "AnchorDiseaseClassifierHiddenDims": [16],
        "AgePersonalizationMode": mode,
        "AgeModulationHiddenDims": [16],
        "AgeModulationAmplitude": 0.5,
    }


def _age_displacement(flow, z, condition, start, end):
    s = z.new_full((z.shape[0], 1), float(start))
    t = z.new_full((z.shape[0], 1), float(end))
    components = flow.velocity_components(z, s, t, age_cond=condition)
    return (t - s) * components["age"]


def test_zero_initialized_personalization_equals_population_age():
    flow = build_personal_flow(_spec("baseline_modulation"), 256, [32], 1)
    anchor = torch.randn(3, 256)
    diagnosis = torch.rand(3, 1)
    condition = flow.compose_condition(diagnosis, anchor)
    s = torch.rand(3, 1)
    t = s + 0.2

    components = flow.velocity_components(anchor, s, t, age_cond=condition)

    assert torch.allclose(condition[:, 1:], torch.ones(3, 16))
    assert torch.allclose(
        components["age"],
        components["age_population"],
    )
    assert torch.allclose(
        components["age_individual"],
        torch.zeros_like(components["age_individual"]),
    )


def test_personalized_age_displacement_composes_exactly():
    flow = build_personal_flow(_spec("baseline_modulation"), 256, [32], 1)
    final_layer = next(
        layer
        for layer in reversed(flow.age_modulation_net)
        if isinstance(layer, torch.nn.Linear)
    )
    with torch.no_grad():
        final_layer.bias.copy_(torch.linspace(-0.4, 0.4, 16))

    anchor = torch.randn(4, 256)
    diagnosis = torch.rand(4, 1)
    condition = flow.compose_condition(diagnosis, anchor)

    direct = _age_displacement(flow, anchor, condition, 0.1, 0.9)
    composed = _age_displacement(
        flow, anchor, condition, 0.1, 0.4
    ) + _age_displacement(flow, anchor, condition, 0.4, 0.9)

    assert torch.allclose(direct, composed, atol=1e-6, rtol=1e-5)


def test_population_mode_matches_experiment2_flow():
    torch.manual_seed(11)
    e2 = build_e2_flow(_spec("population"), 256, [32], 1)
    torch.manual_seed(11)
    population = build_personal_flow(_spec("population"), 256, [32], 1)

    shared_state = {
        key: value
        for key, value in population.state_dict().items()
        if key in e2.state_dict()
    }
    population.load_state_dict(
        {**population.state_dict(), **e2.state_dict()},
    )

    z = torch.randn(5, 256)
    s = torch.rand(5, 1)
    t = torch.rand(5, 1)
    diagnosis = torch.rand(5, 1)

    e2_output = e2.velocity_components(z, s, t, age_cond=diagnosis)
    population_output = population.velocity_components(
        z,
        s,
        t,
        age_cond=diagnosis,
    )

    assert shared_state
    assert torch.allclose(e2_output["velocity"], population_output["velocity"])


def test_future_loss_trains_modulation_without_updating_detached_anchor():
    flow = build_personal_flow(_spec("baseline_modulation"), 256, [32], 1)
    final_layer = next(
        layer
        for layer in reversed(flow.age_modulation_net)
        if isinstance(layer, torch.nn.Linear)
    )
    with torch.no_grad():
        final_layer.weight.normal_(std=0.01)

    anchor = torch.randn(2, 256, requires_grad=True)
    diagnosis = torch.rand(2, 1)
    condition = flow.compose_condition(
        diagnosis,
        anchor,
        detach_anchor=True,
    )
    s = torch.zeros(2, 1)
    t = torch.ones(2, 1)
    velocity = flow(anchor.detach(), s, t, age_cond=condition)

    velocity.square().mean().backward()

    assert anchor.grad is None
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in flow.age_modulation_net.parameters()
    )
