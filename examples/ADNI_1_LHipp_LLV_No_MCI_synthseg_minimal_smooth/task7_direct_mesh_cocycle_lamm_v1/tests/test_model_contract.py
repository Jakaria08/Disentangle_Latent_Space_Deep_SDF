from __future__ import annotations

import inspect

import torch

from conditional_lamm_flow import ConditionalLAMMFlow
from helpers import make_batch, synthetic_config, synthetic_layout, synthetic_statistics


def build(latent_dim: int) -> ConditionalLAMMFlow:
    stats = synthetic_statistics()
    return ConditionalLAMMFlow(
        synthetic_layout(), stats["faces"], synthetic_config(latent_dim), stats
    )


def test_128_and_256_are_internal_bottlenecks() -> None:
    vertices, source, target, disease = make_batch()
    for latent_dim in (128, 256):
        model = build(latent_dim)
        latent, condition = model.encode(vertices, source, target, disease)
        assert latent.shape == (2, latent_dim)
        assert condition.shape == (2, 8)
        assert model.transport(vertices, source, target, disease).shape == vertices.shape
        assert model.global_latent_bottleneck is True


def test_identity_is_bit_exact() -> None:
    model = build(128).eval()
    vertices, source, _, disease = make_batch()
    with torch.no_grad():
        prediction = model.transport(vertices, source, source, disease)
    assert torch.equal(prediction, vertices)


def test_diagnosis_changes_velocity() -> None:
    model = build(128).eval()
    model.set_test_velocity_bias(cn=0.01, ad_residual=0.02)
    vertices, source, _, disease = make_batch()
    with torch.no_grad():
        cn = model.instantaneous_velocity(vertices, source, torch.zeros_like(disease))
        ad = model.instantaneous_velocity(vertices, source, torch.ones_like(disease))
    assert float((ad - cn).abs().mean()) > 0.0


def test_diagonal_velocity_matches_small_time_transport() -> None:
    model = build(128).eval()
    model.set_test_velocity_bias(cn=0.01, ad_residual=0.02)
    vertices, age, _, disease = make_batch()
    epsilon = torch.full_like(age, 1.0e-3)
    with torch.no_grad():
        analytic = model.instantaneous_velocity(vertices, age, disease)
        finite = (
            model.transport(vertices, age, age + epsilon, disease) - vertices
        ) / epsilon.reshape(-1, 1, 1)
    assert float((analytic - finite).abs().max()) < 5.0e-4


def test_all_components_are_trainable_end_to_end() -> None:
    model = build(128).train()
    # Production starts at exact no-change. Give the output maps a tiny non-zero value here
    # so a single backward pass can verify gradient reachability behind the zero-init heads.
    with torch.no_grad():
        for head in model.velocity_heads:
            head.weight.normal_(mean=0.0, std=1.0e-3)
    vertices, source, target, disease = make_batch()
    loss = model.average_velocity(vertices, source, target, disease).square().mean()
    loss.backward()
    prefixes = {
        "condition.": False,
        "tokenizers.": False,
        "encoder.": False,
        "w_down.": False,
        "latent_input.": False,
        "latent_blocks.": False,
        "latent_output.": False,
        "w_up.": False,
        "region_tokens.": False,
        "decoder.": False,
        "velocity_heads.": False,
    }
    for name, parameter in model.named_parameters():
        for prefix in prefixes:
            if name.startswith(prefix) and parameter.grad is not None:
                prefixes[prefix] |= float(parameter.grad.abs().sum()) > 0.0
    assert all(prefixes.values()), prefixes
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_composition_is_finite_and_differentiable() -> None:
    model = build(128).train()
    with torch.no_grad():
        for head in model.velocity_heads:
            head.weight.normal_(mean=0.0, std=1.0e-3)
    vertices, source, target, disease = make_batch()
    middle = 0.5 * (source + target)
    direct = model.transport(vertices, source, target, disease)
    first = model.transport(vertices, source, middle, disease)
    composed = model.transport(first, middle, target, disease)
    defect = (direct - composed).square().mean()
    assert torch.isfinite(defect)
    defect.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_no_ode_solver_is_present() -> None:
    source = inspect.getsource(ConditionalLAMMFlow)
    assert "torchdiffeq" not in source
    assert "odeint" not in source
    assert model_transport_signature_is_direct(source)


def model_transport_signature_is_direct(source: str) -> bool:
    return "vertices + elapsed * self.average_velocity" in source

