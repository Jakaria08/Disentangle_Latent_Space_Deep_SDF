#!/usr/bin/env python3
"""Stage 2 tests: transport algebra, BrainODE equivalence, Latent ODE correctness, contracts.

Needs torch, so run it with the pytorch_geo environment:
    /home/jakaria/anaconda3/envs/pytorch_geo/bin/python tests/test_stage2_dynamics.py
CPU only; reads the stage-1 ADNI view for the leakage-guard test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_common as bc  # noqa: E402
import dynamics_core as D  # noqa: E402
import evaluate_dynamics as EV  # noqa: E402
import rubanova_latent_ode as R  # noqa: E402

PARTS = D.core()
M, C = PARTS["M"], PARTS["C"]


def _randomize(module: torch.nn.Module, seed: int) -> None:
    """Zero-initialized output heads would make several identities trivially true."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def small_latent_ode(**model_overrides) -> R.LatentODE:
    overrides = {"model.z0_dim": 8, "model.encoder_hidden": 16, "model.encoder_ode_width": 16, "model.dynamics_width": 16,
                 "model.decoder_width": 16, "model.latent_dim": 6, "training.integration_substeps": 2}
    overrides.update(model_overrides)
    return R.LatentODE.from_config(D.load_recipe("latent_ode", "pca128", overrides))


# --------------------------------------------------------------------------------------
# integrator and transports
# --------------------------------------------------------------------------------------


def test_rk4_matches_matrix_exponential():
    torch.manual_seed(0)
    A = torch.randn(5, 5, dtype=torch.float64) * 0.5

    class Linear(torch.nn.Module):
        def forward(self, time, latent, condition):
            return latent @ A.T

    z0 = torch.randn(3, 5, dtype=torch.float64)
    times = torch.tensor([[0.0, 1.0]] * 3, dtype=torch.float64)
    states = M.integrate_sequence_rk4(Linear(), z0, times, torch.zeros(3, dtype=torch.float64), 64)
    exact = z0 @ torch.matrix_exp(A).T
    assert float((states[:, -1] - exact).abs().max()) < 1e-8


def test_direct_c4_identity_and_disease_head_structure():
    flow = M.DirectC4Flow(128, 32, 1).double()
    _randomize(flow, 1)
    z = torch.randn(4, 128, dtype=torch.float64)
    s = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    t = s + 0.35
    zero, one = torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
    assert torch.equal(flow.transport(z, s, s, one), z), "Phi(z,s,s,d) must equal z exactly"
    control = flow.transport(z, s, t, zero)
    with torch.no_grad():
        flow.ad_residual_head.weight.add_(1.0)
    assert torch.allclose(flow.transport(z, s, t, zero), control), "d=0 must not depend on the disease head"
    assert not torch.allclose(flow.transport(z, s, t, one), control)


def test_ode_semigroup_and_inverse():
    function = M.PlainODEFunc(128, 32, 1, 0.0).double()
    _randomize(function, 2)
    z = torch.randn(3, 128, dtype=torch.float64) * 0.1
    s, u, t = (torch.full((3,), value, dtype=torch.float64) for value in (0.1, 0.25, 0.4))
    d = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    direct = M.transport_rk4(function, z, s, t, d, 64)
    composed = M.transport_rk4(function, M.transport_rk4(function, z, s, u, d, 64), u, t, d, 64)
    inverse = M.transport_rk4(function, direct, t, s, d, 64)
    assert float((direct - composed).abs().max()) < 1e-6
    assert float((inverse - z).abs().max()) < 1e-6


def test_brainode_vectorized_equals_august_per_subject_loop():
    path = D.AUGUST_SCRIPTS / "models.py"
    spec = importlib.util.spec_from_file_location("august_models_for_test", path)
    august = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(august)
    loop = august.BrainODEAttentionFunc(128, 64, 96, 0.0)
    vectorized = M.BrainODEAttentionFunc(128, 64, 96, 0.0)
    _randomize(loop, 3)
    vectorized.load_state_dict(loop.state_dict(), strict=True)
    z = torch.randn(17, 128, requires_grad=True)
    time, condition = torch.rand(17), (torch.rand(17) > 0.5).float()
    a, b = loop(time, z, condition), vectorized(time, z, condition)
    assert float((a - b).abs().max()) < 1e-5
    grad_a = torch.autograd.grad(a.square().sum(), z)[0]
    grad_b = torch.autograd.grad(b.square().sum(), z)[0]
    assert float((grad_a - grad_b).abs().max()) < 1e-4


# --------------------------------------------------------------------------------------
# Latent ODE
# --------------------------------------------------------------------------------------


def _toy_batch(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([4, 2, 3, 4])
    obs = torch.randn(4, 4, 6, generator=generator)
    times = torch.tensor([[0.0, 0.2, 0.5, 0.9]] * 4)
    for row, n in enumerate(lengths):
        times[row, n:] = times[row, n - 1]
    visit_mask = torch.arange(4)[None] < lengths[:, None]
    prefix_mask = torch.arange(4)[None] < torch.tensor([2, 1, 2, 3])[:, None]
    return obs, times, visit_mask, prefix_mask, torch.tensor([0.0, 1.0, 1.0, 0.0])


def test_latent_ode_loss_is_finite_and_kl_nonnegative():
    torch.manual_seed(0)
    model = small_latent_ode()
    obs, times, visit_mask, prefix_mask, condition = _toy_batch()
    loss, parts = model.loss(obs, times, visit_mask, prefix_mask, condition, kl_weight=1.0, free_bits=0.0)
    assert torch.isfinite(loss) and float(parts["kl"]) >= 0.0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_latent_ode_ignores_unobserved_and_padded_entries():
    torch.manual_seed(0)
    model = small_latent_ode()
    obs, times, _visit_mask, prefix_mask, condition = _toy_batch()
    tampered = obs.clone()
    tampered[~prefix_mask] = 123.0
    mu_a, logvar_a = model.encode(obs, times, prefix_mask, condition)
    mu_b, logvar_b = model.encode(tampered, times, prefix_mask, condition)
    assert torch.equal(mu_a, mu_b) and torch.equal(logvar_a, logvar_b)


def test_latent_ode_encoder_is_order_and_time_aware():
    torch.manual_seed(0)
    model = small_latent_ode()
    _randomize(model, 4)
    obs = torch.randn(1, 3, 6)
    times = torch.tensor([[0.0, 0.4, 0.8]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    d = torch.tensor([1.0])
    base, _ = model.encode(obs, times, mask, d)
    swapped, _ = model.encode(obs[:, [2, 1, 0]], times, mask, d)
    stretched, _ = model.encode(obs, times * 2.0, mask, d)
    assert not torch.allclose(base, swapped) and not torch.allclose(base, stretched)


def test_latent_ode_condition_reaches_the_prediction():
    torch.manual_seed(0)
    model = small_latent_ode()
    _randomize(model, 5)
    model.eval()
    z, s, t = torch.randn(2, 6), torch.tensor([0.1, 0.1]), torch.tensor([0.6, 0.6])
    assert not torch.allclose(model.transport(z, s, t, torch.zeros(2)), model.transport(z, s, t, torch.ones(2)))
    blind = small_latent_ode(**{"model.condition_in_encoder": False, "model.condition_in_dynamics": False})
    blind.load_state_dict(model.state_dict())
    blind.eval()
    assert torch.allclose(blind.transport(z, s, t, torch.zeros(2)), blind.transport(z, s, t, torch.ones(2)))


def test_latent_ode_overfits_a_tiny_batch():
    torch.manual_seed(0)
    model = small_latent_ode()
    obs, times, visit_mask, prefix_mask, condition = _toy_batch(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        loss, parts = model.loss(obs, times, visit_mask, prefix_mask, condition, kl_weight=0.0, free_bits=0.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = float(parts["code_mse"]) if first is None else first
    assert float(parts["code_mse"]) < 0.2 * first, (first, float(parts["code_mse"]))


# --------------------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------------------


def test_residual_latent_ode_starts_at_no_change_and_is_anchored():
    torch.manual_seed(0)
    fresh = small_latent_ode(**{"model.residual": True})
    fresh.eval()
    z = torch.randn(3, 6)
    s, t, d = torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.5, 0.1, 0.9]), torch.tensor([1.0, 0.0, 1.0])
    assert torch.equal(fresh.transport(z, s, t, d), z), "zero-initialized vector field must predict exactly no-change"
    obs = torch.randn(2, 3, 6)
    times = torch.tensor([[0.0, 0.3, 0.6], [0.0, 0.2, 0.2]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    last = torch.stack((obs[0, 2], obs[1, 1]))
    assert torch.allclose(fresh.predict(obs, times, mask, torch.tensor([0.0, 1.0]), torch.tensor([[1.0], [0.9]]))[:, 0], last)
    trained_like = small_latent_ode(**{"model.residual": True})
    _randomize(trained_like, 6)
    trained_like.eval()
    assert torch.allclose(trained_like.transport(z, s, s, d), z, atol=1e-6), "prediction at the reference time must be the observation"
    assert not torch.allclose(trained_like.transport(z, s, t, d), z)
    loss, parts = trained_like.loss(*_toy_batch(2)[:4], torch.tensor([0.0, 1.0, 1.0, 0.0]), kl_weight=1.0, free_bits=0.1)
    assert torch.isfinite(loss)


def test_recipes_cover_all_cells():
    for method in D.METHODS:
        for representation in bc.REPRESENTATIONS:
            config = D.load_recipe(method, representation)
            assert config["method"] == method and config["representation"] == representation
    assert D.load_recipe("latent_ode", "pca128")["model"]["residual"] is False
    assert D.load_recipe("latent_ode_residual", "pca128")["model"]["residual"] is True
    faithful, residual = (bc.read_json(D.RECIPE_DIR / f"{m}.json")["search"] for m in D.LATENT_ODE_METHODS)
    assert faithful["space"] == residual["space"] and faithful["sampler_seed"] == residual["sampler_seed"], "trials must be paired"
    assert D.load_recipe("direct_c4", "adaptive128")["training"]["batch_size"] == 8
    assert D.load_recipe("direct_c4", "spiralnet128")["training"]["batch_size"] == 24
    assert D.load_recipe("plain_ode", "lamm128")["loss"] == {"latent_trajectory_mse_weight": 1.0}
    recipe = bc.read_json(D.RECIPE_DIR / "latent_ode.json")
    for dotted in recipe["search"]["space"]:
        node = recipe["config"]
        for key in dotted.split("."):
            node = node[key]


def test_leakage_guard_and_view_registry():
    registry = D.view_registry("p0_internal_adni")
    assert registry["output_root"] == registry["source_sequence_root"] == str(D.view_root("p0_internal_adni"))
    assert "source_integrity" not in registry
    train = bc.load_npz(D.view_root("p0_internal_adni") / "dataset" / "train_subject_sequences.npz")
    test = bc.load_npz(D.view_root("p0_internal_adni") / "dataset" / "test_subject_sequences.npz")
    D.assert_no_test_leakage("p0_internal_adni", [train])
    try:
        D.assert_no_test_leakage("p0_internal_adni", [test])
    except RuntimeError:
        pass
    else:
        raise AssertionError("the guard must refuse an archive containing test scans")


def test_seeded_construction_is_deterministic():
    C.set_seed(42)
    a = M.DirectC4Flow(128, 32, 1, 0.3)
    C.set_seed(42)
    b = M.DirectC4Flow(128, 32, 1, 0.3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_bootstrap_interval_brackets_the_mean():
    values = np.random.default_rng(0).normal(1.0, 0.2, size=60)
    low, high = EV.bootstrap_ci(values, 2000)
    assert low < values.mean() < high and high - low < 0.2


def run_all() -> int:
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
