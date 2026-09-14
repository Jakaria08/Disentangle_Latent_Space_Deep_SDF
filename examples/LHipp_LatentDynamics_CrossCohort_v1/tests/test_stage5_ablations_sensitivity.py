#!/usr/bin/env python3
"""Stage 5 (step 1) tests: ablation transports, recipe composition, the August coboundary bridge,
the pooled PCA procedure, the stage-1 view writer refactor, and the job files.

Needs torch; the pytorch_geo environment has no pytest, so every test is a plain function and the file
runs itself:
    /home/jakaria/anaconda3/envs/pytorch_geo/bin/python tests/test_stage5_ablations_sensitivity.py
CPU only. Reads stage-1 views; writes only to temporary directories.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_common as bc  # noqa: E402
import dynamics_core as D  # noqa: E402
import rubanova_latent_ode as R  # noqa: E402
import stage1_build_protocol_views as views  # noqa: E402
import stage5_build_jobs as S5  # noqa: E402
import stage5_pooled_pca as P  # noqa: E402

PARTS = D.core()
M, E = PARTS["M"], PARTS["E"]
CPU = torch.device("cpu")


def _randomize(module: torch.nn.Module, seed: int) -> None:
    """Zero-initialized heads would make identities trivially true."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _batch(count: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(count, 128, generator=generator)
    source = torch.rand(count, generator=generator)
    target = source + 0.2 * torch.rand(count, generator=generator)
    condition = torch.arange(count, dtype=torch.float32) % 2
    return z, source, target, condition


def _august_config(key: str) -> dict:
    return bc.read_json(bc.resolve(bc.read_json(bc.CONFIG_DIR / "ablation_sources.json")["files"][key]["path"]))


def test_sensitivity_registry_stays_outside_r1():
    assert tuple(bc.load_registry()["representations"]) == bc.REPRESENTATIONS
    assert "pca128_pooled" in D.view_registry("p3_pooled")["representations"]
    assert "pca128_pooled" not in D.view_registry("p0_internal_adni")["representations"]
    for method in D.METHODS:
        pooled, base = D.load_recipe(method, "pca128_pooled"), D.load_recipe(method, "pca128")
        assert pooled.pop("representation") == "pca128_pooled" and base.pop("representation") == "pca128"
        assert pooled == base


def test_ablation_recipes_are_composed_from_the_main_recipes():
    c4, brainode = D.load_recipe("direct_c4", "pca128"), D.load_recipe("brainode", "pca128")
    a3 = D.load_recipe("direct_c4_no_disease", "pca128")
    assert a3["model"].pop("disease_head") is False and a3["model"] == c4["model"]
    assert all(a3[key] == c4[key] for key in ("training", "loss", "selection"))
    a2 = D.load_recipe("brainode_v", "pca128")
    assert all(a2[key] == c4[key] for key in ("loss", "selection"))
    assert a2["training"] == c4["training"] | {"integration_substeps": brainode["training"]["integration_substeps"]}
    assert all(a2["model"][key] == value for key, value in brainode["model"].items())
    assert a2["model"]["variant"] == "brainode_v" and a2["model"]["ode_used"] is True


def test_no_disease_cocycle_ignores_the_condition():
    config = D.load_recipe("direct_c4_no_disease", "pca128")
    flow = D.build_transport(config, CPU).eval()
    _randomize(flow, 1)
    reference = M.DirectC4Flow(128, int(config["model"]["width"]), int(config["model"]["residual_blocks"]), 0.3).eval()
    reference.load_state_dict(flow.state_dict())
    z, s, t, _d = _batch()
    ones, zeros = torch.ones(len(z)), torch.zeros(len(z))
    with torch.no_grad():
        assert torch.equal(flow.transport(z, s, t, ones), flow.transport(z, s, t, zeros))
        assert torch.equal(flow.transport(z, s, t, ones), reference.transport(z, s, t, zeros))
        assert not torch.allclose(reference.transport(z, s, t, ones), reference.transport(z, s, t, zeros))


def test_brainode_v_is_brainodes_rk4_transport_and_round_trips():
    config = D.load_recipe("brainode_v", "pca128")
    transport = D.build_transport(config, CPU).eval()
    _randomize(transport, 2)
    field = M.build_ode(D.load_recipe("brainode", "pca128")).eval()
    field.load_state_dict(transport.function.state_dict())
    reference = E.ODETransport(field, int(config["training"]["integration_substeps"])).eval()
    z, s, t, d = _batch()
    with torch.no_grad():
        assert torch.equal(transport.transport(z, s, t, d), reference.transport(z, s, t, d))
        assert torch.allclose(transport.transport(z, s, s, d), z)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pt"
            torch.save({"config": config, "flow_state_dict": transport.state_dict()}, path)
            loaded, loaded_config, _payload = D.load_trained_transport(path, CPU)
            assert loaded_config["method"] == "brainode_v"
            assert torch.equal(loaded.transport(z, s, t, d), transport.transport(z, s, t, d))


def test_unconditional_latent_ode_ignores_the_condition():
    overrides = {"model.z0_dim": 8, "model.encoder_hidden": 16, "model.encoder_ode_width": 16, "model.dynamics_width": 16,
                 "model.decoder_width": 16, "model.latent_dim": 6, "training.integration_substeps": 2,
                 "model.condition_in_encoder": False, "model.condition_in_dynamics": False}
    generator = torch.Generator().manual_seed(3)
    obs = torch.randn(3, 4, 6, generator=generator)
    times = torch.sort(torch.rand(3, 4, generator=generator), dim=1).values
    mask = torch.ones(3, 4, dtype=torch.bool)
    target = times[:, -1:] + 0.1
    for method in D.LATENT_ODE_METHODS:
        model = R.LatentODE.from_config(D.load_recipe(method, "pca128", overrides)).eval()
        _randomize(model, 3)
        with torch.no_grad():
            assert torch.equal(model.predict(obs, times, mask, torch.ones(3), target), model.predict(obs, times, mask, torch.zeros(3), target))


def test_coboundary_bridge_is_pinned_and_passes_subject_context():
    D.coboundary_modules("exact_coboundary_c4")
    D.coboundary_modules("volume_exact_coboundary_c4_v2")
    assert Path(sys.modules["common"].__file__).resolve().parent == D.T3_SCRIPTS.resolve()
    assert Path(sys.modules["c4_objective"].__file__).resolve().parent == D.T3_SCRIPTS.resolve()
    assert not any("August_Version" in entry for entry in sys.path), "August script folders must not stay on sys.path"
    z, s, t, d = _batch()
    context, context_time = torch.randn_like(z), s - 0.05
    exact = D.build_transport(_august_config("config_exact_pca128"), CPU).eval()
    coefficient = torch.randn(128, generator=torch.Generator().manual_seed(5)).tolist()
    volume = D.build_transport(_august_config("config_volume_v2_pca128"), CPU, {"volume_axis": {"coefficient": coefficient}}).eval()
    for seed, flow in ((4, exact), (5, volume)):
        _randomize(flow, seed)
        with torch.no_grad():
            assert torch.allclose(flow.transport(z, s, s, d, context, context_time), z, atol=1e-4)
            middle = 0.5 * (s + t)
            chained = flow.transport(flow.transport(z, s, middle, d, context, context_time), middle, t, d, context, context_time)
            assert torch.allclose(chained, flow.transport(z, s, t, d, context, context_time), atol=1e-4)
            values = {"z": z, "age": s, "label": d, "context": context, "context_age": context_time}
            source, target = torch.arange(len(z)), torch.roll(torch.arange(len(z)), 1)
            expected = flow.transport(z, s, s[target], d, context, context_time)
            assert torch.equal(D.transport_call(flow, "exact_coboundary_c4", values, source, target), expected)
        try:
            flow.transport(z, s, t, d)
        except ValueError:
            pass
        else:
            raise AssertionError("a coboundary transport must refuse to run without the subject context")
    try:
        D.build_transport(_august_config("config_volume_v2_pca128"), CPU)
    except ValueError:
        pass
    else:
        raise AssertionError("the volume coboundary must require its train-only volume axis")


def test_pca_fit_is_centered_orthonormal_and_matches_the_covariance_spectrum():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(80, 5)) @ rng.normal(size=(5, 40)) + rng.normal(size=40)
    model = P.fit_pca(data, components=5)
    assert np.allclose(model["components"] @ model["components"].T, np.eye(5), atol=1e-10)
    spectrum = np.sort(np.linalg.eigvalsh(np.cov(data, rowvar=False)))[::-1][:5]
    assert np.allclose(model["explained_variance"], spectrum, rtol=1e-8)
    decoded = (data - model["mean"]) @ model["components"].T @ model["components"] + model["mean"]
    assert np.allclose(decoded, data, atol=1e-8)
    assert (model["components"][np.arange(5), np.argmax(np.abs(model["components"]), axis=1)] > 0).all()


def test_view_representation_writer_reproduces_stage1_archives():
    registry = bc.load_registry()
    root = D.view_root("p3_pooled")
    archives = {split: bc.load_npz(root / "dataset" / f"{split}_subject_sequences.npz") for split in bc.SPLITS}
    cohorts = sorted({str(cohort) for archive in archives.values() for cohort in archive["visit_cohorts"]})
    latents = {(cohort, "pca128"): views.load_latents(cohort, "pca128") for cohort in cohorts}
    with tempfile.TemporaryDirectory() as folder:
        files = views.write_view_representation(Path(folder), archives, "pca128", registry["representations"]["pca128"], latents)
        assert set(files) == {f"representations/pca128/{split}" for split in bc.SPLITS}
        for split in bc.SPLITS:
            new = bc.load_npz(Path(folder) / "representations" / "pca128" / f"{split}_subject_sequences_128.npz")
            old = bc.load_npz(root / "representations" / "pca128" / f"{split}_subject_sequences_128.npz")
            assert set(new) == set(old) and all(np.array_equal(new[key], old[key]) for key in old)


def test_stage5_job_files():
    runs = S5.ablation_runs()
    assert Counter(run["ablation"] for run in runs) == {"A1": 4, "A2": 3, "A3": 3, "A4": 6}
    jobs = S5.ablation_jobs(runs)
    ids = {job["id"] for job in jobs}
    assert len(ids) == len(jobs)
    for run in runs:
        assert {f"train__{run['key']}", f"evaluation_val__{run['key']}", f"evaluation_test__{run['key']}", f"sweep__{run['key']}"} <= ids
    for job in jobs:
        assert all(dependency in ids for dependency in job["after"]) and "--device" not in job["argv"]
        if job["id"].startswith("train__"):
            assert "test" not in job["argv"]
    assert all("model.condition_in_dynamics=false" in run["train_argv"] for run in runs if run["ablation"] == "A4")
    pooled = S5.pooled_runs()
    assert len(pooled) == 15 and {run["representation"] for run in pooled} == {"pca128_pooled"}
    min_epoch, missing = S5.min_epoch_runs()
    assert len(min_epoch) + len(missing) == 42 and all(run["method"] == "direct_c4" and not run["blocked"] for run in min_epoch)
    sensitivity = S5.sensitivity_jobs(pooled, min_epoch)
    sensitivity_ids = {job["id"] for job in sensitivity}
    assert len(sensitivity_ids) == len(sensitivity)
    assert all(dependency in sensitivity_ids for job in sensitivity for dependency in job["after"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
