#!/usr/bin/env python3
"""Stage 5 (step 2) tests: converter dose path, C1 reduction to direct_c4, onset rules, objective fidelity against
task3, the evaluation adapter, onset recovery on noiseless data, and the converter view.

Needs torch; run with the pytorch_geo environment (no pytest there; the file runs itself):
    /home/jakaria/anaconda3/envs/pytorch_geo/bin/python tests/test_stage5_converter.py
CPU only. Reads stage-1 views; writes nothing.
"""

from __future__ import annotations

import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_common as bc  # noqa: E402
import converter_line as L  # noqa: E402
import dynamics_core as D  # noqa: E402

PARTS = D.core()
M, C, O = PARTS["M"], PARTS["C"], PARTS["O"]
CPU = torch.device("cpu")
CONFIG = bc.read_json(bc.CONFIG_DIR / "converter_line.json")


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _flow(seed: int = 0) -> torch.nn.Module:
    flow = M.DirectC4Flow(128, 64, 1, 0.0).eval()
    _randomize(flow, seed)
    return flow


def _toy_archive() -> dict[str, np.ndarray]:
    """Five subjects: CN-stable, AD-stable, MCI->AD, CN->AD, CN->MCI (4 visits each)."""
    groups = ["CN-stable", "AD-stable", "MCI->AD", "CN->AD", "CN->MCI"]
    labels = [["CN"] * 4, ["AD"] * 4, ["MCI", "MCI", "AD", "AD"], ["CN", "AD", "AD", "AD"], ["CN", "CN", "MCI", "MCI"]]
    ages = np.concatenate([0.3 + 0.05 * np.arange(4) + 0.01 * index for index in range(5)]).astype(np.float32)
    return {"subject_ids": np.asarray([f"toy:{index}" for index in range(5)]), "subject_trajectory_groups": np.asarray(groups),
            "subject_visit_offsets": np.arange(0, 21, 4, dtype=np.int64), "visit_trajectory_labels": np.asarray(sum(labels, [])),
            "visit_age_norm_train": ages}


def test_dose_additivity_and_limits():
    generator = torch.Generator().manual_seed(1)
    s = torch.rand(10000, generator=generator, dtype=torch.float64)
    t = s + torch.rand(10000, generator=generator, dtype=torch.float64)
    u = s + (t - s) * torch.rand(10000, generator=generator, dtype=torch.float64)
    onset = torch.rand(10000, generator=generator, dtype=torch.float64) * 2 - 0.5
    for width in (0.25 / 30.0, 0.5 / 30.0, 0.1):
        lhs = (u - s) * L.average_dose(s, u, onset, width) + (t - u) * L.average_dose(u, t, onset, width)
        rhs = (t - s) * L.average_dose(s, t, onset, width)
        assert float((lhs - rhs).abs().max()) <= 1e-12, "gate G5.4: dose additivity"
        dose = L.average_dose(s, t, onset, width)
        assert float(dose.min()) >= -1e-12 and float(dose.max()) <= 1.0 + 1e-12, "a dose is a probability (up to float64 round-off)"
        assert torch.allclose(L.average_dose(s, s, onset, width), torch.sigmoid((s - onset) / width))
        assert torch.allclose(L.average_dose(t, s, onset, width), dose), "the average dose is symmetric in its end points"


def test_c1_is_direct_c4_on_stable_subjects():
    archive = _toy_archive()
    table = L.build_condition_table(archive, "learned")
    flow = _flow()
    model = L.ConverterCocycle(flow, table, width=0.01).eval()
    with torch.no_grad():
        model.theta.copy_(torch.randn(model.theta.shape))
    generator = torch.Generator().manual_seed(2)
    visits = torch.arange(0, 8)  # CN-stable and AD-stable visits
    z = torch.randn(len(visits), 128, generator=generator)
    s = torch.rand(len(visits), generator=generator)
    t = s + 0.3
    labels = torch.tensor([0.0] * 4 + [1.0] * 4)
    with torch.no_grad():
        assert torch.equal(model.transport_visits(z, s, t, visits), flow.transport(z, s, t, labels)), "gate G5.4: C1 == direct_c4 on stable"


def test_onset_rules_never_use_future_labels():
    archive = _toy_archive()
    ages = archive["visit_age_norm_train"].astype(np.float64)
    learned = L.build_condition_table(archive, "learned")
    assert learned.theta_subjects == ["toy:2", "toy:3"] and learned.theta_count == 2
    assert (learned.mode[8:16] == L.MODE_THETA).all() and (learned.mode[16:20] == L.MODE_CONSTANT).all()
    assert np.isclose(learned.window_a[8], ages[9]) and np.isclose(learned.window_b[8], ages[10])
    assert np.isclose(learned.window_a[12], ages[12]) and np.isclose(learned.window_b[12], ages[13])
    oracle = L.build_condition_table(archive, "oracle", theta_count=2)
    assert oracle.learned_onset_visits() == 0 and np.isclose(oracle.fixed_onset[8], 0.5 * (ages[9] + ages[10]))
    prefixes = {index: [0] for index in range(5)}
    first = L.build_condition_table(archive, "prefix", prefixes, theta_count=2)
    assert (first.mode == L.MODE_CONSTANT).all()
    assert first.constant[8] == 0.0 and first.constant[12] == 0.0 and first.constant[4] == 1.0, "no conversion visible in the prefix"
    prefixes = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [1, 2], 4: [0, 1, 2]}
    later = L.build_condition_table(archive, "prefix", prefixes, theta_count=2)
    assert later.mode[8] == L.MODE_FIXED and np.isclose(later.fixed_onset[8], 0.5 * (ages[9] + ages[10]))
    assert later.mode[12] == L.MODE_CONSTANT and later.constant[12] == 1.0, "a prefix that starts after conversion is constant AD"
    partial = L.build_condition_table(archive, "prefix", prefixes, partial_dose=True, theta_count=2)
    assert partial.partial[16:20].all() and partial.mode[16] == L.MODE_FIXED and np.isclose(partial.fixed_onset[16], 0.5 * (ages[17] + ages[18]))


def test_onset_gradient_reaches_only_train_converters():
    archive = _toy_archive()
    model = L.ConverterCocycle(_flow(3), L.build_condition_table(archive, "learned"), width=0.01)
    visits = torch.arange(20)
    s = torch.as_tensor(archive["visit_age_norm_train"])
    loss = model.transport_visits(torch.randn(20, 128), s, s + 0.2, visits).square().sum() + model.onset_penalty()
    loss.backward()
    assert model.theta.grad is not None and bool((model.theta.grad != 0).all())
    assert model.kappa_logit.grad is None, "kappa is frozen unless the partial-dose variant is trained"
    test_table = L.build_condition_table(archive, "oracle", theta_count=model.theta.numel())
    model.set_table(test_table)
    before = model.theta.detach().clone()
    model.zero_grad(set_to_none=True)
    model.transport_visits(torch.randn(20, 128), s, s + 0.2, visits).square().sum().backward()
    assert model.theta.grad is None or bool((model.theta.grad == 0).all()), "gate G5.5: evaluation never touches learned onsets"
    assert torch.equal(before, model.theta.detach())


def _pooled_values(pairs: int = 24):
    registry = D.view_registry("p3_pooled")
    train = C.load_archive("pca128", "train", registry)
    geometry = C.build_geometry("pca128", train, CPU, registry)
    values = C.values_on_device(train, CPU)
    O.attach_reference_geometry(values, geometry, 512)
    values["stable_group"] = torch.as_tensor(train["visit_label_ad"].astype(np.int64))
    rows = C.load_pairs("train", train, registry)
    rows = O.balanced_subset(rows, pairs)
    statistics = O.training_statistics(values, rows, train, pairs)
    return train, geometry, values, rows, statistics


def test_converter_objective_matches_task3_with_fixed_labels():
    train, geometry, values, rows, statistics = _pooled_values()
    flow = _flow(4)
    condition = L.label_condition(values)
    raw = C.collate_pairs(rows[:16])
    torch.manual_seed(7)
    reference = O.pair_terms(flow, geometry, values, raw, statistics)
    torch.manual_seed(7)
    mine = L.pair_terms(flow, geometry, values, raw, statistics, condition)
    assert reference.keys() == mine.keys()
    for key in reference:
        assert torch.allclose(reference[key], mine[key], rtol=1e-6, atol=1e-8), key
    start = O.balanced_sequence_starts(train, 42, 1)[0]
    reference = O.sequence_terms(flow, geometry, values, train, start, statistics)
    mine = L.sequence_terms(flow, geometry, values, train, start, statistics, condition)
    for key in reference:
        assert torch.allclose(reference[key], mine[key], rtol=1e-6, atol=1e-8), key
    reference = O.cocycle_defects(flow, values, rows, statistics, 8)
    mine = L.cocycle_defects(flow, values, rows, statistics, 8, condition)
    assert all(np.isclose(reference[key], mine[key], rtol=1e-6) for key in reference)


def test_evaluation_adapter_matches_task3():
    registry = D.view_registry("p3_pooled")
    val = C.load_archive("pca128", "val", registry)
    train, geometry, _values, _rows, _statistics = _pooled_values()
    values = C.values_on_device(val, CPU)
    O.attach_reference_geometry(values, geometry, 512)
    rows = C.first_last_pairs(val)[:20]
    raw = D.view_vertices(val)
    flow = _flow(5)
    reference = O.evaluate_pairs(flow, geometry, values, rows, raw, 64)["groups"]["overall"]
    same_chunk = L.evaluate_rows(flow, geometry, values, rows, raw, 64, L.label_condition(values))
    assert same_chunk["rows"] == reference["rows"] and all(same_chunk[key] == reference[key] for key in reference if key.endswith("_mean"))
    # Smaller chunks change float32 batch numerics; the signed log-volume rate (~5e-4) is the most sensitive quantity.
    chunked = L.evaluate_rows(flow, geometry, values, rows, raw, 7, L.label_condition(values))
    assert all(np.isclose(chunked[key], reference[key], rtol=1e-4, atol=1e-7) for key in reference if key.endswith("_mean"))


def test_onset_posterior_recovers_a_noiseless_onset():
    flow = _flow(6)
    with torch.no_grad():
        flow.ad_residual_head.weight.mul_(20.0)
    baseline = torch.randn(128, generator=torch.Generator().manual_seed(8))
    ages = torch.tensor([0.40, 0.45, 0.50, 0.55])
    width = 0.25 / 30.0
    truth = torch.tensor([0.47])
    observed = L.dosed_trajectory(flow, baseline, 0.35, ages, truth, width)[0]
    grid, posterior = L.onset_posterior(flow, baseline, 0.35, ages, observed, 1e-3, (0.42, 0.52), width, 201)
    summary = L.posterior_summary(grid, posterior, (0.1, 0.9))
    step = (0.52 - 0.42) / 200
    assert abs(summary["median"] - 0.47) <= 1.5 * step
    assert summary["low"] - 1e-6 <= 0.47 <= summary["high"] + 1e-6  # the float32 grid point is 0.46999999


def test_converter_view():
    view = CONFIG["view"]
    root = D.view_root(view)
    import evaluate_dynamics as EV
    for rep in CONFIG["representations"]:
        EV.normalization_guard(CONFIG["base_view"], view, rep)
        for split in bc.SPLITS:
            C.load_archive(rep, split, D.view_registry(view))
    archives = {split: bc.load_npz(root / "dataset" / f"{split}_subject_sequences.npz") for split in bc.SPLITS}
    counts = Counter(g for a in archives.values() for c, g in zip(a["subject_cohorts"], a["subject_trajectory_groups"]) if c != "adni")
    expected = Counter()
    for cohort in ("aibl", "oasis"):
        frame = pd.read_csv(bc.STAGE1_ROOT / "cohorts" / cohort / "inclusive_manifest.csv", dtype={"subject_key": str})
        expected.update(frame.drop_duplicates("subject_key")["trajectory_group"].loc[lambda s: s.isin(CONFIG["groups"]["kept"])].tolist())
    assert counts == expected, (counts, expected)
    for split, archive in archives.items():
        table = L.build_condition_table(archive, "learned" if split == "train" else "oracle", theta_count=0 if split != "train" else None)
        offsets = archive["subject_visit_offsets"]
        years = archive["visit_time_years_from_baseline"].astype(np.float64)
        for index, group in enumerate(archive["subject_trajectory_groups"].astype(str)):
            if group not in L.AD_CONVERTERS:
                continue
            labels = archive["visit_trajectory_labels"][offsets[index]:offsets[index + 1]].tolist()
            first = labels.index("AD")
            a, b = archive["subject_conv_to_ad_window_years"][index]
            assert np.isclose(years[offsets[index] + first - 1], a, atol=1e-4) and np.isclose(years[offsets[index] + first], b, atol=1e-4)
        assert (table.mode != L.MODE_CONSTANT).sum() == sum((np.diff(offsets)[i] for i, g in enumerate(archive["subject_trajectory_groups"]) if g in L.AD_CONVERTERS))


def test_pseudo_pairs_respect_the_age_window_and_alpha_is_the_target():
    import train_brainode_full as BF
    rng = np.random.default_rng(0)
    ages = np.concatenate([rng.uniform(60, 90, 40), rng.uniform(60, 90, 20)])
    labels = np.asarray([0] * 40 + [1] * 20)
    cn, ad, alpha = BF.pseudo_pairs(ages, labels, 500, 5.0, rng)
    assert (labels[cn] == 0).all() and (labels[ad] == 1).all() and len(alpha) == 500
    assert np.abs(ages[cn] - ages[ad]).max() <= 5.0 and alpha.min() >= 0.0 and alpha.max() <= 1.0
    logits, targets = torch.tensor([0.3, -1.2]), torch.tensor([1.0, 0.0])
    pseudo_logits, soft = torch.tensor([0.1, 0.4, -0.2]), torch.tensor([0.25, 0.5, 0.9])
    weights = torch.tensor([2.0, 1.0])
    loss, observed, pseudo = BF.estimator_loss(logits, targets, weights, pseudo_logits, soft)
    manual_observed = (2 * torch.nn.functional.binary_cross_entropy_with_logits(logits[:1], targets[:1])
                       + torch.nn.functional.binary_cross_entropy_with_logits(logits[1:], targets[1:])) / 3
    assert torch.allclose(observed, manual_observed) and torch.allclose(pseudo, torch.nn.functional.binary_cross_entropy_with_logits(pseudo_logits, soft))
    assert torch.allclose(loss, 0.5 * (observed + pseudo))


def test_pooled_feedback_matches_task3_feedback():
    from concurrent.futures import ThreadPoolExecutor

    import evaluate_brainode_full as EB
    import train_brainode_full as BF
    B, PV, _TB = BF.task3_modules()
    train, geometry, values, _rows, _statistics = _pooled_values()
    codes = values["z"][:3]
    grid = B.VoxelGrid.from_training_vertices(BF.decode(geometry, values["z"][:64]), 32, 2)
    estimator = B.VoxelCognitionCNN(4, 0.0).eval()
    _randomize(estimator, 9)
    field = M.build_ode(D.load_recipe("brainode", "pca128")).eval()
    PV._initialize_worker(geometry.faces.cpu().numpy(), grid.mapping())
    with ThreadPoolExecutor(1) as executor:
        pooled = EB.pooled_feedback_class(B, PV)(field, geometry, estimator, grid, 2, 1.3, executor=executor)
        reference = B.CognitionFeedbackTransport(field, geometry, estimator, grid, 2, 1.3)
        assert torch.allclose(pooled.estimate_condition(codes), reference.estimate_condition(codes))
        s = values["age"][:3]
        assert torch.allclose(pooled.transport(codes, s, s + 0.05), reference.transport(codes, s, s + 0.05))


def test_converter_job_file():
    import stage5_build_converter_jobs as J
    job_list = J.jobs()
    ids = {job["id"] for job in job_list}
    assert len(ids) == len(job_list) and all(dep in ids for job in job_list for dep in job["after"])
    gates = {job["id"] for job in job_list if job["id"].startswith("gate_G5_6")}
    assert gates and all(gates <= set(job["after"]) for job in job_list if job["id"].startswith("evaluate_test"))
    assert not any(gates & set(job["after"]) for job in job_list if job["id"].startswith(("train__", "evaluate_val")))
    expected_c1 = sum(len(spec["representations"]) * len(CONFIG["seeds"]) for spec in CONFIG["variants"].values())
    assert sum(job["id"].startswith("train__") and "brainode_full" not in job["id"] for job in job_list) == expected_c1
    assert all("--skip-pairs" in job["argv"] for job in job_list if job["argv"][0] == "evaluate_dynamics.py")


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
