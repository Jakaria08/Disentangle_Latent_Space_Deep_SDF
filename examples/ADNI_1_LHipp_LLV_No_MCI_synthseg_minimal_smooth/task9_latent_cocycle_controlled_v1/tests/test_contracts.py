#!/usr/bin/env python3
"""Fast contracts for the controlled latent-loss experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TASK_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import objective as R  # noqa: E402
import compare as Compare  # noqa: E402
import sweep as Sweep  # noqa: E402
from _bootstrap import CORE_SCRIPTS, core  # noqa: E402
from configuration import all_jobs, read_experiment, validate_job  # noqa: E402
from model import DirectLatentCocycle  # noqa: E402
from train import learning_rate  # noqa: E402

C, O = core()
if str(CORE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORE_SCRIPTS))
from models import DirectC4Flow  # noqa: E402


def scalar_map(names: tuple[str, ...], offset: float) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor(offset + index / 10.0, requires_grad=True)
        for index, name in enumerate(names, start=1)
    }


def main() -> int:
    experiment = read_experiment()
    jobs = all_jobs(experiment)
    assert len(jobs) == 48
    assert len({job["id"] for job in jobs}) == 48
    for job in jobs:
        validate_job(job)
        assert job["scientific_contract"]["latent_only"] is True
        assert job["scientific_contract"]["test_loaded_during_training"] is False
    adaptive = next(
        job
        for job in jobs
        if job["representation"] == "adaptive128"
        and job["loss_set"] == "full13"
        and job["training"]["seed"] == 42
    )
    assert adaptive["training"]["batch_size"] == 8
    _, pilot = Sweep.selected_jobs(
        SimpleNamespace(
            experiment=None,
            representation="pca128,lamm128",
            loss_set="full13,lean5",
            seed="42",
        )
    )
    assert len(pilot) == 4
    assert {job["representation"] for job in pilot} == {"pca128", "lamm128"}
    assert {job["loss_set"] for job in pilot} == {"full13", "lean5"}
    assert {job["training"]["seed"] for job in pilot} == {42}

    weights = experiment["loss_weights"]
    pair_names = (
        "real_latent", "real_vertex", "real_coordinate", "real_euclidean",
        "observed_semigroup", "virtual_semigroup", "inverse", "volume", "rate",
        "group_rate", "disease_gap",
    )
    sequence_names = (
        "sequence_latent", "sequence_vertex", "sequence_semigroup", "slope",
    )
    pair = scalar_map(pair_names, 0.1)
    sequence = scalar_map(sequence_names, 0.3)
    full, _ = R.total_loss(pair, sequence, weights, "full13")
    baseline_config = {
        "loss": {f"{name}_weight": value for name, value in weights.items()},
        "training": {"consistency_ramp_epochs": 0, "anatomy_ramp_epochs": 0},
    }
    baseline, _ = O.total_loss(pair, sequence, baseline_config, 1)
    assert torch.allclose(full, baseline, rtol=1.0e-6, atol=1.0e-7)

    groups = {
        name: R.loss_groups(pair, sequence, weights, name)
        for name in ("full13", "lean6", "lean5", "lean4")
    }
    assert {name: len(value) for name, value in groups.items()} == R.LOSS_COUNTS
    totals = {name: torch.stack(tuple(value.values())).sum() for name, value in groups.items()}
    leaf = R._weighted_leaves(pair, sequence, weights, 1.0, 1.0)
    assert torch.allclose(
        totals["lean6"],
        totals["full13"] - leaf["real_latent"] - leaf["sequence_latent"] - leaf["slope"],
    )
    assert torch.allclose(totals["lean5"], totals["lean6"] - leaf["volume"])
    assert torch.allclose(
        totals["lean4"], totals["lean5"] - leaf["group_rate"] - leaf["disease_gap"]
    )
    totals["lean4"].backward()

    torch.manual_seed(123)
    reference = DirectC4Flow(128, 256, 2, 0.0)
    torch.manual_seed(123)
    candidate = DirectLatentCocycle(128, 256, 2, 0.0)
    assert list(reference.state_dict()) == list(candidate.state_dict())
    candidate.load_state_dict(reference.state_dict(), strict=True)
    latent = torch.randn(4, 128)
    source = torch.randn(4)
    target = torch.randn(4)
    label = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.equal(
        reference.transport(latent, source, target, label),
        candidate.transport(latent, source, target, label),
    )
    assert torch.equal(candidate.transport(latent, source, source, label), latent)
    assert torch.equal(candidate.transport(latent, source, target, label), latent)

    training = experiment["training"]
    total_steps = int(training["epochs"]) * int(training["steps_per_epoch"])
    assert learning_rate(1, total_steps, training) > float(training["initial_learning_rate"])
    assert learning_rate(int(training["warmup_steps"]), total_steps, training) == float(training["peak_learning_rate"])
    assert abs(learning_rate(total_steps, total_steps, training) - float(training["minimum_learning_rate"])) < 1.0e-12

    registry = json.loads((TASK_ROOT / "configs" / "representations.json").read_text())
    assert set(registry["representations"]) == {
        "pca128", "spiralnet128", "adaptive128", "lamm128"
    }

    synthetic = []
    historical = []
    base_metrics = {
        "legacy_score": 1.0,
        "macro_shape_ratio": 0.8,
        "decoded_target_error_mm": 0.30,
        "observed_mesh_error_mm": 0.32,
        "group_rate_error": 0.003,
        "semigroup_defect": 0.01,
        "inverse_defect": 0.02,
    }
    for representation in experiment["representations"]:
        historical.append({"representation": representation, **base_metrics})
        for loss_set in experiment["loss_sets"]:
            for seed in experiment["seeds"]:
                synthetic.append(
                    {
                        "representation": representation,
                        "loss_set": loss_set,
                        "loss_count": R.LOSS_COUNTS[loss_set],
                        "seed": seed,
                        "best_epoch": 5,
                        "best_any_epoch": 5,
                        "early_checkpoint_better": False,
                        **base_metrics,
                    }
                )
    aggregate, decision = Compare.aggregate(experiment, synthetic, historical)
    assert len(aggregate) == 16
    assert decision["complete"] is True
    assert decision["recommended_loss_set"] == "lean4"
    print("PASS: 48 matched jobs, exact full13 formula, nested 6/5/4 objectives, fixed model, LR schedule, decision rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
