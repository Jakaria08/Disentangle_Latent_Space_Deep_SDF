#!/usr/bin/env python3
"""Fast contracts for the latent-preserving compact-objective experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TASK_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare as Compare  # noqa: E402
import objective as R  # noqa: E402
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
    assert len(jobs) == 33
    assert len({job["id"] for job in jobs}) == 33
    assert sum(job["representation"] == "pca128" for job in jobs) == 15
    assert sum(job["representation"] == "lamm128" for job in jobs) == 6
    for job in jobs:
        validate_job(job)
        assert job["scientific_contract"]["latent_only"] is True
        assert job["scientific_contract"]["test_loaded_during_training"] is False

    pca_low = next(
        job
        for job in jobs
        if job["representation"] == "pca128"
        and job["arm"] == "compact6_pca_low"
        and job["training"]["seed"] == 42
    )
    assert pca_low["training"]["learning_rate_profile"] == "pca_low"
    assert pca_low["training"]["peak_learning_rate"] == 1.0e-4
    assert pca_low["loss_count"] == 6
    assert pca_low["active_leaf_count"] == 12

    _, pilot = Sweep.selected_jobs(
        SimpleNamespace(
            experiment=None,
            representation="pca128,lamm128",
            arm="compact6_standard,compact6_pca_low",
            objective=None,
            seed="42",
        )
    )
    assert len(pilot) == 3
    assert {job["arm"] for job in pilot} == {
        "compact6_standard",
        "compact6_pca_low",
    }
    assert {job["training"]["seed"] for job in pilot} == {42}

    weights = experiment["loss_weights"]
    pair_names = (
        "real_latent",
        "real_vertex",
        "real_coordinate",
        "real_euclidean",
        "observed_semigroup",
        "virtual_semigroup",
        "inverse",
        "volume",
        "rate",
        "group_rate",
        "disease_gap",
    )
    sequence_names = (
        "sequence_latent",
        "sequence_vertex",
        "sequence_semigroup",
        "slope",
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
        for name in ("full13", "compact6", "compact5")
    }
    assert {name: len(value) for name, value in groups.items()} == R.LOSS_COUNTS
    assert R.ACTIVE_LEAF_COUNTS == {"full13": 13, "compact6": 12, "compact5": 10}
    leaf = R.weighted_leaves(pair, sequence, weights, 1.0, 1.0)
    totals = {
        name: torch.stack(tuple(value.values())).sum()
        for name, value in groups.items()
    }
    assert torch.allclose(totals["compact6"], totals["full13"] - leaf["slope"])
    assert torch.allclose(
        totals["compact5"],
        totals["compact6"] - leaf["group_rate"] - leaf["disease_gap"],
    )
    assert torch.allclose(
        groups["compact6"]["endpoint_prediction"],
        leaf["real_vertex"] + leaf["real_latent"],
    )
    assert torch.allclose(
        groups["compact6"]["sequence_prediction"],
        leaf["sequence_vertex"] + leaf["sequence_latent"],
    )
    totals["compact5"].backward()

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
    assert torch.equal(
        candidate.transport(latent, source, source, label), latent
    )

    standard = next(
        job["training"] for job in jobs if job["arm"] == "compact6_standard"
    )
    low = pca_low["training"]
    total_steps = standard["epochs"] * standard["steps_per_epoch"]
    assert abs(learning_rate(128, total_steps, standard) - 3.0e-4) < 1.0e-12
    assert abs(learning_rate(128, total_steps, low) - 1.0e-4) < 1.0e-12
    assert abs(learning_rate(total_steps, total_steps, standard) - 3.0e-5) < 1.0e-12
    assert abs(learning_rate(total_steps, total_steps, low) - 1.0e-5) < 1.0e-12

    registry = json.loads((TASK_ROOT / "configs" / "representations.json").read_text())
    assert registry["output_root"].endswith("task10_latent_cocycle_compact_v1")
    assert set(registry["representations"]) == {
        "pca128",
        "spiralnet128",
        "adaptive128",
        "lamm128",
    }

    base_metrics = {
        "legacy_score": 1.0,
        "macro_shape_ratio": 0.8,
        "decoded_target_error_mm": 0.30,
        "observed_mesh_error_mm": 0.32,
        "group_rate_error": 0.003,
        "semigroup_defect": 0.01,
        "inverse_defect": 0.02,
    }
    task10_records = []
    for job in jobs:
        task10_records.append(
            {
                "source": "task10",
                "representation": job["representation"],
                "arm": job["arm"],
                "objective": job["objective"],
                "loss_count": job["loss_count"],
                "active_leaf_count": job["active_leaf_count"],
                "learning_rate_profile": job["training"]["learning_rate_profile"],
                "seed": job["training"]["seed"],
                "best_epoch": 5,
                "best_any_epoch": 5,
                "early_checkpoint_better": False,
                **base_metrics,
            }
        )
    controls = [
        {
            "source": "task9",
            "representation": representation,
            "arm": "task9_full13_standard",
            "objective": "full13",
            "loss_count": 13,
            "active_leaf_count": 13,
            "learning_rate_profile": "standard",
            "seed": seed,
            "best_epoch": 5,
            "best_any_epoch": 5,
            "early_checkpoint_better": False,
            **base_metrics,
        }
        for representation in experiment["representations"]
        for seed in experiment["seeds"]
    ]
    historical = [
        {"representation": representation, **base_metrics}
        for representation in experiment["representations"]
    ]
    aggregate, decision = Compare.aggregate(
        experiment, task10_records, controls, historical
    )
    assert len(aggregate) == 11
    assert decision["complete"] is True
    assert decision["recommended_recipe"]["objective"] == "compact5"
    assert decision["recommended_recipe"]["arms"]["pca128"] == "compact5_pca_low"
    assert decision["recommended_recipe"]["arms"]["lamm128"] == "compact5_standard"
    assert len(decision["nested_compact5_minus_compact6"]) == 5
    assert all(
        contrast["paired_legacy_score_delta"] == 0.0
        for contrast in decision["nested_compact5_minus_compact6"]
    )

    task9_config_path = C.resolve_path(
        experiment["external_controls"]["task9_experiment_config"]
    )
    task9_experiment = json.loads(task9_config_path.read_text())
    Compare.validate_task9_control_contract(experiment, task9_experiment)

    print(
        "PASS: 33 jobs, exact full13 control, latent-preserving compact6/compact5, "
        "paired PCA LR control, fixed model, and preregistered recipe selector"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
