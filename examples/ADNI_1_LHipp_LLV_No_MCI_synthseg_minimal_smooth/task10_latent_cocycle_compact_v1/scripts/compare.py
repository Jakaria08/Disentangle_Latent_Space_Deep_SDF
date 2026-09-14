#!/usr/bin/env python3
"""Compare compact objectives with matched controls and select a recipe."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import core  # noqa: E402
from configuration import all_jobs, read_experiment  # noqa: E402
from objective import ACTIVE_LEAF_COUNTS, LOSS_COUNTS  # noqa: E402

C, _ = core()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def metrics(validation: dict[str, Any]) -> dict[str, float]:
    ratios = validation["ratios"]
    overall = validation["all_pairs"]["groups"]["overall"]
    return {
        "legacy_score": float(validation["score"]),
        "macro_shape_ratio": float(ratios["macro_first_last_shape"]),
        "decoded_target_error_mm": float(overall["euclidean_mean"]),
        "observed_mesh_error_mm": float(overall["end_to_end_euclidean_mean"]),
        "group_rate_error": abs(
            float(ratios["first_last_cn_predicted_signed_rate"])
            - float(ratios["first_last_cn_observed_signed_rate"])
        )
        + abs(
            float(ratios["first_last_ad_predicted_signed_rate"])
            - float(ratios["first_last_ad_observed_signed_rate"])
        ),
        "semigroup_defect": float(
            validation["defects"]["relative_semigroup_defect_mean"]
        ),
        "inverse_defect": float(
            validation["defects"]["relative_inverse_defect_mean"]
        ),
    }


def historical_record(representation: str, root: Path) -> dict[str, Any]:
    status = json.loads((root / "training_status.json").read_text())
    history = [
        json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()
    ]
    best_epoch = int(status["best_epoch"])
    row = next(item for item in history if int(item["epoch"]) == best_epoch)
    return {
        "source": "historical_task3",
        "representation": representation,
        "arm": "historical_full13",
        "objective": "full13",
        "loss_count": 13,
        "active_leaf_count": 13,
        "learning_rate_profile": "historical",
        "seed": 42,
        "best_epoch": best_epoch,
        "best_step": sum(
            int(item["batches"])
            for item in history
            if int(item["epoch"]) <= best_epoch
        ),
        "best_any_epoch": best_epoch,
        "early_checkpoint_better": False,
        **metrics(row["validation"]),
    }


def run_record(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    return {
        "source": "task10",
        "representation": summary["representation"],
        "arm": summary["arm"],
        "objective": summary["objective"],
        "loss_count": int(summary["loss_count"]),
        "active_leaf_count": int(summary["active_leaf_count"]),
        "learning_rate_profile": summary["learning_rate_profile"],
        "seed": int(summary["seed"]),
        "best_epoch": int(summary["best_mature"]["epoch"]),
        "best_step": int(summary["best_mature"]["step"]),
        "best_any_epoch": int(summary["best_any"]["epoch"]),
        "early_checkpoint_better": bool(summary["early_checkpoint_better"]),
        **metrics(summary["best_mature_validation"]),
    }


def task9_control_record(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    if summary["loss_set"] != "full13":
        raise ValueError(f"Expected a task9 full13 control at {path}")
    return {
        "source": "task9",
        "representation": summary["representation"],
        "arm": "task9_full13_standard",
        "objective": "full13",
        "loss_count": 13,
        "active_leaf_count": 13,
        "learning_rate_profile": "standard",
        "seed": int(summary["seed"]),
        "best_epoch": int(summary["best_mature"]["epoch"]),
        "best_step": int(summary["best_mature"]["step"]),
        "best_any_epoch": int(summary["best_any"]["epoch"]),
        "early_checkpoint_better": bool(summary["early_checkpoint_better"]),
        **metrics(summary["best_mature_validation"]),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def validate_task9_control_contract(
    experiment: dict[str, Any], task9: dict[str, Any]
) -> None:
    """Reject reused controls if any matched protocol field has drifted."""
    if "full13" not in task9["loss_sets"]:
        raise ValueError("Task9 configuration has no full13 control")
    for key in ("representations", "seeds", "model", "loss_weights", "selection"):
        if task9[key] != experiment[key]:
            raise ValueError(f"Task9 control mismatch in {key}")
    training_keys = (
        "epochs",
        "steps_per_epoch",
        "batch_size_by_representation",
        "decoder_batch_size_by_representation",
        "evaluation_batch_size_by_representation",
        "warmup_steps",
        "weight_decay",
        "gradient_clip_norm",
        "consistency_ramp_steps",
        "anatomy_ramp_steps",
        "selection_start_step",
        "early_stopping_enabled",
        "fixed_train_evaluation_pairs",
    )
    for key in training_keys:
        if task9["training"][key] != experiment["training"][key]:
            raise ValueError(f"Task9 control mismatch in training.{key}")
    standard = experiment["training"]["learning_rate_profiles"]["standard"]
    for key, value in standard.items():
        if task9["training"][key] != value:
            raise ValueError(f"Task9 control mismatch in training.{key}")


def paired_contrast(
    records: list[dict[str, Any]],
    representation: str,
    minuend_arm: str,
    subtrahend_arm: str,
) -> dict[str, Any]:
    """Return seed-paired metric deltas for a nested objective contrast."""
    left = {
        int(row["seed"]): row
        for row in records
        if row["representation"] == representation and row["arm"] == minuend_arm
    }
    right = {
        int(row["seed"]): row
        for row in records
        if row["representation"] == representation and row["arm"] == subtrahend_arm
    }
    seeds = sorted(set(left) & set(right))
    result: dict[str, Any] = {
        "representation": representation,
        "contrast": f"{minuend_arm}_minus_{subtrahend_arm}",
        "minuend_arm": minuend_arm,
        "subtrahend_arm": subtrahend_arm,
        "seeds": seeds,
    }
    for key in (
        "legacy_score",
        "observed_mesh_error_mm",
        "group_rate_error",
        "semigroup_defect",
        "inverse_defect",
    ):
        result[f"paired_{key}_delta"] = (
            statistics.fmean(
                float(left[seed][key]) - float(right[seed][key])
                for seed in seeds
            )
            if seeds
            else None
        )
    return result


def aggregate(
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
    task9_controls: list[dict[str, Any]],
    historical: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    criteria = experiment["adoption"]
    controls = list(task9_controls) + list(records)
    history_by_representation = {
        row["representation"]: row for row in historical
    }
    groups: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {
        representation: {} for representation in experiment["representations"]
    }
    metric_names = (
        "legacy_score",
        "macro_shape_ratio",
        "decoded_target_error_mm",
        "observed_mesh_error_mm",
        "group_rate_error",
        "semigroup_defect",
        "inverse_defect",
    )

    for arm in experiment["arms"]:
        arm_name = str(arm["name"])
        objective = str(arm["objective"])
        control_name = str(arm["comparison_control"])
        for representation in arm["representations"]:
            rows = [
                row
                for row in records
                if row["representation"] == representation and row["arm"] == arm_name
            ]
            baseline = [
                row
                for row in controls
                if row["representation"] == representation
                and row["arm"] == control_name
            ]
            if not rows:
                continue
            item: dict[str, Any] = {
                "representation": representation,
                "arm": arm_name,
                "objective": objective,
                "learning_rate_profile": arm["learning_rate_profile"],
                "comparison_control": control_name,
                "loss_count": LOSS_COUNTS[objective],
                "active_leaf_count": ACTIVE_LEAF_COUNTS[objective],
                "seeds": len(rows),
                "control_seeds": len(baseline),
                "best_epoch_mean": mean(rows, "best_epoch"),
                "best_epoch_min": min(row["best_epoch"] for row in rows),
                "best_any_after_epoch_one": all(
                    int(row["best_any_epoch"]) > 1 for row in rows
                ),
                "early_checkpoint_better": any(
                    bool(row["early_checkpoint_better"]) for row in rows
                ),
            }
            for key in metric_names:
                item[f"{key}_mean"] = mean(rows, key)
                item[f"{key}_std"] = (
                    statistics.stdev(float(row[key]) for row in rows)
                    if len(rows) > 1
                    else 0.0
                )

            baseline_by_seed = {int(row["seed"]): row for row in baseline}
            paired = [
                (row, baseline_by_seed[int(row["seed"])])
                for row in rows
                if int(row["seed"]) in baseline_by_seed
            ]
            history = history_by_representation[representation]
            if paired:
                for key in (
                    "legacy_score",
                    "observed_mesh_error_mm",
                    "group_rate_error",
                    "semigroup_defect",
                    "inverse_defect",
                ):
                    item[f"paired_{key}_delta"] = statistics.fmean(
                        float(row[key]) - float(control[key])
                        for row, control in paired
                    )
                item["semigroup_ratio_vs_control"] = item[
                    "semigroup_defect_mean"
                ] / max(mean(baseline, "semigroup_defect"), 1.0e-12)
                item["inverse_ratio_vs_control"] = item[
                    "inverse_defect_mean"
                ] / max(mean(baseline, "inverse_defect"), 1.0e-12)

            enough = int(criteria["minimum_seeds"])
            checks = {
                "enough_seed_pairs": len(paired) >= enough,
                "legacy_vs_control": bool(paired)
                and item["paired_legacy_score_delta"]
                <= float(criteria["max_mean_legacy_score_increase_vs_control"]),
                "legacy_vs_historical": item["legacy_score_mean"]
                - float(history["legacy_score"])
                <= float(criteria["max_mean_legacy_score_increase_vs_historical"]),
                "observed_mesh": bool(paired)
                and item["paired_observed_mesh_error_mm_delta"]
                <= float(
                    criteria[
                        "max_mean_observed_mesh_error_increase_mm_vs_control"
                    ]
                ),
                "group_rate": bool(paired)
                and item["paired_group_rate_error_delta"]
                <= float(criteria["max_mean_group_rate_error_increase_vs_control"]),
                "semigroup": bool(paired)
                and item["semigroup_ratio_vs_control"]
                <= float(criteria["max_defect_ratio_vs_control"]),
                "inverse": bool(paired)
                and item["inverse_ratio_vs_control"]
                <= float(criteria["max_defect_ratio_vs_control"]),
                "genuine_later_best": item["best_any_after_epoch_one"]
                if bool(criteria["require_all_best_any_epochs_after_epoch_one"])
                else True,
            }
            item["checks"] = checks
            item["passes"] = all(checks.values())
            decisions[representation][arm_name] = {
                "passes": item["passes"],
                "checks": checks,
                "comparison_control": control_name,
            }
            groups.append(item)

    recommended = None
    recipe_evaluations: dict[str, Any] = {}
    profiles = criteria["recipe_learning_rate_profile"]
    for objective in criteria["recipe_preference"]:
        arms = {
            representation: f"{objective}_{profiles[representation]}"
            for representation in experiment["representations"]
        }
        checks = {
            representation: bool(
                decisions[representation].get(arm_name, {}).get("passes", False)
            )
            for representation, arm_name in arms.items()
        }
        passes = all(checks.values())
        recipe_evaluations[objective] = {
            "passes": passes,
            "arms": arms,
            "representation_checks": checks,
        }
        if recommended is None and passes:
            recommended = {
                "objective": objective,
                "top_level_loss_count": LOSS_COUNTS[objective],
                "active_leaf_count": ACTIVE_LEAF_COUNTS[objective],
                "arms": arms,
            }

    expected = {
        (job["representation"], job["arm"], int(job["training"]["seed"]))
        for job in all_jobs(experiment)
    }
    observed = {
        (row["representation"], row["arm"], int(row["seed"])) for row in records
    }
    task9_expected = {
        (representation, int(seed))
        for representation in experiment["representations"]
        for seed in experiment["seeds"]
    }
    task9_observed = {
        (row["representation"], int(row["seed"])) for row in task9_controls
    }
    nested_contrasts = [
        paired_contrast(
            records,
            representation,
            "compact5_standard",
            "compact6_standard",
        )
        for representation in experiment["representations"]
    ]
    nested_contrasts.append(
        paired_contrast(
            records,
            "pca128",
            "compact5_pca_low",
            "compact6_pca_low",
        )
    )
    decision = {
        "recommended_recipe": recommended,
        "rule": (
            "smallest compact objective passing all preregistered checks, using "
            "pca_low for PCA and standard learning rate for the other representations"
        ),
        "recipe_evaluations": recipe_evaluations,
        "nested_compact5_minus_compact6": nested_contrasts,
        "per_representation": decisions,
        "complete": expected <= observed and task9_expected <= task9_observed,
        "expected_task10_jobs": len(expected),
        "observed_task10_jobs": len(expected & observed),
        "expected_task9_controls": len(task9_expected),
        "observed_task9_controls": len(task9_expected & task9_observed),
    }
    return groups, decision


def main() -> int:
    args = parse_args()
    experiment = (
        read_experiment(args.experiment) if args.experiment else read_experiment()
    )
    task9_config_path = C.resolve_path(
        experiment["external_controls"]["task9_experiment_config"]
    )
    task9_experiment = json.loads(task9_config_path.read_text())
    validate_task9_control_contract(experiment, task9_experiment)
    registry = C.load_registry()
    root = C.output_root(registry)
    summaries = sorted((root / "runs").glob("*/*/seed_*/summary.json"))
    if not summaries:
        raise SystemExit(f"No completed task10 runs found under {root / 'runs'}")
    records = [run_record(path) for path in summaries]

    task9_root = Path(experiment["external_controls"]["task9_full13_root"])
    task9_paths = [
        task9_root / representation / "full13" / f"seed_{seed}" / "summary.json"
        for representation in experiment["representations"]
        for seed in experiment["seeds"]
    ]
    missing = [path for path in task9_paths if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing read-only task9 controls:\n" + "\n".join(map(str, missing))
        )
    task9_controls = [task9_control_record(path) for path in task9_paths]
    historical = [
        historical_record(
            representation,
            Path(experiment["external_controls"]["historical_roots"][representation]),
        )
        for representation in experiment["representations"]
    ]
    groups, decision = aggregate(
        experiment, records, task9_controls, historical
    )

    output = args.output_dir or (root / "comparison")
    output.mkdir(parents=True, exist_ok=True)
    all_records = historical + task9_controls + records
    fields = sorted({key for row in all_records for key in row})
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)
    C.atomic_json(output / "aggregate.json", {"groups": groups})
    C.atomic_json(output / "decision.json", decision)

    print(
        f"{'representation':14s} {'arm':22s} {'n':>2s} {'score':>10s} "
        f"{'obs-mm':>10s} {'rate-err':>10s} {'best-ep':>8s} {'pass':>6s}"
    )
    for item in groups:
        print(
            f"{item['representation']:14s} {item['arm']:22s} {item['seeds']:2d} "
            f"{item['legacy_score_mean']:10.6f} "
            f"{item['observed_mesh_error_mm_mean']:10.6f} "
            f"{item['group_rate_error_mean']:10.6f} "
            f"{item['best_epoch_mean']:8.2f} {str(item['passes']):>6s}"
        )
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
