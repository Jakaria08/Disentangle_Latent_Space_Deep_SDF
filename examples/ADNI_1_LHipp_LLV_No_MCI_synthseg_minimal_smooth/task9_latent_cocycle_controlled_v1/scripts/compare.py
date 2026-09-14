#!/usr/bin/env python3
"""Compare matched objectives across seeds and choose the smallest acceptable loss set."""

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
from configuration import read_experiment  # noqa: E402
from objective import LOSS_COUNTS  # noqa: E402

C, _ = core()

HISTORICAL = {
    "pca128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/pca128/direct_c4/pca128_direct_c4_s42"),
    "spiralnet128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/spiralnet128/direct_c4/spiralnet128_direct_c4_s42"),
    "adaptive128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/adaptive128/direct_c4/adaptive128_direct_c4_s42"),
    "lamm128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/training/lamm128/direct_c4/lamm128_direct_c4_s42"),
}


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
        "semigroup_defect": float(validation["defects"]["relative_semigroup_defect_mean"]),
        "inverse_defect": float(validation["defects"]["relative_inverse_defect_mean"]),
    }


def historical_record(representation: str) -> dict[str, Any]:
    root = HISTORICAL[representation]
    status = json.loads((root / "training_status.json").read_text())
    history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
    best_epoch = int(status["best_epoch"])
    row = next(item for item in history if int(item["epoch"]) == best_epoch)
    return {
        "representation": representation,
        "loss_set": "historical_full13",
        "loss_count": 13,
        "seed": 42,
        "best_epoch": best_epoch,
        "best_step": sum(int(item["batches"]) for item in history if int(item["epoch"]) <= best_epoch),
        "best_any_epoch": best_epoch,
        "early_checkpoint_better": False,
        **metrics(row["validation"]),
    }


def run_record(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    return {
        "representation": summary["representation"],
        "loss_set": summary["loss_set"],
        "loss_count": int(summary["loss_count"]),
        "seed": int(summary["seed"]),
        "best_epoch": int(summary["best_mature"]["epoch"]),
        "best_step": int(summary["best_mature"]["step"]),
        "best_any_epoch": int(summary["best_any"]["epoch"]),
        "early_checkpoint_better": bool(summary["early_checkpoint_better"]),
        **metrics(summary["best_mature_validation"]),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def aggregate(
    experiment: dict[str, Any], records: list[dict[str, Any]], historical: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    criteria = experiment["adoption"]
    groups: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    for representation in experiment["representations"]:
        rep_rows = [row for row in records if row["representation"] == representation]
        full = [row for row in rep_rows if row["loss_set"] == "full13"]
        history = next(row for row in historical if row["representation"] == representation)
        decisions[representation] = {}
        for loss_set in experiment["loss_sets"]:
            rows = [row for row in rep_rows if row["loss_set"] == loss_set]
            if not rows:
                continue
            item: dict[str, Any] = {
                "representation": representation,
                "loss_set": loss_set,
                "loss_count": LOSS_COUNTS[loss_set],
                "seeds": len(rows),
                "best_epoch_mean": mean(rows, "best_epoch"),
                "best_epoch_min": min(row["best_epoch"] for row in rows),
                "best_any_after_epoch_one": all(row["best_any_epoch"] > 1 for row in rows),
                "early_checkpoint_better": any(row["early_checkpoint_better"] for row in rows),
            }
            for key in (
                "legacy_score",
                "macro_shape_ratio",
                "decoded_target_error_mm",
                "observed_mesh_error_mm",
                "group_rate_error",
                "semigroup_defect",
                "inverse_defect",
            ):
                item[f"{key}_mean"] = mean(rows, key)
                item[f"{key}_std"] = statistics.stdev(float(row[key]) for row in rows) if len(rows) > 1 else 0.0
            if full and loss_set != "full13":
                full_by_seed = {row["seed"]: row for row in full}
                paired = [(row, full_by_seed[row["seed"]]) for row in rows if row["seed"] in full_by_seed]
                for key in (
                    "legacy_score",
                    "observed_mesh_error_mm",
                    "group_rate_error",
                    "semigroup_defect",
                    "inverse_defect",
                ):
                    item[f"paired_{key}_delta"] = statistics.fmean(
                        float(row[key]) - float(control[key]) for row, control in paired
                    )
                item["semigroup_ratio_vs_full13"] = item["semigroup_defect_mean"] / max(mean(full, "semigroup_defect"), 1.0e-12)
                item["inverse_ratio_vs_full13"] = item["inverse_defect_mean"] / max(mean(full, "inverse_defect"), 1.0e-12)
                checks = {
                    "enough_seeds": len(rows) >= int(criteria["minimum_seeds"]),
                    "legacy_vs_full13": item["paired_legacy_score_delta"] <= float(criteria["max_mean_legacy_score_increase_vs_full13"]),
                    "legacy_vs_historical": item["legacy_score_mean"] - float(history["legacy_score"]) <= float(criteria["max_mean_legacy_score_increase_vs_historical"]),
                    "observed_mesh": item["paired_observed_mesh_error_mm_delta"] <= float(criteria["max_mean_observed_mesh_error_increase_mm_vs_full13"]),
                    "group_rate": item["paired_group_rate_error_delta"] <= float(criteria["max_mean_group_rate_error_increase_vs_full13"]),
                    "semigroup": item["semigroup_ratio_vs_full13"] <= float(criteria["max_defect_ratio_vs_full13"]),
                    "inverse": item["inverse_ratio_vs_full13"] <= float(criteria["max_defect_ratio_vs_full13"]),
                    "genuine_later_best": item["best_any_after_epoch_one"] if bool(criteria["require_all_best_any_epochs_after_epoch_one"]) else True,
                }
                item["checks"] = checks
                item["passes"] = all(checks.values())
                decisions[representation][loss_set] = {
                    "passes": item["passes"], "checks": checks
                }
            groups.append(item)

    candidates = []
    for loss_set in ("lean4", "lean5", "lean6"):
        if all(decisions[rep].get(loss_set, {}).get("passes", False) for rep in experiment["representations"]):
            candidates.append(loss_set)
    decision = {
        "recommended_loss_set": candidates[0] if candidates else None,
        "rule": "smallest reduced objective satisfying every preregistered check in all representations",
        "per_representation": decisions,
        "complete": all(
            len([row for row in records if row["representation"] == rep and row["loss_set"] == loss_set])
            >= int(criteria["minimum_seeds"])
            for rep in experiment["representations"]
            for loss_set in experiment["loss_sets"]
        ),
    }
    return groups, decision


def main() -> int:
    args = parse_args()
    experiment = read_experiment(args.experiment) if args.experiment else read_experiment()
    registry = C.load_registry()
    root = C.output_root(registry)
    summaries = sorted((root / "runs").glob("*/*/seed_*/summary.json"))
    if not summaries:
        raise SystemExit(f"No completed runs found under {root / 'runs'}")
    records = [run_record(path) for path in summaries]
    historical = [historical_record(rep) for rep in experiment["representations"]]
    groups, decision = aggregate(experiment, records, historical)
    output = args.output_dir or (root / "comparison")
    output.mkdir(parents=True, exist_ok=True)
    all_records = historical + records
    fields = sorted({key for row in all_records for key in row})
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)
    C.atomic_json(output / "aggregate.json", {"groups": groups})
    C.atomic_json(output / "decision.json", decision)

    print(f"{'representation':14s} {'losses':8s} {'n':>2s} {'score':>10s} {'obs-mm':>10s} {'rate-err':>10s} {'best-ep':>8s} {'pass':>6s}")
    for item in groups:
        print(
            f"{item['representation']:14s} {item['loss_set']:8s} {item['seeds']:2d} "
            f"{item['legacy_score_mean']:10.6f} {item['observed_mesh_error_mm_mean']:10.6f} "
            f"{item['group_rate_error_mean']:10.6f} {item['best_epoch_mean']:8.2f} "
            f"{str(item.get('passes', 'control')):>6s}"
        )
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

