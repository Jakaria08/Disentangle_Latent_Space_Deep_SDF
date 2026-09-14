#!/usr/bin/env python3
"""Compare every sweep run against the published direct_c4 baseline for its representation.

Both sides are measured on the validation split with the same functions, the same score and
the same gates, so the columns are directly comparable. Lower is better for every ratio and
for the score; the volume-trend columns are compared against the observed rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import BASELINE_RUNS, core  # noqa: E402

C, _ = core()
TASK_ROOT = Path(__file__).resolve().parents[1]


def baseline_record(representation: str) -> dict | None:
    run = Path(BASELINE_RUNS[representation])
    history = run / "history.jsonl"
    if not history.is_file():
        return None
    rows = [json.loads(line) for line in history.open()]
    status = json.loads((run / "training_status.json").read_text())
    best_epoch = int(status["best_epoch"])
    chosen = next((r for r in rows if r["epoch"] == best_epoch), rows[0])
    validation = chosen["validation"]
    groups = validation["first_last"]["groups"]
    overall = validation["all_pairs"]["groups"]["overall"]
    return {
        "id": "BASELINE_direct_c4",
        "representation": representation,
        "arch": "postact",
        "loss_set": "full13",
        "parameters": 364288,
        "best_step": best_epoch * {"pca128": 43}.get(representation, 171),
        "best_unit": f"epoch {best_epoch}",
        "score": validation["score"],
        "feasible": validation["feasible"],
        **flatten(validation["ratios"], validation["defects"], groups, overall),
    }


def flatten(ratios: dict, defects: dict, groups: dict, overall: dict) -> dict:
    out = {
        "macro_first_last_shape": ratios["macro_first_last_shape"],
        "all_pair_shape": ratios["all_pair_shape"],
        "macro_first_last_volume": ratios.get("macro_first_last_volume"),
        "semigroup_defect": defects["relative_semigroup_defect_mean"],
        "inverse_defect": defects["relative_inverse_defect_mean"],
        "cn_pred_rate": ratios["first_last_cn_predicted_signed_rate"],
        "cn_obs_rate": ratios["first_last_cn_observed_signed_rate"],
        "ad_pred_rate": ratios["first_last_ad_predicted_signed_rate"],
        "ad_obs_rate": ratios["first_last_ad_observed_signed_rate"],
    }
    out["cn_rate_abs_error"] = abs(out["cn_pred_rate"] - out["cn_obs_rate"])
    out["ad_rate_abs_error"] = abs(out["ad_pred_rate"] - out["ad_obs_rate"])
    out["rate_gap_pred"] = out["ad_pred_rate"] - out["cn_pred_rate"]
    out["rate_gap_obs"] = out["ad_obs_rate"] - out["cn_obs_rate"]
    out["rate_gap_abs_error"] = abs(out["rate_gap_pred"] - out["rate_gap_obs"])
    for diagnosis in ("CN", "AD"):
        if diagnosis in groups:
            g = groups[diagnosis]
            out[f"{diagnosis.lower()}_shape_ratio"] = g["coordinate_mean"] / max(g["nochange_coordinate_mean"], 1e-12)
            out[f"{diagnosis.lower()}_vol_ratio"] = g["volume_relative_mean"] / max(g["nochange_volume_relative_mean"], 1e-12)
    out["all_pair_recon_euclidean_mm"] = overall["euclidean_mean"]
    out["all_pair_nochange_euclidean_mm"] = overall["nochange_euclidean_mean"]
    return out


def sweep_record(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    validation = summary["best_validation"]
    groups = validation["first_last_groups"]
    overall = validation["all_pairs_overall"]
    return {
        "id": summary["id"],
        "representation": summary["representation"],
        "arch": summary["arch"],
        "loss_set": summary["loss_set"],
        "parameters": summary["parameters"],
        "best_step": summary["best_step"],
        "best_unit": f"step {summary['best_step']}/{summary['total_steps']}",
        "score": validation["score"],
        "feasible": validation["feasible"],
        "minutes": round(summary["elapsed_minutes"], 2),
        **flatten(validation["ratios"], validation["defects"], groups, overall),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    runs = C.output_root() / "runs"
    records = [baseline_record(rep) for rep in BASELINE_RUNS]
    records = [r for r in records if r is not None]
    records += [sweep_record(p) for p in sorted(runs.glob("*/*/summary.json"))]

    out_dir = args.output_dir or (C.output_root() / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in records for k in r})
    order = ["representation", "id", "arch", "loss_set", "score", "feasible", "best_unit"]
    fields = order + [f for f in fields if f not in order]
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (out_dir / "comparison.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    print(f"{'rep':13s} {'id':24s} {'score':>8s} {'feas':>5s} {'macroShape':>11s} "
          f"{'semig':>7s} {'inv':>7s} {'cnRateErr':>10s} {'adRateErr':>10s} {'best':>16s}")
    for rep in BASELINE_RUNS:
        rows = sorted([r for r in records if r["representation"] == rep], key=lambda r: r["score"])
        base = next((r for r in rows if r["id"] == "BASELINE_direct_c4"), None)
        for r in rows:
            mark = " *" if base and r["id"] != "BASELINE_direct_c4" and r["score"] < base["score"] and r["feasible"] else "  "
            print(f"{r['representation']:13s} {r['id']:24s} {r['score']:8.5f} {str(r['feasible'])[:5]:>5s} "
                  f"{r['macro_first_last_shape']:11.5f} {r['semigroup_defect']:7.4f} {r['inverse_defect']:7.4f} "
                  f"{r['cn_rate_abs_error']:10.5f} {r['ad_rate_abs_error']:10.5f} {r['best_unit']:>16s}{mark}")
        print()
    print(f"wrote {out_dir/'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
