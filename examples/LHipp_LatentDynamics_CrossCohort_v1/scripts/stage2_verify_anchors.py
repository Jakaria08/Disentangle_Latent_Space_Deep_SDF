#!/usr/bin/env python3
"""Stage 2 gate G2.7a: the new evaluator reproduces every stored anchor test summary.

Each existing seed-42 checkpoint (August PCA/SpiralNet/Adaptive x direct_c4/plain_ode/brainode,
plus the task3_v3 LAMM cocycle) is evaluated on the stage-1 ADNI view's test split. Those views
are bit-identical to the archives the anchors were trained and evaluated on, and every numeric
pair metric is compared with the stored summary.json.

Pass: max |difference| <= 1e-5 per anchor. BrainODE anchors were evaluated with August's
per-subject loop; the vectorized field is mathematically identical but float32 rounding differs,
hence a tolerance rather than exact equality. The LAMM cocycle used an older recipe, so it
checks only the evaluator, not the stage-3 recipe.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Any

import benchmark_common as bc
import dynamics_core as D

AUGUST = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training")
LAMM_V3 = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/training")
TOLERANCE = 1.0e-5


def anchors() -> dict[str, Path]:
    runs = {}
    for rep in ("pca128", "spiralnet128", "adaptive128"):
        runs[f"{rep}/direct_c4"] = AUGUST / rep / "direct_c4" / f"{rep}_direct_c4_s42_v2"
        for method in ("plain_ode", "brainode"):
            runs[f"{rep}/{method}"] = AUGUST / rep / method / f"{rep}_{method}_s42"
    runs["lamm128/direct_c4"] = LAMM_V3 / "lamm128" / "direct_c4" / "lamm128_direct_c4_s42"
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--retrain", action="store_true",
                        help="G2.7b instead: compare the from-scratch PCA retrains (anchor_retrain suite) with the anchors.")
    return parser.parse_args()


RETRAIN_MAE_TOLERANCE = 0.002
RETRAIN_CAPTURE_TOLERANCE = 0.05


def retrain_comparison() -> int:
    """G2.7b: a seed-42 retrain with the stage-3 recipe lands near the anchor (GPU training is not bit-deterministic)."""
    results = {}
    for method in ("direct_c4", "plain_ode"):
        run = D.VALIDATION_ROOT / "anchor_retrain" / "p0_internal_adni" / "pca128" / method / f"pca128_{method}_s42_retrain"
        new = bc.read_json(run / "evaluation__p0_internal_adni" / "test" / "summary.json")["pair_metrics"]["first_last_forward"]["groups"]
        ref = bc.read_json(anchors()[f"pca128/{method}"] / "evaluation" / "test" / "summary.json")["pair_metrics"]["first_last_forward"]["groups"]

        def capture(groups):
            return groups["AD"]["predicted_signed_rate_mean"] / groups["AD"]["observed_signed_rate_mean"]

        mae_new, mae_ref = new["overall"]["end_to_end_coordinate_mae_mean"], ref["overall"]["end_to_end_coordinate_mae_mean"]
        result = {"end_to_end_mae": (mae_new, mae_ref), "ad_capture": (capture(new), capture(ref)),
                  "best_epoch": bc.read_json(run / "training_status.json")["best_epoch"]}
        result["passed"] = abs(mae_new - mae_ref) <= RETRAIN_MAE_TOLERANCE and abs(capture(new) - capture(ref)) <= RETRAIN_CAPTURE_TOLERANCE
        results[method] = result
        print(f"{'PASS' if result['passed'] else 'FAIL'} pca128/{method}: MAE {mae_new:.4f} vs {mae_ref:.4f}; "
              f"AD capture {capture(new):.2f} vs {capture(ref):.2f}; best epoch {result['best_epoch']}", flush=True)
    passed = all(r["passed"] for r in results.values())
    bc.atomic_json(bc.require_bulk(D.VALIDATION_ROOT / "reports") / "anchor_retrain_comparison.json", {
        "passed": passed, "mae_tolerance": RETRAIN_MAE_TOLERANCE, "capture_tolerance": RETRAIN_CAPTURE_TOLERANCE, "methods": results})
    return 0 if passed else 1


def numeric_leaves(tree: Any, prefix: str = "") -> dict[str, float]:
    out = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(numeric_leaves(value, f"{prefix}/{key}" if prefix else key))
    elif isinstance(tree, (int, float)) and not isinstance(tree, bool):
        out[prefix] = float(tree)
    return out


def compare(mine: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    ref = numeric_leaves({k: reference[k] for k in ("pair_metrics", "consistency_defects", "representation_floor")})
    new = numeric_leaves({k: mine[k] for k in ("pair_metrics", "consistency_defects", "representation_floor")})
    missing = sorted(set(ref) - set(new))
    worst_key, worst = "", 0.0
    for key, value in ref.items():
        if key in new and not (math.isnan(value) and math.isnan(new[key])):
            difference = abs(new[key] - value)
            if difference > worst:
                worst_key, worst = key, difference
    return {"leaves_compared": len(ref) - len(missing), "missing": missing[:10], "max_abs_difference": worst, "worst_leaf": worst_key,
            "passed": not missing and worst <= TOLERANCE}


def main() -> int:
    args = parse_args()
    if args.retrain:
        return retrain_comparison()
    root = bc.require_bulk(D.VALIDATION_ROOT / "anchor_evaluations")
    results = {}
    for name, run in anchors().items():
        if args.only and name not in args.only:
            continue
        reference = bc.read_json(run / "evaluation" / "test" / "summary.json")
        checkpoint = Path(reference["checkpoint"])
        output = root / name.replace("/", "__") / "test"
        if not (output / "summary.json").is_file():
            command = [bc.GPU_PYTHON, str(bc.SCRIPT_DIR / "evaluate_dynamics.py"), "--checkpoint", str(checkpoint),
                       "--view", "p0_internal_adni", "--trained-view", "p0_internal_adni", "--split", "test",
                       "--device", args.device, "--output-dir", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                results[name] = {"passed": False, "error": completed.stderr[-2000:]}
                print(f"FAIL {name}: evaluator crashed", flush=True)
                continue
        mine = bc.read_json(output / "summary.json")
        result = compare(mine, reference)
        primary = next(r for r in mine["tasks"] if r["task"] == "one_shot_first" and r["variant"] == "averaged" and r["group"] == "overall")
        ad = next(r for r in mine["tasks"] if r["task"] == "one_shot_first" and r["variant"] == "averaged" and r["group"] == "label=AD")
        result.update({"checkpoint": str(checkpoint), "one_shot_first_euclidean_mm": primary["euclidean_mm_mean"],
                       "one_shot_first_mae_mm": primary["coordinate_mae_mm_mean"], "ad_rate_capture": ad["rate_capture_ratio"]})
        results[name] = result
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}: max|diff|={result['max_abs_difference']:.2e} over "
              f"{result['leaves_compared']} leaves; 1-shot Euclidean {primary['euclidean_mm_mean']:.4f} AD capture {ad['rate_capture_ratio']:.2f}", flush=True)
    passed = all(r["passed"] for r in results.values()) and bool(results)
    report_dir = bc.require_bulk(D.VALIDATION_ROOT / "reports")
    bc.atomic_json(report_dir / "anchor_verification.json", {"tolerance": TOLERANCE, "passed": passed, "anchors": results})
    lines = ["# G2.7a anchor verification (ADNI test)", "", f"Tolerance: max |difference| <= {TOLERANCE:g} on every stored pair metric.", "",
             "| anchor | result | max diff | 1-shot Euclidean mm | 1-shot MAE mm | AD capture |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"| {name} | FAIL (crash) | - | - | - | - |")
        else:
            lines.append(f"| {name} | {'PASS' if r['passed'] else 'FAIL'} | {r['max_abs_difference']:.1e} | {r['one_shot_first_euclidean_mm']:.4f} | "
                         f"{r['one_shot_first_mae_mm']:.4f} | {r['ad_rate_capture']:.2f} |")
    bc.atomic_write_text(report_dir / "anchor_verification.md", "\n".join(lines) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
