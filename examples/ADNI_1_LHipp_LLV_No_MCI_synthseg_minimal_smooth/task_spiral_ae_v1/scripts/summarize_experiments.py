#!/usr/bin/env python3
"""Final comparison: the four spiral-AE experiments against PCA at matched latent dimension.

Writes reports/comparison.csv and reports/comparison.md. The Markdown table is the artifact
that answers "did we beat PCA", and it states the matched-latent verdict either way.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spiral_common as sc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(sc.OUTPUT_ROOT / "reports"))
    return p.parse_args()


def load_pca_rows():
    fp = sc.OUTPUT_ROOT / "pca" / "metrics" / "pca_reconstruction_summary.csv"
    if not fp.exists():
        raise SystemExit(f"Missing {fp}; run fit_pca_baseline.py first.")
    with open(fp, newline="") as handle:
        return list(csv.DictReader(handle))


def pca_lookup(pca_rows, components, split, field="vertex_rmse_mm_mean"):
    for row in pca_rows:
        if int(row["components"]) == int(components) and row["split"] == split:
            return float(row[field])
    return None


def main():
    args = parse_args()
    out_dir = sc.makedirs(sc.require_bulk_path(args.output_dir, "report dir"))
    pca_rows = load_pca_rows()

    rows = []
    for latent in (128, 256):
        for split in sc.SPLITS:
            rows.append(
                {
                    "method": f"PCA-{latent}",
                    "conv_type": "pca",
                    "latent": latent,
                    "split": split,
                    "vertex_rmse_mm_mean": pca_lookup(pca_rows, latent, split),
                    "vertex_rmse_mm_median": pca_lookup(
                        pca_rows, latent, split, "vertex_rmse_mm_median"
                    ),
                    "vertex_rmse_mm_p95": pca_lookup(pca_rows, latent, split, "vertex_rmse_mm_p95"),
                    "volume_abs_relative_error_pct_mean": pca_lookup(
                        pca_rows, latent, split, "volume_abs_relative_error_pct_mean"
                    ),
                    "n_params": latent * 8238,
                    "best_trial": "",
                    "n_trials_complete": "",
                }
            )

    summaries = {}
    for exp_name, spec in sorted(sc.EXPERIMENTS.items()):
        fp = sc.OUTPUT_ROOT / "best" / exp_name / "best_summary.json"
        if not fp.exists():
            print(f"[skip] {exp_name}: no best_summary.json yet")
            continue
        summary = json.loads(fp.read_text())
        summaries[exp_name] = summary
        for split in sc.SPLITS:
            metrics = summary["metrics"][split]
            rows.append(
                {
                    "method": exp_name,
                    "conv_type": spec["conv_type"],
                    "latent": spec["latent"],
                    "split": split,
                    "vertex_rmse_mm_mean": metrics["vertex_rmse_mm_mean"],
                    "vertex_rmse_mm_median": metrics["vertex_rmse_mm_median"],
                    "vertex_rmse_mm_p95": metrics["vertex_rmse_mm_p95"],
                    "volume_abs_relative_error_pct_mean": metrics.get(
                        "volume_abs_relative_error_pct_mean"
                    ),
                    "n_params": summary["n_params"],
                    "best_trial": summary["best_trial"],
                    "n_trials_complete": summary["n_trials_complete"],
                }
            )

    sc.atomic_write_csv(out_dir / "comparison.csv", rows)

    lines = [
        "# Hippocampus spiral AE vs PCA - reconstruction",
        "",
        "ADNI left hippocampus (`hippocampus_pca_cocycle_v4` manifest): 2037 train / 269 val /",
        "277 test scans, 2746 vertices, coordinates in mm. Metric is `vertex_rmse_mm`, the same",
        "per-coordinate RMSE used by the published PCA table (this pipeline reproduces PCA-128",
        f"val = {sc.PCA128_VAL_REFERENCE} as a gate before any model is trained).",
        "",
        "## Test-split reconstruction at matched latent dimension",
        "",
        "| method | latent | params | test RMSE (mm) | val RMSE (mm) | vs PCA (test) |",
        "|---|---|---|---|---|---|",
    ]

    for latent in (128, 256):
        pca_test = pca_lookup(pca_rows, latent, "test")
        pca_val = pca_lookup(pca_rows, latent, "val")
        lines.append(
            f"| **PCA-{latent}** | {latent} | {latent * 8238:,} | {pca_test:.6f} | "
            f"{pca_val:.6f} | - |"
        )
        for exp_name, spec in sorted(sc.EXPERIMENTS.items()):
            if spec["latent"] != latent or exp_name not in summaries:
                continue
            summary = summaries[exp_name]
            test = summary["metrics"]["test"]["vertex_rmse_mm_mean"]
            val = summary["metrics"]["val"]["vertex_rmse_mm_mean"]
            ratio = test / pca_test if pca_test else float("nan")
            verdict = "**beats PCA**" if test < pca_test else f"{ratio:.2f}x worse"
            lines.append(
                f"| {exp_name} | {latent} | {summary['n_params']:,} | {test:.6f} | "
                f"{val:.6f} | {verdict} |"
            )
        lines.append("")

    lines += ["## Selected configurations", ""]
    for exp_name, summary in sorted(summaries.items()):
        lines += [
            f"### {exp_name}",
            "",
            f"- best trial {summary['best_trial']} of {summary['n_trials_complete']} complete",
            f"- pre-latent width {summary['pre_latent_dim']}, params {summary['n_params']:,}",
            f"- conv per level: {summary['conv_types']}",
            f"- dynamic spiral lengths: {summary['dynamic_seq_lengths']}",
            f"- params: `{json.dumps(summary['best_params'], sort_keys=True)}`",
            "",
        ]

    (out_dir / "comparison.md").write_text("\n".join(lines))
    print(f"[report] wrote {out_dir / 'comparison.csv'} and {out_dir / 'comparison.md'}")
    print("\n".join(lines[8:30]))


if __name__ == "__main__":
    main()
