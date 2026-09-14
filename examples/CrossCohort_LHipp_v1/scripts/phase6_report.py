#!/usr/bin/env python3
"""Phase 6: collect every protocol result into one comparison table.

The headline quantity is not the raw error but the **generalization gap**: how much worse a
cohort is under a model it did not fit than under its own.  A small gap means the reference
shape space covers that cohort; a large one means it does not, regardless of how good the
absolute numbers look.

Errors are reported in both conventions, because mixing them silently is the easiest way to
appear three-quarters better than a paper you are comparing against: coordinate RMSE, and
per-vertex Euclidean distance (sqrt(3) larger, the convention BrainODE uses).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import xcohort_common as xc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", type=Path, default=xc.TASK_ROOT / "reports")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=None, help="Default: the configured primary latent size.")
    return parser.parse_args()


def load_rows(reports_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(reports_dir.glob("*.csv")):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = path.name
                rows.append(row)
    return rows


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    hparams = xc.load_hyperparameters()
    k = args.k if args.k is not None else int(hparams["primary_k"])
    rows = [r for r in load_rows(args.reports_dir) if r.get("split") == args.split]
    rows = [r for r in rows if str(r.get("latent_k")) in ("", str(k))]
    if not rows:
        raise SystemExit(f"No results for split={args.split} k={k} in {args.reports_dir}")

    # (model, eval_cohort) -> protocol -> rmse
    table: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = as_float(row.get("vertex_rmse_mm_mean", ""))
        if value is None:
            continue
        protocol = row["protocol"].replace("phase2_", "")
        table[(row["model"], row["eval_cohort"])][protocol] = value

    lines = [
        f"# Cross-cohort left hippocampus - {args.split} split, latent {k}",
        "",
        "`gap` = external - internal. Positive means the reference model is worse than the "
        "cohort's own; that difference is the generalization cost.",
        "",
        "| model | cohort | internal | external | gap | pooled | loco | external (Euclidean) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value: float | None, digits: int = 6) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    for (model, cohort), protocols in sorted(table.items()):
        internal = protocols.get("internal")
        external = protocols.get("external")
        gap = (external - internal) if (internal is not None and external is not None) else None
        euclid = external * xc.EUCLIDEAN_FACTOR if external is not None else None
        lines.append(
            f"| {model} | {cohort} | {fmt(internal)} | {fmt(external)} | {fmt(gap)} | "
            f"{fmt(protocols.get('pooled'))} | {fmt(protocols.get('loco'))} | {fmt(euclid)} |"
        )

    reference_vals = hparams["reference_val_vertex_rmse_mm"]
    lines += [
        "",
        "## ADNI reference (validation, published)",
        "",
        "| model | val vertex_rmse_mm |",
        "|---|---|",
        *[f"| {name} | {value:.6f} |" for name, value in reference_vals.items()],
        "",
        "Note: on ADNI, PCA-128 beats every learned model at matched latent size. Any claim "
        "that a learned model wins on a new cohort should be checked against its own PCA "
        "baseline on that cohort, not against ADNI's.",
    ]

    out = args.reports_dir / f"phase6_summary_{args.split}_k{k}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[report] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
