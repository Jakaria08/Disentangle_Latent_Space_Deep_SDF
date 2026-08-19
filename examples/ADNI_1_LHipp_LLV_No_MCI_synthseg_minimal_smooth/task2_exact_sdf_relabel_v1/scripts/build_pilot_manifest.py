#!/usr/bin/env python3
"""Select a deterministic, subject-complete exact-SDF hippocampus pilot cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_MANIFEST,
    atomic_write_csv,
    atomic_write_json,
    read_manifest,
    require_bulk_path,
    split_subject_leakage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--cohort-name",
        default="hippocampus_pilot",
        help="Safe filename prefix; use a different prefix when reusing this for another structure.",
    )
    parser.add_argument("--train-scans", type=int, default=400)
    parser.add_argument("--val-scans", type=int, default=100)
    parser.add_argument("--test-scans", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-non-bulk-output",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def stable_number(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def subject_groups(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(row)
    result = []
    for subject, visits in grouped.items():
        diagnosis_counts = Counter(row.get("diagnosis", "unknown") for row in visits)
        diagnosis = sorted(diagnosis_counts, key=lambda item: (-diagnosis_counts[item], item))[0]
        volumes = [
            float(row["correspondence_volume_mm3"])
            for row in visits
            if row.get("correspondence_volume_mm3") not in {None, ""}
        ]
        result.append(
            {
                "subject_id": subject,
                "diagnosis": diagnosis,
                "mean_volume": float(np.mean(volumes)) if volumes else float("nan"),
                "rows": sorted(
                    visits,
                    key=lambda row: (float(row.get("visit_order", 0) or 0), row["scan_id"]),
                ),
            }
        )
    return result


def assign_volume_bins(groups: list[dict[str, Any]], bins: int = 3) -> None:
    by_diagnosis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_diagnosis[group["diagnosis"]].append(group)
    for members in by_diagnosis.values():
        finite = np.asarray(
            [member["mean_volume"] for member in members if np.isfinite(member["mean_volume"])],
            dtype=np.float64,
        )
        edges = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1]) if len(finite) else []
        for member in members:
            member["volume_bin"] = (
                int(np.searchsorted(edges, member["mean_volume"], side="right"))
                if np.isfinite(member["mean_volume"])
                else -1
            )


def balanced_subject_order(groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    assign_volume_bins(groups)
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        strata[(group["diagnosis"], group["volume_bin"])].append(group)
    for key, members in strata.items():
        members.sort(key=lambda group: stable_number(f"{key}:{group['subject_id']}", seed))
    ordered: list[dict[str, Any]] = []
    keys = sorted(strata)
    offset = 0
    while True:
        changed = False
        for key in keys:
            if offset < len(strata[key]):
                ordered.append(strata[key][offset])
                changed = True
        if not changed:
            break
        offset += 1
    return ordered


def select_subject_complete(
    rows: list[dict[str, str]], target_scans: int, seed: int
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if target_scans < 1:
        raise ValueError("Every requested split size must be positive.")
    groups = subject_groups(rows)
    if target_scans >= len(rows):
        selected_groups = groups
    else:
        ordered = balanced_subject_order(groups, seed)
        selected_groups = []
        count = 0
        remaining = list(ordered)
        while remaining and count < target_scans:
            # Preserve round-robin stratification while preferring the visit count that
            # lands nearest the target among the next small balanced window.
            window = remaining[: min(12, len(remaining))]
            best = min(
                window,
                key=lambda group: (
                    abs(target_scans - (count + len(group["rows"]))),
                    stable_number(group["subject_id"], seed + count),
                ),
            )
            remaining.remove(best)
            selected_groups.append(best)
            count += len(best["rows"])
        # Adding a whole subject can overshoot. Remove the last group only when doing
        # so is strictly closer to the requested scan count.
        if selected_groups:
            without_last = count - len(selected_groups[-1]["rows"])
            if abs(target_scans - without_last) < abs(target_scans - count):
                selected_groups.pop()
                count = without_last
    selected = [row for group in selected_groups for row in group["rows"]]
    selected.sort(
        key=lambda row: (
            row["subject_id"],
            float(row.get("visit_order", 0) or 0),
            row["scan_id"],
        )
    )
    report = {
        "target_scan_count": int(target_scans),
        "actual_scan_count": len(selected),
        "subject_count": len(selected_groups),
        "all_visits_retained_for_selected_subjects": True,
        "diagnosis_scan_counts": dict(Counter(row.get("diagnosis", "") for row in selected)),
        "diagnosis_subject_counts": dict(
            Counter(group["diagnosis"] for group in selected_groups)
        ),
        "scan_ids": [row["scan_id"] for row in selected],
        "subject_ids": [group["subject_id"] for group in selected_groups],
    }
    return selected, report


def main() -> None:
    args = parse_args()
    if (
        not args.cohort_name
        or args.cohort_name in {".", ".."}
        or "/" in args.cohort_name
        or "\\" in args.cohort_name
    ):
        raise ValueError(f"Unsafe cohort-name: {args.cohort_name!r}")
    root = require_bulk_path(
        args.output_root,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    approximate_manifest = root / "manifests" / f"{args.cohort_name}_approx.csv"
    selection_path = root / "selections" / f"{args.cohort_name}_selection.json"
    for output in (approximate_manifest, selection_path):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite explicitly.")

    rows = read_manifest(args.source_manifest)
    leakage = split_subject_leakage(rows)
    if leakage:
        raise ValueError(f"Source manifest has subject leakage: {dict(list(leakage.items())[:5])}")
    target_by_split = {
        "train": args.train_scans,
        "val": args.val_scans,
        "test": args.test_scans,
    }
    selected: list[dict[str, str]] = []
    split_reports = {}
    for offset, split in enumerate(("train", "val", "test")):
        pool = [row for row in rows if row["split"] == split]
        chosen, split_report = select_subject_complete(
            pool, int(target_by_split[split]), int(args.seed) + offset * 1009
        )
        split_reports[split] = split_report
        selected.extend(chosen)

    selected_ids = {row["scan_id"] for row in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("Pilot selection contains duplicate scans.")
    output_rows = []
    for row in selected:
        for field in ("mesh_path", "sdf_npz_path"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(f"Selected input does not exist ({field}): {row[field]}")
        require_bulk_path(row["sdf_npz_path"], "selected source SDF")
        item = dict(row)
        item["source_sdf_npz_path"] = row["sdf_npz_path"]
        item["sdf_label_kind"] = "approximate_preprocessmesh"
        item["pilot_selection_seed"] = str(args.seed)
        output_rows.append(item)

    atomic_write_csv(
        approximate_manifest,
        output_rows,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    report = {
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "approximate_manifest": str(approximate_manifest),
        "selection_seed": int(args.seed),
        "selection_unit": "subject_all_visits",
        "stratification": "diagnosis_x_within_diagnosis_volume_tertile_round_robin",
        "split_reports": split_reports,
        "total_scans": len(output_rows),
        "total_subjects": len({row["subject_id"] for row in output_rows}),
        "subject_split_disjoint": not bool(split_subject_leakage(output_rows)),
        "test_policy": "selected_and_relabelled_now_but_locked_until_model_selection_is_complete",
    }
    atomic_write_json(selection_path, report, allow_non_bulk=args.allow_non_bulk_output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
