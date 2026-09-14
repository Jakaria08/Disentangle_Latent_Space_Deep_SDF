#!/usr/bin/env python3
"""Stage 1c: materialize every protocol view as August-format archives plus task tables.

A view (configs/protocol_views.json) fixes a protocol's train, val and test scans. For each
view this writes, under the bulk root:

  dataset/{split}_subject_sequences.npz            sequences, ages, volumes, labels
  pairs/{split}_forward_pairs.csv                   every forward visit pair (August schema)
  representations/<rep>/{split}_subject_sequences_128.npz
                                                    raw + train-standardized R1 codes
  tasks/{split}_{task}.csv                          evaluation tasks (val and test)
  view_manifest.json                                membership, normalization, counts, hashes

The archive schema is the one August_Version and task3 trainers read, so they can train on any
view unchanged. Every statistic is fitted on the view's train rows: latent mean/std and the
age range behind ``visit_age_norm_train``. Condition labels are canonical - CN/AD names for
0/1 - with the source names (Control/ALS for CALSNIC) kept in ``*_diagnoses_source``. Ids are
cohort-qualified (``aibl:123``) because bare ADNI and AIBL RIDs collide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import benchmark_common as bc

CANONICAL = {0: "CN", 1: "AD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views", nargs="+", default=None, help="Default: every view, including cross-fit folds.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# membership
# --------------------------------------------------------------------------------------


def crossfit_table(cohort: str, frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    subjects = frame.groupby("subject_key", sort=True).agg(
        diagnosis=("diagnosis_source", "first"), visits=("scan_key", "nunique")
    )
    strata = {key: f"{row.diagnosis}|{bc.visit_count_bin(int(row.visits))}" for key, row in subjects.iterrows()}
    assignment = bc.crossfit_folds(strata, folds, seed)
    table = subjects.reset_index()
    table["stratum"] = table["subject_key"].map(strata)
    table["fold"] = table["subject_key"].map(assignment).astype(int)
    table.insert(0, "cohort", cohort)
    return table


def view_membership(spec: dict[str, Any], frames: dict[str, pd.DataFrame], folds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    if "crossfit" in spec:
        info = spec["crossfit"]
        frame = frames[info["cohort"]]
        fold_of = folds[info["cohort"]].set_index("subject_key")["fold"]
        subject_fold = frame["subject_key"].map(fold_of)
        view_split = np.where(subject_fold == info["fold"], "test", np.where(subject_fold == info["val_fold"], "val", "train"))
        parts.append(frame.assign(view_split=view_split, source_split=frame["split"]))
    else:
        for view_split in bc.SPLITS:
            for cohort, member in spec[view_split]:
                frame = frames[cohort]
                chosen = frame if member == "all" else frame.loc[frame["split"] == member]
                if chosen.empty:
                    raise ValueError(f"member {cohort}/{member} is empty")
                parts.append(chosen.assign(view_split=view_split, source_split=chosen["split"]))
    rows = pd.concat(parts, ignore_index=True)
    leaked = rows.groupby("subject_key")["view_split"].nunique()
    if leaked.gt(1).any():
        raise ValueError(f"subjects in more than one view split: {sorted(leaked[leaked > 1].index)[:5]}")
    if rows["scan_key"].duplicated().any():
        raise ValueError("a scan appears twice in one view")
    for view_split in bc.SPLITS:
        if not (rows["view_split"] == view_split).any():
            raise ValueError(f"view split {view_split} is empty")
    return rows.sort_values(["view_split", "subject_key", "visit_month", "scan_key"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# archives
# --------------------------------------------------------------------------------------


def string_array(values) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=str)


def sequence_archive(rows: pd.DataFrame, age_min: float, age_max: float) -> dict[str, np.ndarray]:
    rows = rows.sort_values(["subject_key", "visit_month", "scan_key"], kind="stable").reset_index(drop=True)
    rows["visit_order_view"] = rows.groupby("subject_key").cumcount().astype(np.int64)
    rows["months_view"] = rows["visit_month"] - rows.groupby("subject_key")["visit_month"].transform("min")
    subjects = rows.drop_duplicates("subject_key", keep="first")
    counts = rows.groupby("subject_key", sort=False)["scan_key"].size().to_numpy()
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    labels = rows["label"].to_numpy(dtype=np.int64)
    return {
        "subject_ids": string_array(subjects["subject_key"]),
        "subject_splits": string_array(subjects["view_split"]),
        "subject_diagnoses": string_array(subjects["label"].map(CANONICAL)),
        "subject_diagnoses_source": string_array(subjects["diagnosis_source"]),
        "subject_label_ad": subjects["label"].to_numpy(dtype=np.int64),
        "subject_cohorts": string_array(subjects["cohort"]),
        "subject_baseline_age_years": subjects["age_years"].to_numpy(dtype=np.float32),
        "subject_visit_offsets": offsets,
        "visit_scan_ids": string_array(rows["scan_key"]),
        "visit_subject_ids": string_array(rows["subject_key"]),
        "visit_splits": string_array(rows["view_split"]),
        "visit_source_splits": string_array(rows["source_split"]),
        "visit_cohorts": string_array(rows["cohort"]),
        "visit_diagnoses": string_array(pd.Series(labels).map(CANONICAL)),
        "visit_diagnoses_source": string_array(rows["diagnosis_source"]),
        "visit_label_ad": labels,
        "visit_orders": rows["visit_order_view"].to_numpy(dtype=np.int64),
        "visit_months_from_baseline": rows["months_view"].to_numpy(dtype=np.float32),
        "visit_time_years_from_baseline": (rows["months_view"].to_numpy(dtype=np.float32) / np.float32(12.0)),
        "visit_age_years": rows["age_years"].to_numpy(dtype=np.float32),
        "visit_age_norm_train": ((rows["age_years"].to_numpy(dtype=np.float64) - age_min) / (age_max - age_min)).astype(np.float32),
        "visit_volume_mm3": rows["correspondence_volume_mm3"].to_numpy(dtype=np.float32),
        "visit_surface_area_mm2": rows["correspondence_surface_area_mm2"].to_numpy(dtype=np.float32),
    }


def forward_pairs(archive: dict[str, np.ndarray], split: str) -> pd.DataFrame:
    """All forward pairs per subject, midpoint intermediate, exactly as the cocycle_v4 packager."""
    rows = []
    offsets = archive["subject_visit_offsets"]
    months = archive["visit_months_from_baseline"].astype(np.float64)
    for subject_index, subject in enumerate(archive["subject_ids"]):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        for source in range(start, end - 1):
            for target in range(source + 1, end):
                gap = target - source
                rows.append({
                    "split": split,
                    "diagnosis": str(archive["subject_diagnoses"][subject_index]),
                    "label_ad": int(archive["subject_label_ad"][subject_index]),
                    "subject_id": str(subject),
                    "source_index": source,
                    "target_index": target,
                    "intermediate_index": source + gap // 2 if gap > 1 else -1,
                    "source_scan_id": str(archive["visit_scan_ids"][source]),
                    "target_scan_id": str(archive["visit_scan_ids"][target]),
                    "source_visit_order": int(archive["visit_orders"][source]),
                    "target_visit_order": int(archive["visit_orders"][target]),
                    "pair_type": "adjacent" if gap == 1 else "nonadjacent",
                    "delta_years": (months[target] - months[source]) / 12.0,
                    "cohort": str(archive["subject_cohorts"][subject_index]),
                })
    frame = pd.DataFrame(rows)
    if (frame["delta_years"] <= 0).any():
        raise ValueError("non-positive pair interval")
    return frame


def task_table(archive: dict[str, np.ndarray], task: str, spec: dict[str, Any]) -> pd.DataFrame:
    rows = []
    offsets = archive["subject_visit_offsets"]
    years = archive["visit_time_years_from_baseline"].astype(np.float64)
    for subject_index, subject in enumerate(archive["subject_ids"]):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        prefix = bc.task_prefix(end - start, spec)
        if prefix is None:
            continue
        indices = [start + local for local in prefix]
        target = end - 1
        rows.append({
            "task": task,
            "subject_id": str(subject),
            "cohort": str(archive["subject_cohorts"][subject_index]),
            "diagnosis": str(archive["subject_diagnoses"][subject_index]),
            "diagnosis_source": str(archive["subject_diagnoses_source"][subject_index]),
            "label_ad": int(archive["subject_label_ad"][subject_index]),
            "n_visits": end - start,
            "k": len(indices),
            "prefix_indices": ";".join(str(index) for index in indices),
            "prefix_scan_ids": ";".join(str(archive["visit_scan_ids"][index]) for index in indices),
            "prefix_years": ";".join(f"{years[index]:.6f}" for index in indices),
            "target_index": target,
            "target_scan_id": str(archive["visit_scan_ids"][target]),
            "target_years": float(years[target]),
            "horizon_from_last_prefix_years": float(years[target] - years[indices[-1]]),
            "horizon_from_first_prefix_years": float(years[target] - years[indices[0]]),
        })
    columns = ["task", "subject_id", "cohort", "diagnosis", "diagnosis_source", "label_ad", "n_visits", "k", "prefix_indices",
               "prefix_scan_ids", "prefix_years", "target_index", "target_scan_id", "target_years",
               "horizon_from_last_prefix_years", "horizon_from_first_prefix_years"]
    return pd.DataFrame(rows, columns=columns)


def load_latents(cohort: str, name: str) -> tuple[dict[str, int], np.ndarray]:
    archive = bc.load_npz(bc.STAGE1_ROOT / "latents" / cohort / f"{name}.npz")
    keys = archive["scan_keys"].astype(str)
    return {key: index for index, key in enumerate(keys)}, archive["codes"].astype(np.float32)


def build_view(name: str, spec: dict[str, Any], frames, folds, latents, registry, tasks, overwrite: bool) -> dict[str, Any]:
    root = bc.require_bulk(bc.STAGE1_ROOT / "views" / name)
    if (root / "view_manifest.json").exists() and not overwrite:
        print(f"[{name}] view exists, skipping (pass --overwrite to rebuild)", flush=True)
        return bc.read_json(root / "view_manifest.json")
    rows = view_membership(spec, frames, folds)
    train_rows = rows.loc[rows["view_split"] == "train"]
    age_min, age_max = float(train_rows["age_years"].min()), float(train_rows["age_years"].max())
    if age_max <= age_min:
        raise ValueError(f"{name}: degenerate training age range")

    archives = {split: sequence_archive(rows.loc[rows["view_split"] == split].copy(), age_min, age_max) for split in bc.SPLITS}
    manifest: dict[str, Any] = {
        "view": name,
        "spec": spec,
        "normalization": {"fitted_on": "view train rows", "age_min_years": age_min, "age_max_years": age_max},
        "canonical_labels": {"0": "CN (cohort control class)", "1": "AD (cohort disease class; ALS for CALSNIC)"},
        "splits": {},
        "tasks": {},
        "files": {},
    }
    for split, archive in archives.items():
        path = bc.atomic_npz(root / "dataset" / f"{split}_subject_sequences.npz", archive)
        pairs = forward_pairs(archive, split)
        pair_path = bc.atomic_csv(root / "pairs" / f"{split}_forward_pairs.csv", pairs)
        manifest["files"][f"dataset/{split}"] = bc.sha256_file(path)
        manifest["files"][f"pairs/{split}"] = bc.sha256_file(pair_path)
        subjects = pd.DataFrame({"cohort": archive["subject_cohorts"], "label": archive["subject_label_ad"]})
        manifest["splits"][split] = {
            "subjects": int(len(archive["subject_ids"])),
            "scans": int(len(archive["visit_scan_ids"])),
            "pairs": int(len(pairs)),
            "subjects_by_cohort_label": {
                f"{cohort}|{CANONICAL[int(label)]}": int(count)
                for (cohort, label), count in subjects.value_counts().sort_index().items()
            },
        }
        if split in ("val", "test"):
            for task, task_spec in tasks["tasks"].items():
                table = task_table(archive, task, task_spec)
                bc.atomic_csv(root / "tasks" / f"{split}_{task}.csv", table)
                manifest["tasks"][f"{split}/{task}"] = {
                    "subjects": int(len(table)),
                    "by_cohort_label": {f"{c}|{d}": int(n) for (c, d), n in table.groupby(["cohort", "diagnosis"]).size().items()},
                }

    for rep, rep_spec in registry["representations"].items():
        manifest["files"].update(write_view_representation(root, archives, rep, rep_spec, latents))
    bc.atomic_json(root / "view_manifest.json", manifest)
    return manifest


def write_view_representation(root: Path, archives: dict[str, dict[str, np.ndarray]], rep: str, rep_spec: dict[str, Any],
                              latents) -> dict[str, str]:
    """One representation's train-standardized code archives for a view; returns their file hashes.

    Also used by stage 5 to add the sensitivity-only pooled PCA codes to an existing view.
    """
    files = {}
    raw = {}
    for split, archive in archives.items():
        codes = []
        for scan_key, cohort in zip(archive["visit_scan_ids"], archive["visit_cohorts"]):
            index_of, matrix = latents[(str(cohort), rep)]
            codes.append(matrix[index_of[str(scan_key)]])
        raw[split] = np.stack(codes).astype(np.float32)
    mean = raw["train"].mean(axis=0).astype(np.float32)
    std = np.maximum(raw["train"].std(axis=0), np.float32(1.0e-8)).astype(np.float32)
    for split, archive in archives.items():
        payload = dict(archive) | {
            "visit_latent_raw_128": raw[split],
            "visit_latent_standardized_128": ((raw[split] - mean) / std).astype(np.float32),
            "train_latent_mean_128": mean,
            "train_latent_std_128": std,
            "representation_name": np.asarray(rep),
            "representation_kind": np.asarray(rep_spec["kind"]),
        }
        if rep_spec["kind"] == "lamm_ae":
            payload["latent_scale_names_128"] = np.asarray(rep_spec["latent_scale_names"], dtype=str)
            payload["latent_scale_offsets_128"] = np.asarray([0, *np.cumsum(rep_spec["expected_latent_split"])], dtype=np.int64)
        path = bc.atomic_npz(root / "representations" / rep / f"{split}_subject_sequences_128.npz", payload)
        files[f"representations/{rep}/{split}"] = bc.sha256_file(path)
    bc.atomic_json(root / "representations" / rep / "manifest.json", {
        "representation": rep,
        "kind": rep_spec["kind"],
        "latent_dim": bc.LATENT_DIM,
        "checkpoint": rep_spec.get("checkpoint"),
        "checkpoint_sha256": rep_spec.get("checkpoint_sha256"),
        "train_only_standardization": True,
        "standardization_fitted_on": "view train rows",
        "split_counts": {split: int(len(archives[split]["visit_scan_ids"])) for split in bc.SPLITS},
    })
    return files


def main() -> int:
    args = parse_args()
    sources = bc.load_cohort_sources()
    registry = bc.load_registry()
    tasks = bc.load_tasks()
    config = bc.load_protocol_config()
    views = bc.expand_views(config)
    names = args.views or list(views)
    unknown = sorted(set(names).difference(views))
    if unknown:
        raise KeyError(f"unknown views {unknown}")

    cohorts = sorted({cohort for name in names for cohort in cohorts_of(views[name])})
    frames = {cohort: bc.read_strict_manifest(cohort, sources) for cohort in cohorts}
    crossfit = config["crossfit"]
    folds = {}
    for cohort in crossfit["cohorts"]:
        if cohort in frames:
            folds[cohort] = crossfit_table(cohort, frames[cohort], int(crossfit["folds"]), int(crossfit["seed"]))
            bc.atomic_csv(bc.require_bulk(bc.STAGE1_ROOT / "folds" / f"{cohort}_cv{crossfit['folds']}.csv"), folds[cohort])
    latents = {(cohort, rep): load_latents(cohort, rep) for cohort in cohorts for rep in registry["representations"]}
    for cohort in cohorts:
        index_of, _ = latents[(cohort, bc.REPRESENTATIONS[0])]
        missing = set(frames[cohort]["scan_key"]).difference(index_of)
        if missing:
            raise KeyError(f"{cohort}: {len(missing)} strict scans have no codes, e.g. {sorted(missing)[:3]}")

    summary = {}
    for name in names:
        manifest = build_view(name, views[name], frames, folds, latents, registry, tasks, args.overwrite)
        summary[name] = {split: manifest["splits"][split]["subjects"] for split in bc.SPLITS}
        print(f"[{name}] subjects {summary[name]} | test tasks "
              f"{ {task: manifest['tasks'][f'test/{task}']['subjects'] for task in tasks['tasks']} }", flush=True)
    bc.atomic_json(bc.STAGE1_ROOT / "views" / "views_summary.json", summary)
    return 0


def cohorts_of(spec: dict[str, Any]) -> set[str]:
    if "crossfit" in spec:
        return {spec["crossfit"]["cohort"]}
    return {cohort for split in bc.SPLITS for cohort, _member in spec[split]}


if __name__ == "__main__":
    raise SystemExit(main())
