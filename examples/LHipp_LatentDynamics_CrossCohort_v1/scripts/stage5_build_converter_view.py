#!/usr/bin/env python3
"""Stage 5 step 2: the converter-line view ``p5_converter_pooled`` (PLAN Part 5B).

The P3 pooled protocol extended with the inclusive AIBL and OASIS cohorts:

* ADNI strict subjects (CN-stable, AD-stable) keep their P3 split;
* AIBL/OASIS inclusive subjects in CN-stable, AD-stable, CN->AD, MCI->AD and CN->MCI keep the inclusive split,
  which preserves every strict subject's split (stage-1 gate G1.7); MCI-stable subjects and reverters are left out;
* ages and codes use the P3 train normalization instead of refitting, so every P3 checkpoint (C0, BrainODE-core
  and the C1 initialization) runs on this view unchanged and evaluate_dynamics.py's normalization guard accepts it;
* the binary archive label is the source-visit label with MCI coded 0 (not yet AD), which is what fixed-label
  comparators receive. The CN/MCI/AD visit labels, trajectory groups and conversion windows are extra keys.

Writes stage1_data_foundation/views/p5_converter_pooled/ in the stage-1 layout. Refuses to overwrite unless
--overwrite.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

import benchmark_common as bc
import dynamics_core as D
import stage1_build_protocol_views as views

CONFIG = bc.read_json(bc.CONFIG_DIR / "converter_line.json")
VIEW, BASE_VIEW = CONFIG["view"], CONFIG["base_view"]
KEEP = tuple(CONFIG["groups"]["kept"])
COLUMNS = ["cohort", "subject_key", "scan_key", "split", "visit_month", "age_years", "label", "visit_label", "diagnosis_source",
           "trajectory_group", "conv_to_ad_window_a_years", "conv_to_ad_window_b_years", "conv_to_mci_window_a_years",
           "conv_to_mci_window_b_years", "correspondence_volume_mm3", "correspondence_surface_area_mm2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None, help="Ignored (CPU only); accepted because the orchestrator appends it.")
    return parser.parse_args()


def member_rows() -> pd.DataFrame:
    adni = bc.read_strict_manifest("adni", bc.load_cohort_sources())
    adni = adni.assign(
        visit_label=np.where(adni["label"] == 1, "AD", "CN"),
        trajectory_group=np.where(adni["label"] == 1, "AD-stable", "CN-stable"),
        conv_to_ad_window_a_years=np.nan, conv_to_ad_window_b_years=np.nan,
        conv_to_mci_window_a_years=np.nan, conv_to_mci_window_b_years=np.nan,
    )
    parts = [adni[COLUMNS]]
    for cohort in ("aibl", "oasis"):
        frame = pd.read_csv(bc.STAGE1_ROOT / "cohorts" / cohort / "inclusive_manifest.csv",
                            dtype={"subject_key": str, "scan_key": str, "subject_id": str, "scan_id": str, "VISCODE": str})
        frame = frame.loc[frame["trajectory_group"].isin(KEEP)].assign(diagnosis_source=lambda f: f["visit_label"])
        parts.append(frame[COLUMNS])
    rows = pd.concat(parts, ignore_index=True)
    if rows[["age_years", "visit_month", "correspondence_volume_mm3"]].isna().any().any():
        raise ValueError("converter view rows with missing age, visit time or volume")
    if rows["scan_key"].duplicated().any():
        raise ValueError("a scan appears twice in the converter view")
    if rows.groupby("subject_key")["split"].nunique().gt(1).any():
        raise ValueError("a subject appears in more than one split")
    if not (rows["label"] == (rows["visit_label"] == "AD").astype(int)).all():
        raise ValueError("binary label must be 1 exactly for AD visits")
    return rows.assign(view_split=rows["split"], source_split=rows["split"])


def add_converter_keys(archive: dict[str, np.ndarray], rows: pd.DataFrame) -> dict[str, np.ndarray]:
    ordered = rows.sort_values(["subject_key", "visit_month", "scan_key"], kind="stable").reset_index(drop=True)
    if not np.array_equal(ordered["scan_key"].to_numpy(dtype=str), archive["visit_scan_ids"]):
        raise RuntimeError("converter keys are not aligned with the sequence archive")
    subjects = ordered.drop_duplicates("subject_key", keep="first").set_index("subject_key").loc[archive["subject_ids"]]
    windows = lambda prefix: np.stack([subjects[f"{prefix}_a_years"].to_numpy(np.float32), subjects[f"{prefix}_b_years"].to_numpy(np.float32)], axis=1)
    return archive | {
        "visit_trajectory_labels": ordered["visit_label"].to_numpy(dtype=str),
        "subject_trajectory_groups": subjects["trajectory_group"].to_numpy(dtype=str),
        "subject_conv_to_ad_window_years": windows("conv_to_ad_window"),
        "subject_conv_to_mci_window_years": windows("conv_to_mci_window"),
    }


def write_representation(root, archives: dict[str, dict[str, np.ndarray]], rep: str, spec: dict[str, Any]) -> dict[str, str]:
    """Codes standardized with the base view's train statistics (not refitted)."""
    base_train = bc.load_npz(D.view_root(BASE_VIEW) / "representations" / rep / "train_subject_sequences_128.npz")
    mean, std = base_train["train_latent_mean_128"], base_train["train_latent_std_128"]
    latents = {cohort: views.load_latents(cohort, rep) for cohort in ("adni", "aibl", "oasis")}
    files = {}
    for split, archive in archives.items():
        raw = np.stack([latents[str(c)][1][latents[str(c)][0][str(k)]] for k, c in zip(archive["visit_scan_ids"], archive["visit_cohorts"])]).astype(np.float32)
        payload = dict(archive) | {
            "visit_latent_raw_128": raw, "visit_latent_standardized_128": ((raw - mean) / std).astype(np.float32),
            "train_latent_mean_128": mean, "train_latent_std_128": std,
            "representation_name": np.asarray(rep), "representation_kind": np.asarray(spec["kind"]),
        }
        path = bc.atomic_npz(root / "representations" / rep / f"{split}_subject_sequences_128.npz", payload)
        files[f"representations/{rep}/{split}"] = bc.sha256_file(path)
    bc.atomic_json(root / "representations" / rep / "manifest.json", {
        "representation": rep, "kind": spec["kind"], "latent_dim": bc.LATENT_DIM, "checkpoint": spec.get("checkpoint"),
        "checkpoint_sha256": spec.get("checkpoint_sha256"), "train_only_standardization": True,
        "standardization_fitted_on": f"{BASE_VIEW} train rows (copied, not refitted)",
        "split_counts": {split: int(len(archives[split]["visit_scan_ids"])) for split in bc.SPLITS},
    })
    return files


def check_against_base(archives: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    base = {split: bc.load_npz(D.view_root(BASE_VIEW) / "dataset" / f"{split}_subject_sequences.npz") for split in bc.SPLITS}
    base_split = {str(s): split for split, a in base.items() for s in a["subject_ids"]}
    moved = [str(s) for split, a in archives.items() for s in a["subject_ids"] if base_split.get(str(s), split) != split]
    missing = sorted(set(base_split).difference(str(s) for a in archives.values() for s in a["subject_ids"]))
    store = D.VertexStore()
    keys = [str(k) for a in archives.values() for k in a["visit_scan_ids"]]
    for cohort in sorted({key.split(bc.ID_SEPARATOR, 1)[0] for key in keys}):
        store._cohort(cohort)  # loads that cohort's cache index
    uncached = [key for key in keys if key not in store._index[key.split(bc.ID_SEPARATOR, 1)[0]]]
    return {"strict_subjects_moved_split": moved, "base_subjects_missing": missing, "scans_without_mesh": uncached}


def main() -> int:
    args = parse_args()
    root = bc.require_bulk(bc.STAGE1_ROOT / "views" / VIEW)
    if (root / "view_manifest.json").exists() and not args.overwrite:
        print(f"[{VIEW}] exists; pass --overwrite to rebuild")
        return 0
    rows = member_rows()
    normalization = bc.read_json(D.view_root(BASE_VIEW) / "view_manifest.json")["normalization"]
    age_min, age_max = float(normalization["age_min_years"]), float(normalization["age_max_years"])
    archives = {split: add_converter_keys(views.sequence_archive(rows.loc[rows["view_split"] == split].copy(), age_min, age_max),
                                          rows.loc[rows["view_split"] == split]) for split in bc.SPLITS}
    checks = check_against_base(archives)
    if any(checks.values()):
        raise RuntimeError(f"converter view inconsistent with {BASE_VIEW}: { {k: v[:5] for k, v in checks.items()} }")

    tasks = bc.load_tasks()
    manifest: dict[str, Any] = {
        "view": VIEW, "spec": {"protocol": "P5-converter", "description": CONFIG["note"], "base_view": BASE_VIEW, "kept_groups": list(KEEP)},
        "normalization": {"fitted_on": f"{BASE_VIEW} train rows (copied, not refitted)", "age_min_years": age_min, "age_max_years": age_max},
        "canonical_labels": {"0": "CN or MCI (not yet AD)", "1": "AD"}, "splits": {}, "tasks": {}, "files": {}, "checks": checks,
    }
    for split, archive in archives.items():
        manifest["files"][f"dataset/{split}"] = bc.sha256_file(bc.atomic_npz(root / "dataset" / f"{split}_subject_sequences.npz", archive))
        pairs = views.forward_pairs(archive, split)
        manifest["files"][f"pairs/{split}"] = bc.sha256_file(bc.atomic_csv(root / "pairs" / f"{split}_forward_pairs.csv", pairs))
        groups = pd.DataFrame({"cohort": archive["subject_cohorts"], "group": archive["subject_trajectory_groups"]})
        manifest["splits"][split] = {"subjects": int(len(archive["subject_ids"])), "scans": int(len(archive["visit_scan_ids"])), "pairs": int(len(pairs)),
                                     "subjects_by_cohort_group": {f"{c}|{g}": int(n) for (c, g), n in groups.value_counts().sort_index().items()}}
        if split in ("val", "test"):
            for task, spec in tasks["tasks"].items():
                table = views.task_table(archive, task, spec)
                bc.atomic_csv(root / "tasks" / f"{split}_{task}.csv", table)
                manifest["tasks"][f"{split}/{task}"] = {"subjects": int(len(table))}
    registry = bc.load_registry()
    for rep in CONFIG["representations"]:
        manifest["files"].update(write_representation(root, archives, rep, registry["representations"][rep]))
    bc.atomic_json(root / "view_manifest.json", manifest)
    for split in bc.SPLITS:
        print(f"[{VIEW}] {split}: {manifest['splits'][split]['subjects']} subjects, {manifest['splits'][split]['scans']} scans | "
              f"{manifest['splits'][split]['subjects_by_cohort_group']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
