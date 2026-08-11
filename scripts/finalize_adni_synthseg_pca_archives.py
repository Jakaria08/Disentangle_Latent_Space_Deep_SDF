#!/usr/bin/env python3
"""Create pickle-free PCA coefficient archives from completed PCA outputs.

Completed PCA score arrays are preserved exactly.  Only object-dtype string
metadata (scan IDs, subject IDs, split labels, diagnoses) is rewritten as
fixed-width Unicode, so downstream code can safely load every array with
``allow_pickle=False``.  Original archives remain untouched as provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
EXPERIMENTS = {
    "hippocampus": {
        "root": BASE_ROOT / "hippocampus_pca_cocycle_v4",
        "manifest": "hippocampus_qc_keep_manifest.csv",
    },
    "lateral_ventricle": {
        "root": BASE_ROOT / "lateral_ventricle_pca_cocycle_v4",
        "manifest": "lateral_ventricle_qc_keep_manifest.csv",
    },
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", choices=("hippocampus", "lateral_ventricle", "all"), default="all")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_array(value: np.ndarray, key: str) -> np.ndarray:
    """Convert only string-object metadata; reject arbitrary object payloads."""
    array = np.asarray(value)
    if array.dtype != object:
        return array
    flattened = array.ravel().tolist()
    if not all(isinstance(item, str) for item in flattened):
        raise TypeError(f"Archive key {key!r} has non-string object values and cannot be safely converted.")
    return np.asarray(flattened, dtype=np.str_).reshape(array.shape)


def load_original(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def validate_replacement(original: dict[str, np.ndarray], replacement: Path) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    with np.load(replacement, allow_pickle=False) as archive:
        if set(archive.files) != set(original):
            raise RuntimeError(f"Archive key mismatch in {replacement}")
        for key in original:
            new = archive[key]
            old = original[key]
            if new.dtype == object:
                raise RuntimeError(f"Pickle-dependent object dtype remained in {replacement}: {key}")
            if old.dtype == object:
                equal = np.array_equal(new.astype(object), old)
            else:
                equal = np.array_equal(new, old)
            if not equal:
                raise RuntimeError(f"Archive value changed during safe conversion: {replacement.name} / {key}")
            dtypes[key] = str(new.dtype)
    return dtypes


def finalize_structure(name: str) -> dict[str, Any]:
    spec = EXPERIMENTS[name]
    root = Path(spec["root"])
    coefficient_dir = root / "pca" / "coefficients"
    manifest_path = root / "metadata" / spec["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype={"scan_id": str, "subject_id": str})
    archive_names = ["all_coefficients.npz", *[f"{split}_coefficients.npz" for split in SPLITS]]
    results: dict[str, Any] = {}
    for source_name in archive_names:
        source = coefficient_dir / source_name
        target = coefficient_dir / source_name.replace(".npz", "_safe.npz")
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(f"Safe replacement already exists and is protected: {target}")
        original = load_original(source)
        converted = {key: safe_array(value, key) for key, value in original.items()}
        np.savez_compressed(target, **converted)
        dtypes = validate_replacement(original, target)
        results[source_name] = {
            "original": str(source),
            "replacement": str(target),
            "keys": sorted(original),
            "safe_dtypes": dtypes,
        }

    all_safe = coefficient_dir / "all_coefficients_safe.npz"
    with np.load(all_safe, allow_pickle=False) as archive:
        if not np.array_equal(archive["scan_ids"].astype(str), manifest["scan_id"].astype(str).to_numpy()):
            raise RuntimeError(f"Safe all-coefficient scan order does not match {manifest_path}")
        if archive["pca_150"].shape != (len(manifest), 150) or archive["pca_standardized_150"].shape != (len(manifest), 150):
            raise RuntimeError(f"Unexpected PCA score shape in {all_safe}")
        if not np.isfinite(archive["pca_150"]).all() or not np.isfinite(archive["pca_standardized_150"]).all():
            raise RuntimeError(f"Non-finite PCA scores in {all_safe}")
    report = {
        "passed": True,
        "structure": name,
        "source_meshes_modified": False,
        "pca_refitted": False,
        "original_archives_preserved": True,
        "original_archive_status": "superseded_for_downstream_loading_only",
        "safe_archive_status": "active_for_downstream_loading",
        "manifest": str(manifest_path),
        "scans": int(len(manifest)),
        "subjects": int(manifest["subject_id"].nunique()),
        "archives": results,
        "safe_all_archive": str(all_safe),
        "load_policy": "all safe archives validated with numpy.load(..., allow_pickle=False)",
    }
    write_json(root / "pca" / "metadata" / "pca_archive_finalization.json", report)
    return report


def main() -> int:
    args = parse_args()
    names = tuple(EXPERIMENTS) if args.structure == "all" else (args.structure,)
    print("=" * 88)
    print("Finalize PCA coefficient archives without refitting PCA")
    print("Original archives are preserved; only safe replacements are created.")
    print("=" * 88, flush=True)
    reports = {}
    for name in names:
        print(f"Finalizing {name}…", flush=True)
        reports[name] = finalize_structure(name)
        print(f"  safe archive: {reports[name]['safe_all_archive']}", flush=True)
    print(json.dumps({name: {"scans": report["scans"], "passed": report["passed"]} for name, report in reports.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
