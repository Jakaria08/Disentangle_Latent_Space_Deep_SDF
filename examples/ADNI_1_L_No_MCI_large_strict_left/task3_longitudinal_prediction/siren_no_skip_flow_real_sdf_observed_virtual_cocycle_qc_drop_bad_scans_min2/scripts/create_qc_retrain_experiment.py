#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    source_experiment = Path(__file__).resolve().parents[1]
    default_output = source_experiment.parent / (
        source_experiment.name + "_qc_drop_bad_scans_min2"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Create a separate direct-flow experiment using QC-clean metadata "
            "and matching filtered latent NPZ files."
        )
    )
    parser.add_argument(
        "--source-experiment",
        default=str(source_experiment),
        help="Existing trained direct-flow experiment to clone configuration from.",
    )
    parser.add_argument(
        "--clean-metadata",
        default=str(
            source_experiment
            / "analysis"
            / "mesh_longitudinal_qc"
            / "metadata_drop_bad_scans_min2.csv"
        ),
        help="QC-clean metadata CSV produced by qc_longitudinal_mesh_consistency.py.",
    )
    parser.add_argument(
        "--output-experiment",
        default=str(default_output),
        help="New experiment directory to create.",
    )
    parser.add_argument(
        "--metadata-name",
        default="adni_large_strict_no_mci_left_direct_flow_records_qc_drop_bad_scans_min2.csv",
        help="File name to use under the new experiment metadata/ directory.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow updating generated metadata/latents/specs in an existing output directory.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_metadata(frame: pd.DataFrame) -> None:
    required = {
        "scan_id",
        "subject_id",
        "split",
        "diagnosis",
        "visit_order",
        "continuous_age_norm",
        "continuous_age_years",
        "label_ad",
        "sdf_npz_path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Clean metadata is missing required columns: {missing}")
    duplicated = frame["scan_id"].astype(str).duplicated()
    if duplicated.any():
        first = frame.loc[duplicated, "scan_id"].astype(str).iloc[0]
        raise ValueError(f"Duplicate scan_id in clean metadata: {first}")
    splits = set(frame["split"].astype(str))
    if splits != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test splits, got {sorted(splits)}")
    diagnosis_values = set(frame["diagnosis"].astype(str))
    if diagnosis_values != {"CN", "AD"}:
        raise ValueError(f"Expected CN/AD diagnoses, got {sorted(diagnosis_values)}")
    subject_splits = frame.groupby("subject_id")["split"].nunique()
    crossing = subject_splits.loc[subject_splits > 1]
    if not crossing.empty:
        raise ValueError(f"Subjects cross splits; first={crossing.index[0]}")
    subject_diagnoses = frame.groupby("subject_id")["diagnosis"].nunique()
    changing = subject_diagnoses.loc[subject_diagnoses > 1]
    if not changing.empty:
        raise ValueError(f"Subjects change diagnosis; first={changing.index[0]}")


def load_latent_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    if "scan_ids" not in payload or "latents" not in payload:
        raise ValueError(f"Latent archive lacks scan_ids/latents: {path}")
    return payload["scan_ids"].astype(str), payload["latents"]


def filtered_latents_for_split(
    *,
    source_latent_path: Path,
    clean_metadata: pd.DataFrame,
    split: str,
    output_path: Path,
) -> Dict[str, Any]:
    source_scan_ids, source_latents = load_latent_archive(source_latent_path)
    index_by_scan_id = {scan_id: index for index, scan_id in enumerate(source_scan_ids)}
    split_frame = clean_metadata.loc[
        clean_metadata["split"].astype(str) == split
    ].copy()
    ordered_scan_ids = split_frame["scan_id"].astype(str).tolist()
    missing = [scan_id for scan_id in ordered_scan_ids if scan_id not in index_by_scan_id]
    if missing:
        raise KeyError(
            f"{len(missing)} clean {split} scans are missing frozen latents; "
            f"first={missing[0]}"
        )
    selected_indices = [index_by_scan_id[scan_id] for scan_id in ordered_scan_ids]
    selected_latents = np.asarray(source_latents[selected_indices], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        scan_ids=np.asarray(ordered_scan_ids, dtype=str),
        latents=selected_latents,
    )
    return {
        "source": str(source_latent_path),
        "output": str(output_path),
        "scan_count": int(len(ordered_scan_ids)),
        "latent_dim": int(selected_latents.shape[1]),
        "dropped_extra_latents": int(len(source_scan_ids) - len(ordered_scan_ids)),
    }


def pair_count_for_subject_lengths(lengths: Iterable[int]) -> int:
    return int(sum(length * (length - 1) // 2 for length in lengths))


def observed_intermediate_pair_count(lengths: Iterable[int]) -> int:
    total = 0
    for length in lengths:
        if length >= 3:
            total += int((length - 1) * (length - 2) // 2)
    return total


def split_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split, split_frame in frame.groupby("split", sort=True):
        lengths = split_frame.groupby("subject_id").size().astype(int).tolist()
        result[str(split)] = {
            "scans": int(len(split_frame)),
            "subjects": int(split_frame["subject_id"].nunique()),
            "pairs": pair_count_for_subject_lengths(lengths),
            "pairs_with_observed_intermediate": observed_intermediate_pair_count(lengths),
            "diagnosis_scans": {
                str(key): int(value)
                for key, value in split_frame["diagnosis"].value_counts().items()
            },
            "diagnosis_subjects": {
                str(key): int(value)
                for key, value in (
                    split_frame.drop_duplicates("subject_id")["diagnosis"]
                    .value_counts()
                    .items()
                )
            },
        }
    return result


def copy_support_files(source_experiment: Path, output_experiment: Path) -> None:
    readme = source_experiment / "README.md"
    if readme.is_file():
        shutil.copy2(readme, output_experiment / "README_original.md")
    source_scripts = source_experiment / "scripts"
    output_scripts = output_experiment / "scripts"
    output_scripts.mkdir(parents=True, exist_ok=True)
    if source_scripts.is_dir():
        for script_path in source_scripts.glob("*.py"):
            shutil.copy2(script_path, output_scripts / script_path.name)


def write_retrain_readme(
    path: Path,
    *,
    output_experiment: Path,
    clean_metadata_path: Path,
    summary: Dict[str, Any],
) -> None:
    text = f"""# QC-Clean Direct Flow Retrain

This experiment is a clean retrain copy of
`siren_no_skip_flow_real_sdf_observed_virtual_cocycle`.

QC metadata source:

`{clean_metadata_path}`

The SIREN decoder is still frozen. Only the longitudinal direct-flow model is
trained here. The latent NPZ files were filtered so each split contains exactly
the scan IDs present in the QC-clean metadata.

## Counts

```json
{json.dumps(summary["splits"], indent=2)}
```

## Commands

From repo root:

```bash
cd /home/jakaria/INR/Deep3DComp
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e {output_experiment} --validate-only
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e {output_experiment} --gpu 0 --smoke-batches 2
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e {output_experiment} --gpu 0
```

After training:

```bash
cd /home/jakaria/INR/Deep3DComp
/home/jakaria/anaconda3/envs/inr_sdf/bin/python evaluate_deep_sdf_longitudinal_direct_flow.py -e {output_experiment} --checkpoint best --split all --gpu 0
/home/jakaria/anaconda3/envs/inr_sdf/bin/python {output_experiment}/scripts/summarize_gap_bins.py --analysis {output_experiment}/analysis/checkpoint_best
/home/jakaria/anaconda3/envs/inr_sdf/bin/python {output_experiment}/scripts/run_rich_visualization_analysis.py --checkpoint best --gpu 0
```

If `best.pth` is not created, evaluate `best_candidate` instead.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_experiment = Path(args.source_experiment).expanduser().resolve()
    clean_metadata_path = Path(args.clean_metadata).expanduser().resolve()
    output_experiment = Path(args.output_experiment).expanduser().resolve()

    if not source_experiment.is_dir():
        raise FileNotFoundError(f"Missing source experiment: {source_experiment}")
    if not clean_metadata_path.is_file():
        raise FileNotFoundError(f"Missing clean metadata: {clean_metadata_path}")
    if output_experiment.exists() and not args.allow_existing:
        raise FileExistsError(
            f"Output experiment already exists: {output_experiment}. "
            "Pass --allow-existing to update generated files."
        )
    output_experiment.mkdir(parents=True, exist_ok=True)
    (output_experiment / "metadata").mkdir(exist_ok=True)
    (output_experiment / "latents").mkdir(exist_ok=True)

    source_specs = read_json(source_experiment / "specs.json")
    clean_metadata = pd.read_csv(clean_metadata_path)
    clean_metadata["scan_id"] = clean_metadata["scan_id"].astype(str)
    clean_metadata["subject_id"] = clean_metadata["subject_id"].astype(str)
    clean_metadata["split"] = clean_metadata["split"].astype(str)
    clean_metadata = clean_metadata.sort_values(
        ["split", "subject_id", "continuous_age_norm", "visit_order", "scan_id"]
    ).reset_index(drop=True)
    validate_metadata(clean_metadata)

    metadata_output = output_experiment / "metadata" / str(args.metadata_name)
    clean_metadata.to_csv(metadata_output, index=False)

    latent_outputs: Dict[str, Any] = {}
    latent_specs: Dict[str, str] = {}
    source_latent_specs = source_specs.get("FrozenLatentFiles")
    if not isinstance(source_latent_specs, dict):
        raise TypeError("Source specs FrozenLatentFiles must be an object.")
    for split in ("train", "val", "test"):
        source_latent_path = resolve_path(source_latent_specs[split], source_experiment)
        output_latent_path = output_experiment / "latents" / f"{split}_latents.npz"
        latent_outputs[split] = filtered_latents_for_split(
            source_latent_path=source_latent_path,
            clean_metadata=clean_metadata,
            split=split,
            output_path=output_latent_path,
        )
        latent_specs[split] = f"latents/{split}_latents.npz"

    specs = dict(source_specs)
    specs["Description"] = list(source_specs.get("Description", [])) + [
        "QC-clean retrain: bad longitudinal mesh scans were removed with "
        "metadata_drop_bad_scans_min2.csv, keeping subjects with at least two "
        "remaining visits."
    ]
    specs["ExperimentTag"] = (
        str(source_specs.get("ExperimentTag", "direct_flow"))
        + "_qc_drop_bad_scans_min2"
    )
    metadata_relative = f"metadata/{metadata_output.name}"
    specs["TrainSplit"] = metadata_relative
    specs["TestSplit"] = metadata_relative
    specs["FinalTestSplit"] = metadata_relative
    specs["LongitudinalMetadataFile"] = metadata_relative
    specs["FrozenLatentFiles"] = latent_specs
    specs["QualityControlFilter"] = {
        "rule": "drop_bad_scans_and_keep_subjects_with_at_least_two_visits",
        "source_qc_report": str(
            clean_metadata_path.parent / "qc_summary.json"
        ),
        "source_metadata": str(clean_metadata_path),
        "minimum_scans_after_filter": 2,
    }
    write_json(output_experiment / "specs.json", specs)

    summary = {
        "source_experiment": str(source_experiment),
        "output_experiment": str(output_experiment),
        "clean_metadata_source": str(clean_metadata_path),
        "metadata_output": str(metadata_output),
        "splits": split_summary(clean_metadata),
        "latent_outputs": latent_outputs,
    }
    write_json(output_experiment / "metadata" / "qc_retrain_summary.json", summary)
    copy_support_files(source_experiment, output_experiment)
    write_retrain_readme(
        output_experiment / "README.md",
        output_experiment=output_experiment,
        clean_metadata_path=clean_metadata_path,
        summary=summary,
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
