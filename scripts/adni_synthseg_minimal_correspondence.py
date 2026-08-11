#!/usr/bin/env python3
"""Create raw, minimally smoothed, and correspondence meshes from ADNI SynthSeg.

The pipeline deliberately keeps the three mesh representations separate:

``raw_ply``
    Direct marching-cubes meshes from the requested SynthSeg label.  These are
    retained for every processed scan and are never used as correspondence
    inputs.
``minimal_smooth_ply``
    A deliberately conservative smoothing pass.  By default it applies only
    Gaussian smoothing with sigma 0.5 voxel.  If that result is not
    watertight, the pipeline escalates only as far as local voxel hole filling,
    one-voxel closing, and finally mesh-hole filling.
``minimal_smooth_correspondence/final_ply``
    ShapeWorks rigid alignment plus a per-structure Deformetrica atlas.  The
    resulting meshes share both vertex order and face connectivity, and are
    rescaled to the minimally smoothed input volume in the legacy prepared
    coordinate scale.  Physical-mm counterparts are saved in ``final_ply_mm``
    and legacy-compatible fixed-range OBJ files in ``minimal_scaled_obj_files``.

The default ``run`` command is a safe three-subject CN/AD pilot for the left
hippocampus and left lateral ventricle.  It completes and saves the raw meshes
for the entire selection before beginning any smoothing.  Use
``--sample-count 0`` only for a full run.  A pilot and a full run use different
output directories by default.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path("/home/jakaria/ADNI/ADNI_1_GO_Large")
DEFAULT_SEGMENTATION_ROOT = DEFAULT_BASE_DIR / "adni_synthseg" / "segmentations"
DEFAULT_CLINICAL_CSV = DEFAULT_BASE_DIR / "ClinicalInfo.csv"
DEFAULT_OUTPUT_BASE = DEFAULT_BASE_DIR / "synthseg_minimal_correspondence"
DEFAULT_SHAPEWORKS_PYTHON = "/home/jakaria/anaconda3/envs/shapeworks/bin/python"
DEFAULT_DEFORMETRICA_PYTHON = (
    "/home/jakaria/Explaining_Shape_Variability/preprocessing/deformetrica_reg/bin/python"
)
DEFAULT_INR_PYTHON = "/home/jakaria/anaconda3/envs/inr_sdf/bin/python"
DEFAULT_NOTEBOOK_SCRIPT = REPO_ROOT / "scripts" / "create_adni_synthseg_correspondence_notebook.py"
DEFAULT_NOTEBOOK_OUTPUT_DIR = (
    REPO_ROOT / "examples" / "ADNI_1_L_No_MCI_large_strict_left" / "task3_longitudinal_prediction" / "notebooks"
)

GLOBAL_SCALE_BUFFER = 1.2
TARGET_RANGE_MIN = -0.9
TARGET_RANGE_MAX = 0.9
RECON_SUBJECT_RE = re.compile(r"__subject_(.+)\.vtk$")

# These maps match the existing ADNI mask-processing code in this repository.
VISIT_DX_MAP = {
    "NL": "CN",
    "MCI": "MCI",
    "Dementia": "AD",
    "NL to MCI": "MCI",
    "NL to Dementia": "AD",
    "MCI to Dementia": "AD",
    "MCI to NL": "CN",
    "Dementia to MCI": "MCI",
}
SEX_MAP = {"M": 0.0, "MALE": 0.0, "F": 1.0, "FEMALE": 1.0}


@dataclass(frozen=True)
class StructureSpec:
    name: str
    label: int
    display_name: str
    side: str


STRUCTURES: dict[str, StructureSpec] = {
    "left_hippocampus": StructureSpec("left_hippocampus", 17, "Left hippocampus", "left"),
    "right_hippocampus": StructureSpec("right_hippocampus", 53, "Right hippocampus", "right"),
    "left_lateral_ventricle": StructureSpec(
        "left_lateral_ventricle", 4, "Left lateral ventricle", "left"
    ),
    "right_lateral_ventricle": StructureSpec(
        "right_lateral_ventricle", 43, "Right lateral ventricle", "right"
    ),
}
DEFAULT_STRUCTURES = ("left_hippocampus", "left_lateral_ventricle")
DEFAULT_DIAGNOSES = ("CN", "AD")
DIAGNOSIS_CHOICES = ("CN", "AD", "MCI")
STRUCTURE_ARGUMENT_CHOICES = ("all", *STRUCTURES)
DIAGNOSIS_ARGUMENT_CHOICES = ("all", *DIAGNOSIS_CHOICES)
SUCCESS_STATUSES = frozenset({"ok", "skipped_existing"})


@dataclass(frozen=True)
class RawMeshTask:
    scan_id: str
    rid: str
    viscode: str
    diagnosis: str
    segmentation_path: str
    structure_name: str
    label: int
    raw_path: str
    overwrite: bool


@dataclass(frozen=True)
class SmoothMeshTask:
    scan_id: str
    rid: str
    viscode: str
    diagnosis: str
    segmentation_path: str
    structure_name: str
    label: int
    raw_path: str
    smooth_path: str
    gaussian_sigma: float
    closing_iterations: int
    fill_holes: bool
    crop_margin: int
    component_policy: str
    overwrite: bool


def print_header(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_dataframe(path: Path):
    import pandas as pd

    return pd.read_csv(path, dtype={"RID": "string", "VISCODE": "string", "scan_id": "string"})


def save_dataframe(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def reports_dir(output_root: Path) -> Path:
    return output_root / "reports"


def manifests_dir(output_root: Path) -> Path:
    return output_root / "manifests"


def selected_manifest_path(output_root: Path) -> Path:
    return manifests_dir(output_root) / "selected_scans.csv"


def structure_root(output_root: Path, structure_name: str) -> Path:
    return output_root / structure_name


def raw_ply_dir(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "raw_ply"


def smooth_ply_dir(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "minimal_smooth_ply"


def prepared_ply_dir(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "minimal_smooth_prepared_ply"


def rigid_ply_dir(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "minimal_smooth_rigid_ply"


def correspondence_root(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "minimal_smooth_correspondence"


def correspondence_vtk_dir(output_root: Path, structure_name: str) -> Path:
    return correspondence_root(output_root, structure_name) / "input_vtk"


def deformetrica_dir(output_root: Path, structure_name: str) -> Path:
    return correspondence_root(output_root, structure_name) / "deformetrica"


def final_ply_dir(output_root: Path, structure_name: str) -> Path:
    return correspondence_root(output_root, structure_name) / "final_ply"


def final_ply_mm_dir(output_root: Path, structure_name: str) -> Path:
    return correspondence_root(output_root, structure_name) / "final_ply_mm"


def final_vtk_dir(output_root: Path, structure_name: str) -> Path:
    return correspondence_root(output_root, structure_name) / "final_vtk"


def scaled_obj_dir(output_root: Path, structure_name: str) -> Path:
    return correspondence_root(output_root, structure_name) / "minimal_scaled_obj_files"


def mesh_qc_path(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "mesh_qc.csv"


def metadata_path(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "prepare_metadata.json"


def prepare_details_path(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "prepare_details.csv"


def rigid_details_path(output_root: Path, structure_name: str) -> Path:
    return structure_root(output_root, structure_name) / "rigid_details.csv"


def volume_lineage_path(output_root: Path) -> Path:
    return reports_dir(output_root) / "mesh_volume_lineage.csv"


def viscode_to_month(value: Any) -> float:
    text = str(value).strip()
    if text == "bl":
        return 0.0
    if text.startswith("m") and text[1:].isdigit():
        return float(int(text[1:]))
    return math.nan


def normalize_sex(value: Any) -> float | None:
    if value is None:
        return None
    result = SEX_MAP.get(str(value).strip().upper())
    return result if result is not None else None


def parse_structures(values: Sequence[str] | None) -> list[StructureSpec]:
    requested = list(values or DEFAULT_STRUCTURES)
    if "all" in requested:
        requested = list(STRUCTURES)
    result: list[StructureSpec] = []
    for name in requested:
        if name not in STRUCTURES:
            raise ValueError(f"Unsupported structure '{name}'. Choices: {', '.join(STRUCTURES)}")
        if STRUCTURES[name] not in result:
            result.append(STRUCTURES[name])
    return result


def parse_diagnoses(values: Sequence[str] | None) -> list[str]:
    diagnoses = [str(value).upper() for value in (values or DEFAULT_DIAGNOSES)]
    if "ALL" in diagnoses:
        diagnoses = list(DIAGNOSIS_CHOICES)
    invalid = sorted(set(diagnoses).difference(DIAGNOSIS_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported diagnoses: {invalid}")
    return list(dict.fromkeys(diagnoses))


def read_clinical(path: Path):
    import pandas as pd

    frame = pd.read_csv(path, dtype={"RID": "string", "VISCODE": "string"}, keep_default_na=False)
    required = {"RID", "VISCODE", "DX", "DX.bl", "AGE", "PTGENDER"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {missing}")
    frame["RID"] = frame["RID"].astype(str).str.strip()
    frame["VISCODE"] = frame["VISCODE"].astype(str).str.strip()
    frame["scan_id"] = frame["RID"] + "_" + frame["VISCODE"]
    frame["month_from_viscode"] = frame["VISCODE"].map(viscode_to_month)
    frame["visit_dx_3class"] = frame["DX"].map(VISIT_DX_MAP)
    frame["baseline_dx_3class"] = frame["DX.bl"].map(
        {"CN": "CN", "NL": "CN", "EMCI": "MCI", "LMCI": "MCI", "MCI": "MCI", "AD": "AD", "Dementia": "AD"}
    )
    frame["age_numeric"] = pd.to_numeric(frame["AGE"].replace({"": None, "NA": None}), errors="coerce")
    frame["sex_numeric"] = frame["PTGENDER"].map(normalize_sex)
    return frame


def discover_segmentations(segmentation_root: Path):
    import pandas as pd

    records: list[dict[str, str]] = []
    for path in sorted(segmentation_root.glob("*/*.synthseg.mgz")):
        scan_id = path.parent.name
        expected_name = f"{scan_id}.synthseg.mgz"
        records.append(
            {
                "scan_id": scan_id,
                "segmentation_path": str(path),
                "filename_matches_parent": str(path.name == expected_name),
            }
        )
    if not records:
        raise FileNotFoundError(f"No SynthSeg MGZ files found under {segmentation_root}")
    frame = pd.DataFrame(records)
    frame[["RID", "VISCODE"]] = frame["scan_id"].str.rsplit("_", n=1, expand=True)
    return frame.sort_values(["RID", "VISCODE", "scan_id"], kind="stable").reset_index(drop=True)


def structure_label_counts(segmentation_path: Path, structures: Sequence[StructureSpec]) -> dict[str, int]:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(segmentation_path))
    data = np.asanyarray(image.dataobj)
    return {f"label_{spec.label}_voxels": int((data == spec.label).sum()) for spec in structures}


def scan_has_requested_labels(row: dict[str, Any], structures: Sequence[StructureSpec]) -> tuple[bool, dict[str, int]]:
    counts = structure_label_counts(Path(row["segmentation_path"]), structures)
    return all(counts[f"label_{spec.label}_voxels"] > 0 for spec in structures), counts


def choose_pilot_rows(frame, structures: Sequence[StructureSpec], diagnoses: Sequence[str], count: int, seed: int):
    """Choose distinct, label-present subjects while retaining both diagnoses if possible."""

    if count < 1:
        return frame
    candidates = (
        frame.sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable")
        .groupby("RID", sort=False, as_index=False)
        .first()
    )
    rng = random.Random(seed)
    selected_indices: list[int] = []
    selected_counts: dict[int, dict[str, int]] = {}

    def choose_from(indices: list[int]) -> None:
        rng.shuffle(indices)
        for index in indices:
            if index in selected_indices:
                continue
            row = candidates.loc[index].to_dict()
            try:
                valid, counts = scan_has_requested_labels(row, structures)
            except Exception as exc:
                print(f"Skipping unreadable pilot candidate {row['scan_id']}: {exc}")
                continue
            if valid:
                selected_indices.append(index)
                selected_counts[index] = counts
                return

    # This makes a three-subject CN/AD pilot informative rather than accidentally one-sided.
    if count >= 2:
        for diagnosis in diagnoses:
            if len(selected_indices) >= count:
                break
            choose_from(list(candidates.index[candidates["visit_dx_3class"].eq(diagnosis)]))
    while len(selected_indices) < count:
        before = len(selected_indices)
        choose_from(list(candidates.index))
        if len(selected_indices) == before:
            break

    if len(selected_indices) != count:
        raise RuntimeError(
            f"Could only find {len(selected_indices)} distinct pilot scans containing all requested labels; "
            f"requested {count}."
        )
    selected = candidates.loc[selected_indices].copy()
    for index, counts in selected_counts.items():
        for key, value in counts.items():
            selected.loc[index, key] = value
    return selected.sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable").reset_index(drop=True)


def run_stage_manifest(args: argparse.Namespace) -> None:
    import pandas as pd

    print_header("Stage: SynthSeg manifest")
    output_root = Path(args.output_root).resolve()
    segmentation_root = Path(args.segmentation_root).resolve()
    clinical_csv = Path(args.clinical_csv).resolve()
    structures = parse_structures(args.structures)
    diagnoses = parse_diagnoses(args.diagnoses)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_dir(output_root).mkdir(parents=True, exist_ok=True)

    clinical = read_clinical(clinical_csv)
    segments = discover_segmentations(segmentation_root)
    merged = segments.merge(clinical, on=["scan_id", "RID", "VISCODE"], how="left", indicator=True)
    matched = merged[merged["_merge"].eq("both")].copy()
    unmatched_segments = merged[merged["_merge"].ne("both")][["scan_id", "RID", "VISCODE", "segmentation_path"]].copy()

    clinical_keys = set(segments["scan_id"])
    unmatched_clinical = clinical[~clinical["scan_id"].isin(clinical_keys)].copy()
    matched["eligible_diagnosis"] = matched["visit_dx_3class"].isin(diagnoses)
    matched["strict_subject_no_mci"] = ~matched.groupby("RID")["visit_dx_3class"].transform(
        lambda values: values.fillna("").eq("MCI").any()
    )
    eligible = matched[matched["eligible_diagnosis"]].copy()
    if args.strict_no_mci_subjects:
        eligible = eligible[eligible["strict_subject_no_mci"]].copy()
    eligible = eligible.sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable").reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No scans remain after the diagnosis and clinical filters.")

    if args.sample_count == 0:
        selected = eligible.copy()
        selection_mode = "all_eligible_scans"
    else:
        selected = choose_pilot_rows(eligible, structures, diagnoses, args.sample_count, args.seed)
        selection_mode = "distinct_subject_pilot_with_requested_labels"

    selected["requested_structures"] = "|".join(spec.name for spec in structures)
    selected["requested_diagnoses"] = "|".join(diagnoses)
    selected["mesh_stem"] = selected["scan_id"]
    save_dataframe(eligible, manifests_dir(output_root) / "eligible_scans.csv")
    save_dataframe(selected, selected_manifest_path(output_root))
    save_dataframe(unmatched_segments, manifests_dir(output_root) / "unmatched_segmentations.csv")
    save_dataframe(unmatched_clinical, manifests_dir(output_root) / "unmatched_clinical.csv")

    summary = {
        "segmentation_root": str(segmentation_root),
        "clinical_csv": str(clinical_csv),
        "total_segmentations": int(len(segments)),
        "clinical_rows": int(len(clinical)),
        "matched_scans": int(len(matched)),
        "unmatched_segmentations": int(len(unmatched_segments)),
        "unmatched_clinical_rows": int(len(unmatched_clinical)),
        "eligible_scans": int(len(eligible)),
        "selected_scans": int(len(selected)),
        "selected_subjects": int(selected["RID"].nunique()),
        "selection_mode": selection_mode,
        "sample_count": int(args.sample_count),
        "seed": int(args.seed),
        "structures": [asdict(spec) for spec in structures],
        "diagnoses": diagnoses,
        "strict_no_mci_subjects": bool(args.strict_no_mci_subjects),
        "selected_diagnosis_counts": selected["visit_dx_3class"].value_counts().to_dict(),
    }
    write_json(reports_dir(output_root) / "manifest_summary.json", summary)
    write_json(
        output_root / "run_configuration.json",
        {
            **summary,
            "minimal_smoothing_defaults": {
                "gaussian_sigma_voxels": float(args.gaussian_sigma),
                "closing_iterations": int(args.closing_iterations),
                "fill_holes": bool(args.fill_holes),
                "crop_margin_voxels": int(args.crop_margin),
                "component_policy": args.component_policy,
                "watertight_policy": "adaptive_minimal_voxel_repair_required",
            },
        },
    )
    print(json.dumps(summary, indent=2))


def component_summary(mask) -> tuple[int, list[int], Any]:
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return 0, [], mask.astype(bool)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return int(count), [int(value) for value in sizes], labels


def largest_component(mask):
    import numpy as np

    count, sizes, labels = component_summary(mask)
    if count == 0:
        return mask.astype(bool), count, sizes
    return labels == (int(np.argmax(sizes)) + 1), count, sizes


def crop_and_pad(mask, margin: int):
    import numpy as np

    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Requested SynthSeg label is empty.")
    lower = np.maximum(coordinates.min(axis=0) - margin, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin + 1, mask.shape)
    cropped = mask[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
    return np.pad(cropped, 1, mode="constant", constant_values=False), lower - 1


def mesh_qc(mesh) -> dict[str, Any]:
    import numpy as np

    components = mesh.split(only_watertight=False)
    volume = abs(float(mesh.volume)) if len(mesh.faces) else math.nan
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "mesh_components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "surface_area_mm2": float(mesh.area),
        "mesh_volume_mm3": volume,
        "finite_vertices": bool(np.isfinite(mesh.vertices).all()),
    }


def mesh_from_binary_mask(
    image,
    mask,
    *,
    mode: str,
    gaussian_sigma: float,
    closing_iterations: int,
    fill_holes: bool,
    crop_margin: int,
    component_policy: str,
):
    import nibabel as nib
    import numpy as np
    import trimesh
    from scipy import ndimage
    from skimage import measure

    source_components, source_sizes, _ = component_summary(mask)
    working, origin = crop_and_pad(mask.astype(bool), crop_margin)
    post_components = source_components
    post_sizes = source_sizes

    if mode == "minimal_smooth":
        if component_policy == "largest":
            working, post_components, post_sizes = largest_component(working)
        elif component_policy == "error" and source_components != 1:
            raise ValueError(f"Expected one mask component but found {source_components}.")
        if closing_iterations > 0:
            working = ndimage.binary_closing(working, iterations=closing_iterations)
        if fill_holes:
            working = ndimage.binary_fill_holes(working)
        if closing_iterations > 0 or fill_holes:
            working, post_components, post_sizes = largest_component(working)
        field = ndimage.gaussian_filter(working.astype(np.float32), sigma=gaussian_sigma)
    elif mode == "raw":
        field = working.astype(np.float32)
    else:
        raise ValueError(f"Unsupported mesh mode: {mode}")

    if field.min() > 0.5 or field.max() < 0.5:
        raise ValueError("Marching-cubes level 0.5 is outside the cropped field range.")
    vertices, faces, _normals, _values = measure.marching_cubes(field, level=0.5)
    world_vertices = nib.affines.apply_affine(image.affine, vertices + origin)
    mesh = trimesh.Trimesh(vertices=world_vertices, faces=faces, process=False)
    metadata = {
        "source_voxel_components": source_components,
        "source_component_sizes": "|".join(str(value) for value in source_sizes),
        "post_smooth_voxel_components": post_components,
        "post_smooth_component_sizes": "|".join(str(value) for value in post_sizes),
        **mesh_qc(mesh),
    }
    return mesh, metadata


def minimally_repair_to_watertight_mesh(image, mask, task: SmoothMeshTask):
    """Use the requested light smoothing first, escalating only when necessary.

    The final escalation remains deliberately local: one voxel closing plus
    hole filling.  A last mesh-hole fill is attempted only after those
    voxel-space repairs.  A mesh that still is not watertight is rejected from
    correspondence rather than silently passed to ShapeWorks/Deformetrica.
    """
    import trimesh

    candidates: list[tuple[str, int, bool]] = [
        ("requested_minimal_smoothing", int(task.closing_iterations), bool(task.fill_holes))
    ]
    if not task.fill_holes:
        candidates.append(("minimal_voxel_hole_fill", int(task.closing_iterations), True))
    if task.closing_iterations < 1 or not task.fill_holes:
        candidates.append(("minimal_one_voxel_closing_and_fill", max(1, int(task.closing_iterations)), True))

    unique_candidates: list[tuple[str, int, bool]] = []
    seen_settings: set[tuple[int, bool]] = set()
    for candidate in candidates:
        settings = candidate[1:]
        if settings not in seen_settings:
            unique_candidates.append(candidate)
            seen_settings.add(settings)

    attempts: list[str] = []
    last_mesh = None
    last_metadata: dict[str, Any] | None = None
    for strategy, closing_iterations, fill_holes in unique_candidates:
        mesh, metadata = mesh_from_binary_mask(
            image,
            mask,
            mode="minimal_smooth",
            gaussian_sigma=task.gaussian_sigma,
            closing_iterations=closing_iterations,
            fill_holes=fill_holes,
            crop_margin=task.crop_margin,
            component_policy=task.component_policy,
        )
        attempts.append(strategy)
        metadata.update(
            {
                "watertight_strategy": strategy,
                "effective_closing_iterations": closing_iterations,
                "effective_fill_holes": fill_holes,
                "watertight_attempt_count": len(attempts),
                "watertight_attempts": "|".join(attempts),
            }
        )
        if mesh.is_watertight:
            return mesh, metadata
        last_mesh, last_metadata = mesh, metadata

    if last_mesh is None or last_metadata is None:
        raise RuntimeError("No minimal-smoothing candidate was generated.")
    repaired_mesh = last_mesh.copy()
    trimesh.repair.fill_holes(repaired_mesh)
    repaired_mesh.remove_unreferenced_vertices()
    repaired_metadata = {
        **last_metadata,
        **mesh_qc(repaired_mesh),
        "watertight_strategy": "minimal_voxel_repair_then_mesh_hole_fill",
        "watertight_attempt_count": len(attempts) + 1,
        "watertight_attempts": "|".join([*attempts, "mesh_hole_fill"]),
    }
    if repaired_mesh.is_watertight:
        return repaired_mesh, repaired_metadata
    raise ValueError(
        "Minimal smoothing could not produce a watertight mesh after voxel hole filling, "
        "one-voxel closing, and mesh-hole filling."
    )


def process_raw_mesh_task(task: RawMeshTask) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np
    import trimesh

    result: dict[str, Any] = {
        "scan_id": task.scan_id,
        "RID": task.rid,
        "VISCODE": task.viscode,
        "diagnosis": task.diagnosis,
        "structure": task.structure_name,
        "label": task.label,
        "segmentation_path": task.segmentation_path,
        "raw_ply": task.raw_path,
        "status": "started",
        "error": "",
    }
    try:
        raw_path = Path(task.raw_path)
        if raw_path.is_file() and not task.overwrite:
            raw_mesh = trimesh.load(raw_path, force="mesh", process=False)
            result.update({f"raw_{key}": value for key, value in mesh_qc(raw_mesh).items()})
            result["status"] = "skipped_existing"
            return result

        image = nib.load(task.segmentation_path)
        data = np.asanyarray(image.dataobj)
        mask = data == task.label
        voxel_volume = abs(float(np.linalg.det(image.affine[:3, :3])))
        result["mask_voxels"] = int(mask.sum())
        result["mask_volume_mm3"] = float(mask.sum() * voxel_volume)
        result["voxel_volume_mm3"] = voxel_volume
        if not mask.any():
            raise ValueError("Requested SynthSeg label is empty.")

        raw_mesh, raw_meta = mesh_from_binary_mask(
            image,
            mask,
            mode="raw",
            gaussian_sigma=0.0,
            closing_iterations=0,
            fill_holes=False,
            crop_margin=4,
            component_policy="largest",
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_mesh.export(raw_path)
        result.update({f"raw_{key}": value for key, value in raw_meta.items()})
        result["status"] = "ok"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result


def process_minimal_smooth_task(task: SmoothMeshTask) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np
    import trimesh

    result: dict[str, Any] = {
        "scan_id": task.scan_id,
        "RID": task.rid,
        "VISCODE": task.viscode,
        "diagnosis": task.diagnosis,
        "structure": task.structure_name,
        "label": task.label,
        "segmentation_path": task.segmentation_path,
        "raw_ply": task.raw_path,
        "minimal_smooth_ply": task.smooth_path,
        "gaussian_sigma_voxels": task.gaussian_sigma,
        "closing_iterations": task.closing_iterations,
        "fill_holes": task.fill_holes,
        "crop_margin_voxels": task.crop_margin,
        "component_policy": task.component_policy,
        "status": "started",
        "error": "",
    }
    try:
        raw_path = Path(task.raw_path)
        smooth_path = Path(task.smooth_path)
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw mesh is required before smoothing: {raw_path}")
        if smooth_path.is_file() and not task.overwrite:
            smooth_mesh = trimesh.load(smooth_path, force="mesh", process=False)
            if smooth_mesh.is_watertight:
                result.update({f"smooth_{key}": value for key, value in mesh_qc(smooth_mesh).items()})
                result["smooth_watertight_strategy"] = "existing_verified_watertight"
                result["status"] = "skipped_existing"
                return result
            result["replaced_existing_non_watertight_smooth"] = True

        image = nib.load(task.segmentation_path)
        data = np.asanyarray(image.dataobj)
        mask = data == task.label
        voxel_volume = abs(float(np.linalg.det(image.affine[:3, :3])))
        result["mask_voxels"] = int(mask.sum())
        result["mask_volume_mm3"] = float(mask.sum() * voxel_volume)
        result["voxel_volume_mm3"] = voxel_volume
        if not mask.any():
            raise ValueError("Requested SynthSeg label is empty.")
        smooth_mesh, smooth_meta = minimally_repair_to_watertight_mesh(image, mask, task)
        smooth_path.parent.mkdir(parents=True, exist_ok=True)
        smooth_mesh.export(smooth_path)
        result.update({f"smooth_{key}": value for key, value in smooth_meta.items()})
        result["smooth_vs_mask_volume_pct"] = (
            100.0 * (smooth_meta["mesh_volume_mm3"] - result["mask_volume_mm3"]) / result["mask_volume_mm3"]
        )
        result["status"] = "ok"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result


def run_tasks(tasks: Sequence[Any], workers: int, task_function, description: str):
    import pandas as pd

    def task_name(task) -> str:
        return f"{task.structure_name} | {task.scan_id}"

    records: list[dict[str, Any]] = []
    print(f"{description}: {len(tasks)} task(s) with {workers} worker(s).")
    if not tasks:
        return pd.DataFrame()
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            print(f"  [{index}/{len(tasks)}] processing {task_name(task)}")
            result = task_function(task)
            records.append(result)
            print(f"  [{index}/{len(tasks)}] {result['status']}: {task_name(task)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(task_function, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                records.append(result)
                print(f"  [{index}/{len(tasks)}] {result['status']}: {task_name(futures[future])}")
    frame = pd.DataFrame(records)
    return frame.sort_values(["structure", "RID", "VISCODE", "scan_id"], kind="stable").reset_index(drop=True)


def update_combined_mesh_qc(output_root: Path, structures: Sequence[StructureSpec]) -> None:
    import pandas as pd

    for spec in structures:
        raw_path = structure_root(output_root, spec.name) / "raw_mesh_qc.csv"
        smooth_path = structure_root(output_root, spec.name) / "minimal_smooth_qc.csv"
        if not raw_path.is_file():
            continue
        raw = load_dataframe(raw_path).rename(
            columns={"status": "raw_status", "error": "raw_error", "traceback": "raw_traceback"}
        )
        if smooth_path.is_file():
            smooth = load_dataframe(smooth_path)
            keep = ["scan_id", "structure", "minimal_smooth_ply", "status", "error", "traceback"]
            keep.extend(column for column in smooth.columns if column.startswith("smooth_"))
            keep.extend(
                column
                for column in ["smooth_vs_mask_volume_pct", "replaced_existing_non_watertight_smooth"]
                if column in smooth.columns
            )
            smooth = smooth.loc[:, list(dict.fromkeys(column for column in keep if column in smooth.columns))].rename(
                columns={"status": "smooth_status", "error": "smooth_error", "traceback": "smooth_traceback"}
            )
            combined = raw.merge(smooth, on=["scan_id", "structure"], how="left")
            combined["status"] = combined["smooth_status"].fillna(
                combined["raw_status"].map(
                    lambda value: "raw_ready" if value in {"ok", "skipped_existing"} else "raw_failed"
                )
            )
        else:
            combined = raw.copy()
            combined["smooth_status"] = pd.NA
            combined["status"] = combined["raw_status"].map(
                lambda value: "raw_ready" if value in {"ok", "skipped_existing"} else "raw_failed"
            )
        save_dataframe(combined, mesh_qc_path(output_root, spec.name))


def merge_lineage_stage(frame, stage_frame):
    """Merge one optional per-scan stage table into the volume lineage table."""
    if stage_frame is None or stage_frame.empty:
        return frame
    keys = ["scan_id", "structure"]
    if not set(keys).issubset(stage_frame.columns):
        return frame
    extra = stage_frame.loc[:, keys + [column for column in stage_frame.columns if column not in keys]].copy()
    extra = extra.drop_duplicates(keys, keep="last")
    overlapping = [column for column in extra.columns if column not in keys and column in frame.columns]
    if overlapping:
        extra = extra.drop(columns=overlapping)
    return frame.merge(extra, on=keys, how="left")


def write_volume_lineage(output_root: Path, structures: Sequence[StructureSpec]) -> None:
    """Write one reversible volume/coordinate provenance table for every requested mesh."""
    import pandas as pd

    all_rows: list[Any] = []
    for spec in structures:
        raw_path = structure_root(output_root, spec.name) / "raw_mesh_qc.csv"
        if not raw_path.is_file():
            continue
        raw = load_dataframe(raw_path).copy()
        if raw.empty:
            continue
        raw = raw.rename(columns={"status": "raw_status", "error": "raw_error"})
        base_columns = [
            "scan_id",
            "structure",
            "RID",
            "VISCODE",
            "diagnosis",
            "label",
            "segmentation_path",
            "raw_ply",
            "raw_status",
            "raw_error",
        ]
        base_columns.extend(column for column in raw.columns if column.startswith("raw_"))
        base = raw.loc[:, list(dict.fromkeys(column for column in base_columns if column in raw.columns))].copy()

        smooth_path = structure_root(output_root, spec.name) / "minimal_smooth_qc.csv"
        if smooth_path.is_file():
            smooth = load_dataframe(smooth_path).copy()
            if not smooth.empty:
                smooth = smooth.rename(columns={"status": "minimal_smooth_status", "error": "minimal_smooth_error"})
                smooth_columns = [
                    "scan_id",
                    "structure",
                    "minimal_smooth_ply",
                    "minimal_smooth_status",
                    "minimal_smooth_error",
                    "mask_voxels",
                    "mask_volume_mm3",
                    "voxel_volume_mm3",
                    "gaussian_sigma_voxels",
                    "closing_iterations",
                    "fill_holes",
                    "crop_margin_voxels",
                    "component_policy",
                    "smooth_vs_mask_volume_pct",
                ]
                smooth_columns.extend(column for column in smooth.columns if column.startswith("smooth_"))
                base = merge_lineage_stage(
                    base, smooth.loc[:, list(dict.fromkeys(column for column in smooth_columns if column in smooth.columns))]
                )

        for path in [prepare_details_path(output_root, spec.name), rigid_details_path(output_root, spec.name)]:
            if path.is_file():
                base = merge_lineage_stage(base, load_dataframe(path))
        rescale_path = correspondence_root(output_root, spec.name) / "rescale_details.csv"
        if rescale_path.is_file():
            base = merge_lineage_stage(base, load_dataframe(rescale_path))
        all_rows.append(base)

    lineage = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    if not lineage.empty:
        lineage = lineage.sort_values(["structure", "RID", "VISCODE", "scan_id"], kind="stable")
    save_dataframe(lineage, volume_lineage_path(output_root))


def run_stage_raw(args: argparse.Namespace) -> None:
    import pandas as pd

    print_header("Stage: Raw meshes from SynthSeg labels")
    output_root = Path(args.output_root).resolve()
    structures = parse_structures(args.structures)
    manifest = load_dataframe(selected_manifest_path(output_root))
    tasks: list[RawMeshTask] = []
    for spec in structures:
        for row in manifest.to_dict("records"):
            scan_id = str(row["scan_id"])
            tasks.append(
                RawMeshTask(
                    scan_id=scan_id,
                    rid=str(row["RID"]),
                    viscode=str(row["VISCODE"]),
                    diagnosis=str(row["visit_dx_3class"]),
                    segmentation_path=str(row["segmentation_path"]),
                    structure_name=spec.name,
                    label=spec.label,
                    raw_path=str(raw_ply_dir(output_root, spec.name) / f"{scan_id}.ply"),
                    overwrite=bool(args.overwrite),
                )
            )
    qc = run_tasks(tasks, max(1, int(args.workers)), process_raw_mesh_task, "Creating raw meshes")
    for spec in structures:
        structure_qc = qc[qc["structure"].eq(spec.name)].copy()
        save_dataframe(structure_qc, structure_root(output_root, spec.name) / "raw_mesh_qc.csv")
    save_dataframe(qc, reports_dir(output_root) / "raw_mesh_qc_all_structures.csv")
    update_combined_mesh_qc(output_root, structures)
    write_volume_lineage(output_root, structures)
    summary = {
        "tasks": int(len(qc)),
        "status_counts": qc["status"].value_counts(dropna=False).to_dict(),
        "structures": {
            spec.name: {
                "tasks": int((qc["structure"] == spec.name).sum()),
                "raw_saved": int(
                    qc[qc["structure"].eq(spec.name)]["status"].isin(["ok", "skipped_existing"]).sum()
                ),
                "raw_ply_dir": str(raw_ply_dir(output_root, spec.name)),
            }
            for spec in structures
        },
    }
    write_json(reports_dir(output_root) / "raw_mesh_summary.json", summary)
    print(json.dumps(summary, indent=2))


def run_stage_minimal_smooth(args: argparse.Namespace) -> None:
    """Create watertight minimal-smooth meshes only after the raw stage."""
    import pandas as pd

    print_header("Stage: Minimal-smooth watertight meshes")
    output_root = Path(args.output_root).resolve()
    structures = parse_structures(args.structures)
    all_qc: list[Any] = []
    for spec in structures:
        raw_qc_path = structure_root(output_root, spec.name) / "raw_mesh_qc.csv"
        if not raw_qc_path.is_file():
            raise FileNotFoundError(
                f"Raw QC is required before minimal smoothing for {spec.name}: {raw_qc_path}"
            )
        raw_qc = load_dataframe(raw_qc_path)
        tasks: list[SmoothMeshTask] = []
        blocked: list[dict[str, Any]] = []
        for row in raw_qc.to_dict("records"):
            scan_id = str(row["scan_id"])
            smooth_path = smooth_ply_dir(output_root, spec.name) / f"{scan_id}.ply"
            if str(row.get("status", "")) not in SUCCESS_STATUSES:
                blocked.append(
                    {
                        "scan_id": scan_id,
                        "RID": str(row.get("RID", "")),
                        "VISCODE": str(row.get("VISCODE", "")),
                        "diagnosis": str(row.get("diagnosis", "")),
                        "structure": spec.name,
                        "label": spec.label,
                        "segmentation_path": str(row.get("segmentation_path", "")),
                        "raw_ply": str(row.get("raw_ply", "")),
                        "minimal_smooth_ply": str(smooth_path),
                        "status": "blocked_by_raw_failure",
                        "error": f"Raw mesh status is {row.get('status')}: {row.get('error', '')}",
                    }
                )
                continue
            tasks.append(
                SmoothMeshTask(
                    scan_id=scan_id,
                    rid=str(row["RID"]),
                    viscode=str(row["VISCODE"]),
                    diagnosis=str(row["diagnosis"]),
                    segmentation_path=str(row["segmentation_path"]),
                    structure_name=spec.name,
                    label=spec.label,
                    raw_path=str(row["raw_ply"]),
                    smooth_path=str(smooth_path),
                    gaussian_sigma=float(args.gaussian_sigma),
                    closing_iterations=int(args.closing_iterations),
                    fill_holes=bool(args.fill_holes),
                    crop_margin=int(args.crop_margin),
                    component_policy=args.component_policy,
                    overwrite=bool(args.overwrite),
                )
            )
        task_qc = run_tasks(
            tasks,
            max(1, int(args.workers)),
            process_minimal_smooth_task,
            f"Creating minimal-smooth watertight {spec.display_name} meshes",
        )
        frames = [frame for frame in [task_qc, pd.DataFrame(blocked)] if not frame.empty]
        structure_qc = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if not structure_qc.empty:
            structure_qc = structure_qc.sort_values(
                ["structure", "RID", "VISCODE", "scan_id"], kind="stable"
            ).reset_index(drop=True)
        save_dataframe(structure_qc, structure_root(output_root, spec.name) / "minimal_smooth_qc.csv")
        all_qc.append(structure_qc)

    frames = [frame for frame in all_qc if not frame.empty]
    qc = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    save_dataframe(qc, reports_dir(output_root) / "minimal_smooth_qc_all_structures.csv")
    update_combined_mesh_qc(output_root, structures)
    write_volume_lineage(output_root, structures)
    summary = {
        "tasks": int(len(qc)),
        "status_counts": qc["status"].value_counts(dropna=False).to_dict() if not qc.empty else {},
        "structures": {
            spec.name: {
                "tasks": int((qc["structure"] == spec.name).sum()) if not qc.empty else 0,
                "watertight_saved": int(
                    qc[qc["structure"].eq(spec.name)]["status"].isin(SUCCESS_STATUSES).sum()
                )
                if not qc.empty
                else 0,
                "minimal_smooth_ply_dir": str(smooth_ply_dir(output_root, spec.name)),
            }
            for spec in structures
        },
    }
    write_json(reports_dir(output_root) / "minimal_smooth_summary.json", summary)
    print(json.dumps(summary, indent=2))


def true_values(series):
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def run_stage_quality_gate(args: argparse.Namespace, phase: str) -> None:
    """Require a complete usable cohort unless the user explicitly allows partial output."""
    import pandas as pd

    if phase not in {"raw", "minimal_smooth"}:
        raise ValueError(f"Unsupported quality-gate phase: {phase}")
    print_header(f"Stage: {phase.replace('_', ' ')} quality gate")
    output_root = Path(args.output_root).resolve()
    structures = parse_structures(args.structures)
    qc_filename = "raw_mesh_qc.csv" if phase == "raw" else "minimal_smooth_qc.csv"
    summary: dict[str, Any] = {
        "phase": phase,
        "allow_partial_cohort": bool(args.allow_partial_cohort),
        "passed": True,
        "structures": {},
    }
    blocking_issues: list[str] = []
    for spec in structures:
        qc_path = structure_root(output_root, spec.name) / qc_filename
        if not qc_path.is_file():
            summary["passed"] = False
            summary["structures"][spec.name] = {"successes": 0, "failures": 1, "issues": ["missing_qc"]}
            blocking_issues.append(f"{spec.name}: missing {qc_filename}")
            continue
        qc = load_dataframe(qc_path)
        if qc.empty or "status" not in qc.columns:
            successes = pd.DataFrame()
            failures = qc
        else:
            successes = qc[qc["status"].isin(SUCCESS_STATUSES)].copy()
            failures = qc[~qc["status"].isin(SUCCESS_STATUSES)].copy()
        issues: list[str] = []
        if successes.empty:
            issues.append("no_usable_meshes")
        if not failures.empty:
            issues.append("task_failures")
        if phase == "minimal_smooth":
            if "smooth_watertight" not in successes.columns:
                issues.append("missing_watertight_qc")
            else:
                non_watertight = successes[~true_values(successes["smooth_watertight"])]
                if not non_watertight.empty:
                    issues.append("non_watertight_meshes")
        summary["structures"][spec.name] = {
            "attempted": int(len(qc)),
            "successes": int(len(successes)),
            "failures": int(len(failures)),
            "issues": issues,
        }
        if issues:
            summary["passed"] = False
            blocking_issues.append(f"{spec.name}: {', '.join(issues)}")
    write_json(reports_dir(output_root) / f"{phase}_gate_summary.json", summary)
    print(json.dumps(summary, indent=2))

    irrecoverable = any(
        "no_usable_meshes" in details["issues"] or "missing_qc" in details["issues"]
        for details in summary["structures"].values()
    )
    if irrecoverable or (not summary["passed"] and not args.allow_partial_cohort):
        raise SystemExit(
            f"{phase} quality gate stopped the pipeline: {'; '.join(blocking_issues)}. "
            "Inspect the QC reports; generate the notebook for visual review before continuing."
        )
    if not summary["passed"]:
        print("WARNING: Continuing with only the successful meshes because --allow-partial-cohort was set.")


def run_stage_raw_qc(args: argparse.Namespace) -> None:
    run_stage_quality_gate(args, "raw")


def run_stage_minimal_smooth_qc(args: argparse.Namespace) -> None:
    run_stage_quality_gate(args, "minimal_smooth")


def load_mesh(path: Path):
    import trimesh

    return trimesh.load(path, force="mesh", process=False)


def list_subject_mesh_paths(directory: Path, extension: str = "ply") -> list[Path]:
    return sorted(path for path in directory.glob(f"*.{extension}") if not path.stem.startswith("template_"))


def successful_smooth_mesh_paths(output_root: Path, spec: StructureSpec) -> list[Path]:
    """Return only the current selection's verified, watertight smooth meshes.

    Reading paths from the current QC table prevents old files in a reused
    output directory from leaking into a new pilot or full run.
    """
    qc_path = structure_root(output_root, spec.name) / "minimal_smooth_qc.csv"
    if not qc_path.is_file():
        raise FileNotFoundError(f"Minimal-smooth QC is missing for {spec.name}: {qc_path}")
    qc = load_dataframe(qc_path)
    successful = qc[qc["status"].isin(SUCCESS_STATUSES)].copy()
    if successful.empty:
        raise RuntimeError(f"No successful minimal-smooth meshes for {spec.name}.")
    if "smooth_watertight" not in successful.columns or not true_values(successful["smooth_watertight"]).all():
        raise RuntimeError(f"Minimal-smooth QC contains a non-watertight {spec.name} mesh.")
    paths = [Path(path) for path in successful["minimal_smooth_ply"].astype(str)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Verified minimal-smooth mesh files are missing for {spec.name}: {missing[:10]}")
    return sorted(paths)


def mesh_max_dimension(mesh) -> float:
    return float((mesh.bounds[1] - mesh.bounds[0]).max())


def run_stage_prepare(args: argparse.Namespace) -> None:
    import pandas as pd

    print_header("Stage: Center and globally scale minimal-smooth meshes")
    output_root = Path(args.output_root).resolve()
    structures = parse_structures(args.structures)
    summary: dict[str, Any] = {"structures": {}}
    for spec in structures:
        source_paths = successful_smooth_mesh_paths(output_root, spec)
        source_meshes = [(path, load_mesh(path)) for path in source_paths]
        non_watertight = [path.name for path, mesh in source_meshes if not mesh.is_watertight]
        if non_watertight:
            raise RuntimeError(
                f"Refusing to prepare non-watertight minimal-smooth {spec.name} meshes: {non_watertight[:10]}"
            )
        dimensions = [mesh_max_dimension(mesh) for _path, mesh in source_meshes]
        maximum = max(dimensions)
        scale = 1.0 / (maximum * GLOBAL_SCALE_BUFFER)
        destination = prepared_ply_dir(output_root, spec.name)
        destination.mkdir(parents=True, exist_ok=True)
        details: list[dict[str, Any]] = []
        for index, (source_path, mesh) in enumerate(source_meshes, start=1):
            print(f"  [{index}/{len(source_meshes)}] preparing {spec.name} | {source_path.stem}")
            source_volume_mm3 = abs(float(mesh.volume))
            translation = -mesh.bounding_box.centroid
            mesh.apply_translation(translation)
            mesh.apply_scale(scale)
            output_path = destination / source_path.name
            mesh.export(output_path)
            prepared_volume_scaled = abs(float(mesh.volume))
            details.append(
                {
                    "scan_id": source_path.stem,
                    "structure": spec.name,
                    "prepared_ply": str(output_path),
                    "centering_translation_x_mm": float(translation[0]),
                    "centering_translation_y_mm": float(translation[1]),
                    "centering_translation_z_mm": float(translation[2]),
                    "minimal_smooth_volume_mm3": source_volume_mm3,
                    "global_scale_factor": float(scale),
                    "global_volume_scale_factor": float(scale**3),
                    "distance_unscale_factor": float(1.0 / scale),
                    "volume_unscale_factor": float((1.0 / scale) ** 3),
                    "prepared_volume_scaled": prepared_volume_scaled,
                    "prepared_volume_mm3_recovered": prepared_volume_scaled * float((1.0 / scale) ** 3),
                }
            )
        metadata = {
            "structure": asdict(spec),
            "input_count": len(source_paths),
            "dimension_min_mm": float(min(dimensions)),
            "dimension_max_mm": float(maximum),
            "global_scale_factor": float(scale),
            "distance_unscale_factor": float(1.0 / scale),
            "volume_unscale_factor": float((1.0 / scale) ** 3),
            "source_dir": str(smooth_ply_dir(output_root, spec.name)),
            "prepared_dir": str(destination),
        }
        write_json(metadata_path(output_root, spec.name), metadata)
        save_dataframe(pd.DataFrame(details), prepare_details_path(output_root, spec.name))
        summary["structures"][spec.name] = metadata
    write_json(reports_dir(output_root) / "prepare_summary.json", summary)
    write_volume_lineage(output_root, structures)
    print(json.dumps(summary, indent=2))


def run_stage_rigid(args: argparse.Namespace) -> None:
    import pandas as pd
    import shapeworks as sw

    print_header("Stage: ShapeWorks rigid registration")
    output_root = Path(args.output_root).resolve()
    structures = parse_structures(args.structures)
    summary: dict[str, Any] = {"structures": {}}
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        for spec in structures:
            source_paths = [
                prepared_ply_dir(output_root, spec.name) / path.name
                for path in successful_smooth_mesh_paths(output_root, spec)
            ]
            missing = [str(path) for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Prepared meshes are missing for {spec.name}: {missing[:10]}")
            if not source_paths:
                raise RuntimeError(f"No prepared meshes for {spec.name}.")
            items: list[tuple[str, Any]] = []
            for source_path in source_paths:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    items.append((source_path.stem, sw.Mesh(str(source_path))))
            reference_index = sw.find_reference_mesh_index([mesh for _, mesh in items])
            reference_name, reference_mesh = items[reference_index]
            destination = rigid_ply_dir(output_root, spec.name)
            destination.mkdir(parents=True, exist_ok=True)
            details: list[dict[str, Any]] = []
            for index, (name, mesh) in enumerate(items, start=1):
                print(f"  [{index}/{len(items)}] rigid registration {spec.name} | {name}")
                prepared_mesh = load_mesh(prepared_ply_dir(output_root, spec.name) / f"{name}.ply")
                prepared_volume = abs(float(prepared_mesh.volume))
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    transform = mesh.createTransform(
                        reference_mesh, sw.Mesh.AlignmentType.Rigid, int(args.shapeworks_iterations)
                    )
                    mesh.applyTransform(transform)
                    output_path = destination / f"{name}.ply"
                    mesh.write(str(output_path))
                rigid_volume = abs(float(load_mesh(output_path).volume))
                details.append(
                    {
                        "scan_id": name,
                        "structure": spec.name,
                        "rigid_ply": str(output_path),
                        "rigid_linear_scale_factor": 1.0,
                        "rigid_volume_scaled": rigid_volume,
                        "rigid_vs_prepared_volume_error_pct": (
                            100.0 * (rigid_volume - prepared_volume) / prepared_volume
                        ),
                    }
                )
            payload = {
                "input_count": len(items),
                "registered": len(items),
                "reference_medoid": reference_name,
                "iterations": int(args.shapeworks_iterations),
                "output_dir": str(destination),
            }
            write_json(structure_root(output_root, spec.name) / "rigid_metadata.json", payload)
            save_dataframe(pd.DataFrame(details), rigid_details_path(output_root, spec.name))
            summary["structures"][spec.name] = payload
    write_json(reports_dir(output_root) / "rigid_summary.json", summary)
    write_volume_lineage(output_root, structures)
    print(json.dumps(summary, indent=2))


def write_legacy_vtk(vertices, faces, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\nSynthSeg mesh\nASCII\nDATASET POLYDATA\n")
        handle.write(f"POINTS {len(vertices)} float\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        handle.write(f"\nPOLYGONS {len(faces)} {len(faces) * 4}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def read_legacy_vtk(path: Path):
    import numpy as np

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    reading_points = False
    reading_polygons = False
    points_remaining = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("POINTS"):
            points_remaining = int(line.split()[1])
            reading_points, reading_polygons = True, False
            continue
        if line.startswith("POLYGONS"):
            reading_points, reading_polygons = False, True
            continue
        if line.startswith(("CELL", "POINT_DATA", "METADATA")):
            reading_points, reading_polygons = False, False
            continue
        if reading_points and points_remaining:
            parts = line.split()
            if len(parts) >= 3:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                points_remaining -= 1
            continue
        if reading_polygons:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "3":
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    if not vertices or not faces:
        raise ValueError(f"Could not parse VTK mesh: {path}")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def subject_id_from_reconstruction(path: Path) -> str:
    match = RECON_SUBJECT_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not determine reconstruction subject from {path.name}")
    return match.group(1)


def run_deformetrica_atlas(vtk_dir: Path, template_file: Path, output_dir: Path, iterations: int, gpu_mode: str) -> None:
    import deformetrica as dfca

    vtk_files = sorted(vtk_dir.glob("*.vtk"))
    if not vtk_files:
        raise RuntimeError(f"No VTK input meshes found in {vtk_dir}")
    subject_ids = [path.stem for path in vtk_files]
    dataset_filenames = [[{"structure": str(path)}] for path in vtk_files]
    template_specifications = {
        "structure": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": "keops",
            "kernel_width": 0.03,
            "noise_std": 0.01,
            "filename": str(template_file),
            "attachment_type": "varifold",
        }
    }
    estimator_options = {
        "optimization_method_type": "GradientAscent",
        "max_line_search_iterations": 10,
        "gpu_mode": gpu_mode,
        "max_iterations": int(iterations),
        "initial_step_size": 0.5,
        "convergence_tolerance": 1e-6,
        "save_every_n_iters": max(10, int(iterations) // 5),
    }
    model_options = {
        "deformation_kernel_type": "keops",
        "deformation_kernel_width": 0.05,
        "number_of_timepoints": 25,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    deformetrica = dfca.Deformetrica(output_dir=str(output_dir), verbosity="ERROR")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        deformetrica.estimate_deterministic_atlas(
            template_specifications,
            {"dataset_filenames": dataset_filenames, "subject_ids": subject_ids},
            estimator_options=estimator_options,
            model_options=model_options,
        )


def export_old_style_range_objects(
    output_root: Path, spec: StructureSpec, subject_paths: Sequence[Path]
) -> dict[str, Any]:
    """Match the legacy pipeline's one-scalar affine OBJ mapping to [-0.9, 0.9]."""
    import numpy as np

    template_path = correspondence_root(output_root, spec.name) / "template.ply"
    source_paths = [*subject_paths, *([template_path] if template_path.is_file() else [])]
    if not source_paths:
        raise RuntimeError(f"No final PLY files are available for fixed-range export: {spec.name}")
    global_min = math.inf
    global_max = -math.inf
    for path in source_paths:
        vertices = np.asarray(load_mesh(path).vertices)
        global_min = min(global_min, float(vertices.min()))
        global_max = max(global_max, float(vertices.max()))
    if not math.isfinite(global_min) or not math.isfinite(global_max) or global_max <= global_min:
        raise RuntimeError(f"Could not compute a valid global coordinate range for {spec.name}.")
    scale = (TARGET_RANGE_MAX - TARGET_RANGE_MIN) / (global_max - global_min)
    destination = scaled_obj_dir(output_root, spec.name)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source_paths:
        mesh = load_mesh(path)
        mesh.vertices = (mesh.vertices - global_min) * scale + TARGET_RANGE_MIN
        mesh.export(destination / path.with_suffix(".obj").name)
    metadata = {
        "source_coordinate_system": "prepared_correspondence_scale",
        "global_min": float(global_min),
        "global_max": float(global_max),
        "target_min": float(TARGET_RANGE_MIN),
        "target_max": float(TARGET_RANGE_MAX),
        "scaling_factor": float(scale),
        "translation_before_scale": float(-global_min),
        "formula": "scaled = (source - global_min) * scaling_factor + target_min",
        "source_ply_count": len(source_paths),
        "output_dir": str(destination),
    }
    write_json(destination / "scale_info.json", metadata)
    return metadata


def run_stage_correspond(args: argparse.Namespace) -> None:
    import pandas as pd
    import trimesh

    print_header("Stage: Deformetrica point correspondence")
    output_root = Path(args.output_root).resolve()
    structures = parse_structures(args.structures)
    summary: dict[str, Any] = {"structures": {}}
    for spec in structures:
        print_header(f"Correspondence: {spec.display_name}")
        rigid_paths = [
            rigid_ply_dir(output_root, spec.name) / path.name
            for path in successful_smooth_mesh_paths(output_root, spec)
        ]
        missing_rigid = [str(path) for path in rigid_paths if not path.is_file()]
        if missing_rigid:
            raise FileNotFoundError(f"Rigid meshes are missing for {spec.name}: {missing_rigid[:10]}")
        if not rigid_paths:
            raise RuntimeError(f"No rigidly aligned meshes for {spec.name}.")
        vtk_input = correspondence_vtk_dir(output_root, spec.name)
        deforms = deformetrica_dir(output_root, spec.name)
        final_ply = final_ply_dir(output_root, spec.name)
        final_ply_mm = final_ply_mm_dir(output_root, spec.name)
        final_vtk = final_vtk_dir(output_root, spec.name)
        vtk_input.mkdir(parents=True, exist_ok=True)
        final_ply.mkdir(parents=True, exist_ok=True)
        final_ply_mm.mkdir(parents=True, exist_ok=True)
        final_vtk.mkdir(parents=True, exist_ok=True)
        volumes: dict[str, float] = {}
        filenames: dict[str, str] = {}
        for index, path in enumerate(rigid_paths, start=1):
            print(f"  [{index}/{len(rigid_paths)}] exporting Deformetrica input {spec.name} | {path.stem}")
            mesh = trimesh.load(path, force="mesh", process=False)
            volumes[path.stem] = abs(float(mesh.volume))
            filenames[path.stem] = path.name
            write_legacy_vtk(mesh.vertices, mesh.faces, vtk_input / f"{path.stem}.vtk")
        smooth_qc = load_dataframe(structure_root(output_root, spec.name) / "minimal_smooth_qc.csv")
        smooth_volume_mm3 = {
            str(row["scan_id"]): float(row["smooth_mesh_volume_mm3"])
            for row in smooth_qc[smooth_qc["status"].isin(SUCCESS_STATUSES)].to_dict("records")
        }
        scale_metadata = json.loads(metadata_path(output_root, spec.name).read_text())
        global_scale = float(scale_metadata["global_scale_factor"])
        distance_unscale = float(scale_metadata["distance_unscale_factor"])
        volume_unscale = float(scale_metadata["volume_unscale_factor"])
        rigid_metadata = json.loads((structure_root(output_root, spec.name) / "rigid_metadata.json").read_text())
        template_input = vtk_input / f"{rigid_metadata['reference_medoid']}.vtk"
        if not template_input.is_file():
            raise RuntimeError(f"Rigid-registration medoid is missing: {template_input}")
        reconstructions = sorted(deforms.glob("DeterministicAtlas__Reconstruction__*.vtk"))
        template_output = deforms / "DeterministicAtlas__EstimatedParameters__Template_structure.vtk"
        expected_stems = {path.stem for path in rigid_paths}
        existing_stems = {subject_id_from_reconstruction(path) for path in reconstructions}
        atlas_action = "estimated"
        effective_iterations = int(args.deformetrica_iterations)
        if reconstructions or template_output.is_file():
            if existing_stems != expected_stems or not template_output.is_file():
                raise RuntimeError(
                    f"Existing Deformetrica output for {spec.name} does not match the current meshes. "
                    "Use a new output directory instead of mixing pilot and full runs."
                )
            atlas_action = "reused_existing"
            previous_summary = correspondence_root(output_root, spec.name) / "summary.json"
            if previous_summary.is_file():
                effective_iterations = int(
                    json.loads(previous_summary.read_text()).get("deformetrica_iterations", effective_iterations)
                )
        else:
            print(f"  estimating Deformetrica atlas for {spec.name} ({int(args.deformetrica_iterations)} iterations)")
            run_deformetrica_atlas(
                vtk_input, template_input, deforms, int(args.deformetrica_iterations), args.deformetrica_gpu_mode
            )
            reconstructions = sorted(deforms.glob("DeterministicAtlas__Reconstruction__*.vtk"))
        if not reconstructions or not template_output.is_file():
            raise RuntimeError(f"Deformetrica did not create the expected outputs for {spec.name}.")

        template_vertices, template_faces = read_legacy_vtk(template_output)
        write_legacy_vtk(template_vertices, template_faces, correspondence_root(output_root, spec.name) / "template.vtk")
        trimesh.Trimesh(vertices=template_vertices, faces=template_faces, process=False).export(
            correspondence_root(output_root, spec.name) / "template.ply"
        )
        rescale_rows: list[dict[str, Any]] = []
        for index, reconstruction in enumerate(reconstructions, start=1):
            stem = subject_id_from_reconstruction(reconstruction)
            if stem not in volumes:
                raise RuntimeError(f"Unknown Deformetrica reconstruction subject: {stem}")
            if stem not in smooth_volume_mm3:
                raise RuntimeError(f"Missing minimal-smooth physical volume for {spec.name} | {stem}")
            print(f"  [{index}/{len(reconstructions)}] volume-preserving correspondence {spec.name} | {stem}")
            vertices, faces = read_legacy_vtk(reconstruction)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            reconstructed_volume = abs(float(mesh.volume))
            if reconstructed_volume <= 0:
                raise RuntimeError(f"Non-positive reconstruction volume for {stem}")
            scale_factor = (volumes[stem] / reconstructed_volume) ** (1.0 / 3.0)
            mesh.vertices *= scale_factor
            write_legacy_vtk(mesh.vertices, mesh.faces, final_vtk / f"{stem}.vtk")
            final_scaled_path = final_ply / filenames[stem]
            mesh.export(final_scaled_path)
            physical_mesh = mesh.copy()
            physical_mesh.vertices *= distance_unscale
            final_mm_path = final_ply_mm / filenames[stem]
            physical_mesh.export(final_mm_path)
            final_scaled_volume = abs(float(mesh.volume))
            final_mm_volume = abs(float(physical_mesh.volume))
            rescale_rows.append(
                {
                    "scan_id": stem,
                    "structure": spec.name,
                    "minimal_smooth_volume_mm3": smooth_volume_mm3[stem],
                    "rigid_input_volume_scaled": volumes[stem],
                    "deformetrica_volume_scaled": reconstructed_volume,
                    "rescale_factor": scale_factor,
                    "correspondence_rescale_factor": scale_factor,
                    "global_scale_factor": global_scale,
                    "distance_unscale_factor": distance_unscale,
                    "volume_unscale_factor": volume_unscale,
                    "final_volume_scaled": final_scaled_volume,
                    "final_volume_mm3": final_mm_volume,
                    "final_vs_smooth_volume_error_pct": (
                        100.0 * (final_mm_volume - smooth_volume_mm3[stem]) / smooth_volume_mm3[stem]
                    ),
                    "final_total_linear_scale_to_mm": scale_factor * distance_unscale,
                    "final_ply": str(final_scaled_path),
                    "final_ply_mm": str(final_mm_path),
                }
            )
        range_metadata = export_old_style_range_objects(
            output_root, spec, [final_ply / filenames[stem] for stem in sorted(volumes)]
        )
        for row in rescale_rows:
            row.update(
                {
                    "target_range_min": range_metadata["target_min"],
                    "target_range_max": range_metadata["target_max"],
                    "range_global_min": range_metadata["global_min"],
                    "range_global_max": range_metadata["global_max"],
                    "range_linear_scale_factor": range_metadata["scaling_factor"],
                    "scaled_obj": str(scaled_obj_dir(output_root, spec.name) / f"{row['scan_id']}.obj"),
                }
            )
        rescale_frame = pd.DataFrame(rescale_rows).sort_values("scan_id")
        save_dataframe(rescale_frame, correspondence_root(output_root, spec.name) / "rescale_details.csv")
        payload = {
            "structure": asdict(spec),
            "input_count": len(rigid_paths),
            "final_count": len(rescale_rows),
            "reference_medoid": rigid_metadata["reference_medoid"],
            "deformetrica_iterations": effective_iterations,
            "requested_deformetrica_iterations": int(args.deformetrica_iterations),
            "atlas_action": atlas_action,
            "final_ply_dir": str(final_ply),
            "final_ply_mm_dir": str(final_ply_mm),
            "scaled_obj_dir": str(scaled_obj_dir(output_root, spec.name)),
            "global_scale_factor": global_scale,
            "volume_unscale_factor": volume_unscale,
            "target_range": [TARGET_RANGE_MIN, TARGET_RANGE_MAX],
        }
        write_json(correspondence_root(output_root, spec.name) / "summary.json", payload)
        summary["structures"][spec.name] = payload
    write_json(reports_dir(output_root) / "correspondence_summary.json", summary)
    write_volume_lineage(output_root, structures)
    print(json.dumps(summary, indent=2))


def validate_topology(paths: Sequence[Path]) -> tuple[int, int, bool]:
    import numpy as np

    reference_faces = None
    vertex_count = 0
    face_count = 0
    for path in paths:
        mesh = load_mesh(path)
        faces = np.asarray(mesh.faces)
        if reference_faces is None:
            reference_faces = faces
            vertex_count, face_count = len(mesh.vertices), len(faces)
        elif len(mesh.vertices) != vertex_count or len(faces) != face_count or not np.array_equal(faces, reference_faces):
            return vertex_count, face_count, False
    return vertex_count, face_count, True


def run_stage_validate(args: argparse.Namespace) -> None:
    import pandas as pd

    print_header("Stage: Validate mesh sets and correspondence")
    output_root = Path(args.output_root).resolve()
    structures = parse_structures(args.structures)
    manifest = load_dataframe(selected_manifest_path(output_root))
    requested_diagnoses = set(manifest["requested_diagnoses"].iloc[0].split("|"))
    summary: dict[str, Any] = {"passed": True, "structures": {}}
    details: list[dict[str, Any]] = []
    for spec in structures:
        qc = load_dataframe(structure_root(output_root, spec.name) / "minimal_smooth_qc.csv")
        successful = qc[qc["status"].isin(SUCCESS_STATUSES)].copy()
        expected_stems = set(successful["scan_id"].astype(str))
        source_stems = {path.stem for path in successful_smooth_mesh_paths(output_root, spec)}
        prepared_stems = {
            stem
            for stem in expected_stems
            if (prepared_ply_dir(output_root, spec.name) / f"{stem}.ply").is_file()
        }
        rigid_stems = {
            stem for stem in expected_stems if (rigid_ply_dir(output_root, spec.name) / f"{stem}.ply").is_file()
        }
        final_stems = {
            stem for stem in expected_stems if (final_ply_dir(output_root, spec.name) / f"{stem}.ply").is_file()
        }
        final_mm_stems = {
            stem for stem in expected_stems if (final_ply_mm_dir(output_root, spec.name) / f"{stem}.ply").is_file()
        }
        scaled_obj_stems = {
            stem for stem in expected_stems if (scaled_obj_dir(output_root, spec.name) / f"{stem}.obj").is_file()
        }
        failures: list[str] = []
        if not expected_stems:
            failures.append("no_successful_minimal_smooth_meshes")
        if source_stems != expected_stems:
            failures.append("minimal_smooth_stem_set_mismatch")
        non_watertight_smooth = [
            path.stem
            for path in successful_smooth_mesh_paths(output_root, spec)
            if not load_mesh(path).is_watertight
        ]
        if non_watertight_smooth:
            failures.append("non_watertight_minimal_smooth_mesh")
        if prepared_stems != expected_stems:
            failures.append("prepared_stem_set_mismatch")
        if rigid_stems != expected_stems:
            failures.append("rigid_stem_set_mismatch")
        if final_stems != expected_stems:
            failures.append("final_stem_set_mismatch")
        if final_mm_stems != expected_stems:
            failures.append("final_mm_stem_set_mismatch")
        if scaled_obj_stems != expected_stems:
            failures.append("scaled_obj_stem_set_mismatch")
        if not set(successful["diagnosis"].astype(str)).issubset(requested_diagnoses):
            failures.append("diagnosis_filter_mismatch")
        final_paths = [final_ply_dir(output_root, spec.name) / f"{stem}.ply" for stem in sorted(expected_stems)]
        vertex_count, face_count, identical_topology = validate_topology(final_paths)
        if not identical_topology:
            failures.append("correspondence_topology_mismatch")
        final_mm_paths = [final_ply_mm_dir(output_root, spec.name) / f"{stem}.ply" for stem in sorted(expected_stems)]
        _mm_vertices, _mm_faces, mm_topology = validate_topology(final_mm_paths)
        if not mm_topology:
            failures.append("physical_mm_topology_mismatch")
        max_normalized_volume_error = 0.0
        max_physical_volume_error = 0.0
        for stem in sorted(expected_stems):
            prepared_volume = abs(float(load_mesh(prepared_ply_dir(output_root, spec.name) / f"{stem}.ply").volume))
            final_volume_scaled = abs(float(load_mesh(final_ply_dir(output_root, spec.name) / f"{stem}.ply").volume))
            smooth_volume_mm3 = abs(float(load_mesh(smooth_ply_dir(output_root, spec.name) / f"{stem}.ply").volume))
            final_volume_mm3 = abs(float(load_mesh(final_ply_mm_dir(output_root, spec.name) / f"{stem}.ply").volume))
            normalized_error_pct = 100.0 * abs(final_volume_scaled - prepared_volume) / prepared_volume
            physical_error_pct = 100.0 * abs(final_volume_mm3 - smooth_volume_mm3) / smooth_volume_mm3
            max_normalized_volume_error = max(max_normalized_volume_error, normalized_error_pct)
            max_physical_volume_error = max(max_physical_volume_error, physical_error_pct)
            details.append(
                {
                    "structure": spec.name,
                    "scan_id": stem,
                    "prepared_volume_scaled": prepared_volume,
                    "final_volume_scaled": final_volume_scaled,
                    "normalized_volume_error_pct": normalized_error_pct,
                    "minimal_smooth_volume_mm3": smooth_volume_mm3,
                    "final_volume_mm3": final_volume_mm3,
                    "final_vs_smooth_volume_error_pct": physical_error_pct,
                }
            )
        scale_info = scaled_obj_dir(output_root, spec.name) / "scale_info.json"
        if not scale_info.is_file():
            failures.append("missing_scale_info")
            scaled_min, scaled_max = math.nan, math.nan
        else:
            scaled_vertices = [
                load_mesh(path).vertices
                for path in sorted(scaled_obj_dir(output_root, spec.name).glob("*.obj"))
            ]
            if not scaled_vertices:
                failures.append("empty_scaled_obj_output")
                scaled_min, scaled_max = math.nan, math.nan
            else:
                import numpy as np

                all_vertices = np.concatenate(scaled_vertices, axis=0)
                scaled_min, scaled_max = float(all_vertices.min()), float(all_vertices.max())
                if (
                    scaled_min < TARGET_RANGE_MIN - float(args.range_tolerance)
                    or scaled_max > TARGET_RANGE_MAX + float(args.range_tolerance)
                ):
                    failures.append("scaled_obj_range_out_of_bounds")
        if max_normalized_volume_error > float(args.volume_error_threshold_pct):
            failures.append("normalized_volume_preservation_error")
        if max_physical_volume_error > float(args.volume_error_threshold_pct):
            failures.append("physical_volume_preservation_error")
        passed = not failures
        summary["passed"] &= passed
        summary["structures"][spec.name] = {
            "passed": passed,
            "failures": failures,
            "successful_meshes": len(expected_stems),
            "failed_or_blocked_meshes": int((~qc["status"].isin(SUCCESS_STATUSES)).sum()),
            "non_watertight_minimal_smooth_meshes": non_watertight_smooth,
            "vertex_count": vertex_count,
            "face_count": face_count,
            "identical_face_connectivity": identical_topology,
            "max_normalized_volume_error_pct": max_normalized_volume_error,
            "max_final_vs_smooth_volume_error_pct": max_physical_volume_error,
            "scaled_obj_min": scaled_min,
            "scaled_obj_max": scaled_max,
            "scaled_obj_target_range": [TARGET_RANGE_MIN, TARGET_RANGE_MAX],
        }
    save_dataframe(pd.DataFrame(details), reports_dir(output_root) / "validation_details.csv")
    write_json(reports_dir(output_root) / "validation_summary.json", summary)
    write_volume_lineage(output_root, structures)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


def run_stage_notebook(args: argparse.Namespace) -> None:
    print_header("Stage: Create visualization notebook")
    notebook_script = Path(args.notebook_script).resolve()
    output_root = Path(args.output_root).resolve()
    notebook_output = (
        Path(args.notebook_output).resolve()
        if args.notebook_output is not None
        else DEFAULT_NOTEBOOK_OUTPUT_DIR / f"visualize_synthseg_meshes_{output_root.name}.ipynb"
    )
    command = [
        args.inr_python,
        str(notebook_script),
        "--output-root",
        str(output_root),
        "--notebook-output",
        str(notebook_output),
    ]
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


def add_common_stage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--structures", nargs="+", default=list(DEFAULT_STRUCTURES), choices=list(STRUCTURE_ARGUMENT_CHOICES)
    )


def default_run_output_root(sample_count: int) -> Path:
    name = "full" if sample_count == 0 else f"pilot_{sample_count}"
    return DEFAULT_OUTPUT_BASE / name


def run_subprocess(command: Sequence[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def run_pipeline(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else default_run_output_root(args.sample_count)
    structures = [spec.name for spec in parse_structures(args.structures)]
    diagnoses = parse_diagnoses(args.diagnoses)
    shared = ["--output-root", str(output_root), "--structures", *structures]
    manifest_command = [
        args.inr_python,
        str(script),
        "manifest",
        *shared,
        "--segmentation-root",
        str(Path(args.segmentation_root).resolve()),
        "--clinical-csv",
        str(Path(args.clinical_csv).resolve()),
        "--diagnoses",
        *diagnoses,
        "--sample-count",
        str(args.sample_count),
        "--seed",
        str(args.seed),
        "--gaussian-sigma",
        str(args.gaussian_sigma),
        "--closing-iterations",
        str(args.closing_iterations),
        "--crop-margin",
        str(args.crop_margin),
        "--component-policy",
        args.component_policy,
    ]
    if args.strict_no_mci_subjects:
        manifest_command.append("--strict-no-mci-subjects")
    if args.fill_holes:
        manifest_command.append("--fill-holes")
    raw_command = [
        args.inr_python,
        str(script),
        "raw",
        *shared,
        "--workers",
        str(args.workers),
    ]
    if args.overwrite:
        raw_command.append("--overwrite")
    minimal_smooth_command = [
        args.inr_python,
        str(script),
        "minimal-smooth",
        *shared,
        "--gaussian-sigma",
        str(args.gaussian_sigma),
        "--closing-iterations",
        str(args.closing_iterations),
        "--crop-margin",
        str(args.crop_margin),
        "--component-policy",
        args.component_policy,
        "--workers",
        str(args.workers),
    ]
    if args.fill_holes:
        minimal_smooth_command.append("--fill-holes")
    if args.overwrite:
        minimal_smooth_command.append("--overwrite")
    raw_qc_command = [args.inr_python, str(script), "raw-qc", *shared]
    minimal_smooth_qc_command = [args.inr_python, str(script), "minimal-smooth-qc", *shared]
    if args.allow_partial_cohort:
        raw_qc_command.append("--allow-partial-cohort")
        minimal_smooth_qc_command.append("--allow-partial-cohort")
    notebook_command = [
        args.inr_python,
        str(script),
        "notebook",
        *shared,
        "--inr-python",
        args.inr_python,
        "--notebook-script",
        str(Path(args.notebook_script).resolve()),
        "--notebook-output",
        str(
            Path(args.notebook_output).resolve()
            if args.notebook_output is not None
            else DEFAULT_NOTEBOOK_OUTPUT_DIR / f"visualize_synthseg_meshes_{output_root.name}.ipynb"
        ),
    ]
    commands = [
        manifest_command,
        raw_command,
        raw_qc_command,
        minimal_smooth_command,
        minimal_smooth_qc_command,
        [args.inr_python, str(script), "prepare", *shared],
        [
            args.shapeworks_python,
            str(script),
            "rigid",
            *shared,
            "--shapeworks-iterations",
            str(args.shapeworks_iterations),
        ],
        [
            args.deformetrica_python,
            str(script),
            "correspond",
            *shared,
            "--deformetrica-iterations",
            str(args.deformetrica_iterations),
            "--deformetrica-gpu-mode",
            args.deformetrica_gpu_mode,
        ],
        [
            args.inr_python,
            str(script),
            "validate",
            *shared,
            "--volume-error-threshold-pct",
            str(args.volume_error_threshold_pct),
            "--range-tolerance",
            str(args.range_tolerance),
        ],
        notebook_command,
    ]
    print_header(f"Run SynthSeg correspondence pipeline: {output_root}")
    for command in commands:
        run_subprocess(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full raw → smooth → correspondence workflow.")
    run_parser.add_argument("--segmentation-root", type=Path, default=DEFAULT_SEGMENTATION_ROOT)
    run_parser.add_argument("--clinical-csv", type=Path, default=DEFAULT_CLINICAL_CSV)
    run_parser.add_argument("--output-root", type=Path, default=None)
    run_parser.add_argument(
        "--structures", nargs="+", default=list(DEFAULT_STRUCTURES), choices=list(STRUCTURE_ARGUMENT_CHOICES)
    )
    run_parser.add_argument(
        "--diagnoses", nargs="+", default=list(DEFAULT_DIAGNOSES), choices=list(DIAGNOSIS_ARGUMENT_CHOICES)
    )
    run_parser.add_argument("--sample-count", type=int, default=3, help="0 processes every eligible scan.")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--strict-no-mci-subjects", action="store_true")
    run_parser.add_argument("--gaussian-sigma", type=float, default=0.5)
    run_parser.add_argument("--closing-iterations", type=int, default=0)
    run_parser.add_argument("--fill-holes", action="store_true")
    run_parser.add_argument("--crop-margin", type=int, default=4)
    run_parser.add_argument("--component-policy", choices=["largest", "error"], default="largest")
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument(
        "--allow-partial-cohort",
        action="store_true",
        help="Continue after QC failures with only successful meshes; the default stops at each QC gate.",
    )
    run_parser.add_argument("--shapeworks-iterations", type=int, default=100)
    run_parser.add_argument("--deformetrica-iterations", type=int, default=10)
    run_parser.add_argument("--deformetrica-gpu-mode", choices=["auto", "cpu"], default="auto")
    run_parser.add_argument("--volume-error-threshold-pct", type=float, default=0.01)
    run_parser.add_argument("--range-tolerance", type=float, default=1e-6)
    run_parser.add_argument("--shapeworks-python", default=DEFAULT_SHAPEWORKS_PYTHON)
    run_parser.add_argument("--deformetrica-python", default=DEFAULT_DEFORMETRICA_PYTHON)
    run_parser.add_argument("--inr-python", default=DEFAULT_INR_PYTHON)
    run_parser.add_argument("--notebook-script", type=Path, default=DEFAULT_NOTEBOOK_SCRIPT)
    run_parser.add_argument(
        "--notebook-output",
        type=Path,
        default=None,
        help="Notebook path; default is task3_longitudinal_prediction/notebooks with the pilot/full name.",
    )

    manifest_parser = subparsers.add_parser("manifest", help="Create the clinical/SynthSeg selection manifests.")
    add_common_stage_args(manifest_parser)
    manifest_parser.add_argument("--segmentation-root", required=True, type=Path)
    manifest_parser.add_argument("--clinical-csv", required=True, type=Path)
    manifest_parser.add_argument(
        "--diagnoses", nargs="+", default=list(DEFAULT_DIAGNOSES), choices=list(DIAGNOSIS_ARGUMENT_CHOICES)
    )
    manifest_parser.add_argument("--sample-count", type=int, default=3)
    manifest_parser.add_argument("--seed", type=int, default=42)
    manifest_parser.add_argument("--strict-no-mci-subjects", action="store_true")
    manifest_parser.add_argument("--gaussian-sigma", type=float, default=0.5)
    manifest_parser.add_argument("--closing-iterations", type=int, default=0)
    manifest_parser.add_argument("--fill-holes", action="store_true")
    manifest_parser.add_argument("--crop-margin", type=int, default=4)
    manifest_parser.add_argument("--component-policy", choices=["largest", "error"], default="largest")

    raw_parser = subparsers.add_parser("raw", help="Save every raw segmentation mesh before smoothing.")
    add_common_stage_args(raw_parser)
    raw_parser.add_argument("--workers", type=int, default=1)
    raw_parser.add_argument("--overwrite", action="store_true")

    smooth_parser = subparsers.add_parser(
        "minimal-smooth", help="Create minimal-smooth, verified-watertight meshes from saved raw meshes."
    )
    add_common_stage_args(smooth_parser)
    smooth_parser.add_argument("--gaussian-sigma", type=float, default=0.5)
    smooth_parser.add_argument("--closing-iterations", type=int, default=0)
    smooth_parser.add_argument("--fill-holes", action="store_true")
    smooth_parser.add_argument("--crop-margin", type=int, default=4)
    smooth_parser.add_argument("--component-policy", choices=["largest", "error"], default="largest")
    smooth_parser.add_argument("--workers", type=int, default=1)
    smooth_parser.add_argument("--overwrite", action="store_true")

    raw_qc_parser = subparsers.add_parser("raw-qc", help="Gate smoothing on complete raw-mesh output.")
    add_common_stage_args(raw_qc_parser)
    raw_qc_parser.add_argument("--allow-partial-cohort", action="store_true")

    smooth_qc_parser = subparsers.add_parser(
        "minimal-smooth-qc", help="Gate correspondence on watertight minimal-smooth output."
    )
    add_common_stage_args(smooth_qc_parser)
    smooth_qc_parser.add_argument("--allow-partial-cohort", action="store_true")

    prepare_parser = subparsers.add_parser("prepare", help="Center and globally scale minimal-smooth meshes.")
    add_common_stage_args(prepare_parser)

    rigid_parser = subparsers.add_parser("rigid", help="Rigidly align each structure with ShapeWorks.")
    add_common_stage_args(rigid_parser)
    rigid_parser.add_argument("--shapeworks-iterations", type=int, default=100)

    correspond_parser = subparsers.add_parser("correspond", help="Create per-structure Deformetrica correspondence.")
    add_common_stage_args(correspond_parser)
    correspond_parser.add_argument("--deformetrica-iterations", type=int, default=10)
    correspond_parser.add_argument("--deformetrica-gpu-mode", choices=["auto", "cpu"], default="auto")

    validate_parser = subparsers.add_parser("validate", help="Validate mesh counts, topology, and volume preservation.")
    add_common_stage_args(validate_parser)
    validate_parser.add_argument("--volume-error-threshold-pct", type=float, default=0.01)
    validate_parser.add_argument("--range-tolerance", type=float, default=1e-6)

    notebook_parser = subparsers.add_parser("notebook", help="Generate the interactive visualization notebook.")
    add_common_stage_args(notebook_parser)
    notebook_parser.add_argument("--inr-python", default=DEFAULT_INR_PYTHON)
    notebook_parser.add_argument("--notebook-script", type=Path, default=DEFAULT_NOTEBOOK_SCRIPT)
    notebook_parser.add_argument(
        "--notebook-output",
        type=Path,
        default=None,
        help="Notebook path; default is task3_longitudinal_prediction/notebooks with the pilot/full name.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            run_pipeline(args)
        elif args.command == "manifest":
            run_stage_manifest(args)
        elif args.command == "raw":
            run_stage_raw(args)
        elif args.command == "minimal-smooth":
            run_stage_minimal_smooth(args)
        elif args.command == "raw-qc":
            run_stage_raw_qc(args)
        elif args.command == "minimal-smooth-qc":
            run_stage_minimal_smooth_qc(args)
        elif args.command == "prepare":
            run_stage_prepare(args)
        elif args.command == "rigid":
            run_stage_rigid(args)
        elif args.command == "correspond":
            run_stage_correspond(args)
        elif args.command == "validate":
            run_stage_validate(args)
        elif args.command == "notebook":
            run_stage_notebook(args)
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
