#!/usr/bin/env python3
"""Run a pilot or full ADNI hippocampus preprocessing pipeline from flat PLY inputs.

This script orchestrates a minimal-only workflow for the large ADNI dataset:

1. Select eligible paired scans from a flat left/right PLY directory.
2. Prepare centered and globally scaled left/right/combined PLY folders.
3. Perform rigid registration with ShapeWorks.
4. Establish dense correspondence with Deformetrica and preserve input volumes.
5. Convert correspondence PLY outputs to OBJ and scale them to [-0.9, 0.9].
6. Create label files carrying diagnosis, age, sex, mask volume, and mesh volume.
7. Create subject-level train/val/test splits.
8. Generate DeepSDF samples with preprocess_data.py.
9. Validate the pilot or full output before downstream training.

The top-level ``run`` command uses the same script as a stage runner under
multiple Python interpreters so the user only needs a single entry point.
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
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_SHAPEWORKS_PYTHON = "/home/jakaria/anaconda3/envs/shapeworks/bin/python"
DEFAULT_DEFORMETRICA_PYTHON = (
    "/home/jakaria/Explaining_Shape_Variability/preprocessing/deformetrica_reg/bin/python"
)
DEFAULT_INR_PYTHON = "/home/jakaria/anaconda3/envs/inr_sdf/bin/python"
DEFAULT_PREPROCESS_SCRIPT = "/home/jakaria/INR/Deep3DComp/preprocess_data.py"

TARGET_OBJ_MIN = -0.9
TARGET_OBJ_MAX = 0.9
GLOBAL_SCALE_BUFFER = 1.2
DEFAULT_GROUPS = ("left", "right", "combined")
DIAGNOSIS_FILTER_CHOICES = ("all", "cn_ad_only", "no_mci", "strict_no_mci")
DIAGNOSIS_MAP = {"CN": 0.0, "AD": 1.0, "MCI": -1.0}
SEX_MAP = {"M": 0.0, "MALE": 0.0, "F": 1.0, "FEMALE": 1.0}
GROUP_CONFIGS = {
    "left": {
        "label_suffix": "left",
        "mask_volume_col": "left_mask_volume_mm3",
        "mesh_volume_col": "left_mesh_volume_mm3",
        "source_column": "left_source_ply",
    },
    "right": {
        "label_suffix": "right",
        "mask_volume_col": "right_mask_volume_mm3",
        "mesh_volume_col": "right_mesh_volume_mm3",
        "source_column": "right_source_ply",
    },
    "combined": {
        "label_suffix": "combined",
        "mask_volume_col": "total_mask_volume_mm3",
        "mesh_volume_col": "total_mesh_volume_mm3",
        "source_column": None,
    },
}
RECON_SUBJECT_RE = re.compile(r"__subject_(.+)\.vtk$")


def parse_groups(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_GROUPS)
    groups: list[str] = []
    for value in values:
        if value not in GROUP_CONFIGS:
            raise ValueError(f"Unsupported group: {value}")
        if value not in groups:
            groups.append(value)
    return groups


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_header(title: str) -> None:
    print(f"\n{'=' * 88}")
    print(title)
    print(f"{'=' * 88}")


def manifest_dir(output_root: Path) -> Path:
    return output_root / "manifests"


def reports_dir(output_root: Path) -> Path:
    return output_root / "reports"


def split_root(output_root: Path) -> Path:
    return output_root / "splits"


def group_prepare_base(output_root: Path, group: str) -> Path:
    return output_root / f"{group}_hippocampus_ply"


def group_prepare_dir(output_root: Path, group: str) -> Path:
    return group_prepare_base(output_root, group) / "minimal"


def group_rigid_base(output_root: Path, group: str) -> Path:
    return output_root / f"{group}_hippocampus_ply_rigid_reg"


def group_rigid_dir(output_root: Path, group: str) -> Path:
    return group_rigid_base(output_root, group) / "minimal"


def group_correspondence_base(output_root: Path, group: str) -> Path:
    return output_root / f"{group}_hippocampus_correspondence"


def group_split_dir(output_root: Path, group: str) -> Path:
    return split_root(output_root) / group


def stage_summary_path(output_root: Path, stage_name: str) -> Path:
    return reports_dir(output_root) / f"{stage_name}_summary.json"


def selected_manifest_path(output_root: Path) -> Path:
    return manifest_dir(output_root) / "selected_scans.csv"


def eligible_manifest_path(output_root: Path) -> Path:
    return manifest_dir(output_root) / "eligible_scans.csv"


def group_stem(scan_id: str, group: str) -> str:
    return f"{scan_id}_{GROUP_CONFIGS[group]['label_suffix']}"


def filename_from_stem(stem: str, extension: str) -> str:
    return f"{stem}.{extension}"


def is_subject_stem(stem: str) -> bool:
    return not stem.startswith("template_")


def list_mesh_stems(mesh_dir: Path, extension: str) -> list[str]:
    stems = [path.stem for path in mesh_dir.glob(f"*.{extension}") if is_subject_stem(path.stem)]
    return sorted(stems)


def list_mesh_paths(mesh_dir: Path, extension: str, include_templates: bool = False) -> list[Path]:
    paths = sorted(mesh_dir.glob(f"*.{extension}"))
    if include_templates:
        return paths
    return [path for path in paths if is_subject_stem(path.stem)]


def viscode_to_month(viscode: str) -> float:
    text = str(viscode).strip()
    if text == "bl":
        return 0.0
    if text.startswith("m") and text[1:].isdigit():
        return float(int(text[1:]))
    return math.nan


def normalize_gender(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().upper()
    if not text:
        return math.nan
    return SEX_MAP.get(text, math.nan)


def normalize_diagnosis(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().upper()
    if not text:
        return math.nan
    return DIAGNOSIS_MAP.get(text, math.nan)


def remove_if_exists(path: Path) -> None:
    if path.is_file():
        path.unlink()


def run_subprocess(command: Sequence[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def pangolin_window_uri_for_current_env(requested_uri: str) -> str:
    if requested_uri != "auto":
        return requested_uri

    inherited_uri = os.environ.get("PANGOLIN_WINDOW_URI")
    if inherited_uri:
        return inherited_uri

    if os.environ.get("DISPLAY"):
        return "x11://"

    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland://"

    return "headless://"


def pangolin_env_for_preprocess(window_uri: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PANGOLIN_WINDOW_URI"] = window_uri
    if window_uri in {"headless://", "nogui://", "none://"}:
        env.setdefault("EGL_PLATFORM", "surfaceless")
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    return env


def load_dataframe(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def save_dataframe(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_manifest(output_root: Path):
    path = selected_manifest_path(output_root)
    if not path.is_file():
        raise FileNotFoundError(f"Selected manifest not found: {path}")
    return load_dataframe(path)


def mesh_bounds_and_volume(mesh_path: Path) -> tuple[Any, float]:
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return mesh.bounds.copy(), abs(float(mesh.volume))


def load_mesh(mesh_path: Path):
    import trimesh

    return trimesh.load(mesh_path, force="mesh", process=False)


def mesh_max_dimension(mesh) -> float:
    extents = mesh.bounds[1] - mesh.bounds[0]
    return float(extents.max())


def center_mesh_in_place(mesh) -> None:
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)


def write_group_readme(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def write_legacy_vtk(vertices, faces, vtk_path: Path) -> None:
    with vtk_path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("vtk output\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {len(vertices)} float\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        handle.write(f"\nPOLYGONS {len(faces)} {len(faces) * 4}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def read_legacy_vtk(vtk_path: Path):
    import numpy as np

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    reading_points = False
    reading_polygons = False
    points_remaining = 0

    with vtk_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("POINTS"):
                parts = line.split()
                points_remaining = int(parts[1])
                reading_points = True
                reading_polygons = False
                continue
            if line.startswith("POLYGONS"):
                reading_points = False
                reading_polygons = True
                continue
            if line.startswith("CELL") or line.startswith("POINT_DATA") or line.startswith("METADATA"):
                reading_points = False
                reading_polygons = False
                continue
            if reading_points and points_remaining > 0:
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
        raise ValueError(f"Could not parse VTK mesh: {vtk_path}")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def mesh_volume_from_vtk(vtk_path: Path) -> float:
    import trimesh

    vertices, faces = read_legacy_vtk(vtk_path)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return abs(float(mesh.volume))


def subject_id_from_reconstruction(path: Path) -> str:
    match = RECON_SUBJECT_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse subject id from reconstruction name: {path.name}")
    return match.group(1)


def suppress_outputs():
    return contextlib.ExitStack()


def apply_diagnosis_filter(frame, diagnosis_filter: str):
    if diagnosis_filter == "all":
        return frame.copy()
    if diagnosis_filter in {"cn_ad_only", "no_mci"}:
        return frame[frame["visit_dx_3class"].isin(["CN", "AD"])].copy()
    if diagnosis_filter == "strict_no_mci":
        required_columns = {"RID", "baseline_dx_3class", "visit_dx_3class"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Clinical CSV is missing required strict no-MCI columns: {sorted(missing)}")
        subject_has_mci = frame.groupby("RID")["visit_dx_3class"].transform(
            lambda values: values.fillna("").eq("MCI").any()
        )
        return frame[
            frame["baseline_dx_3class"].isin(["CN", "AD"])
            & frame["visit_dx_3class"].isin(["CN", "AD"])
            & ~subject_has_mci
        ].copy()
    raise ValueError(f"Unsupported diagnosis filter: {diagnosis_filter}")


def run_stage_manifest(args: argparse.Namespace) -> None:
    import pandas as pd

    print_header("Stage: Manifest")
    output_root = Path(args.output_root).resolve()
    source_ply_dir = Path(args.source_ply_dir).resolve()
    clinical_csv = Path(args.clinical_csv).resolve()
    groups = parse_groups(args.groups)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir(output_root).mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(clinical_csv, dtype={"RID": "string", "VISCODE": "string"})
    frame["RID"] = frame["RID"].astype(str).str.strip()
    frame["VISCODE"] = frame["VISCODE"].astype(str).str.strip()
    if "month_from_viscode" not in frame.columns:
        frame["month_from_viscode"] = frame["VISCODE"].map(viscode_to_month)
    frame["scan_id"] = frame["RID"] + "_" + frame["VISCODE"]
    frame["left_source_ply"] = frame["scan_id"].map(lambda value: str(source_ply_dir / f"{value}.L.hipp.ply"))
    frame["right_source_ply"] = frame["scan_id"].map(lambda value: str(source_ply_dir / f"{value}.R.hipp.ply"))
    frame["has_left_ply"] = frame["left_source_ply"].map(lambda value: Path(value).is_file())
    frame["has_right_ply"] = frame["right_source_ply"].map(lambda value: Path(value).is_file())
    frame["paired_available"] = frame["has_left_ply"] & frame["has_right_ply"]
    frame["age_numeric"] = pd.to_numeric(frame["AGE"], errors="coerce")
    frame["gender_numeric"] = frame["PTGENDER"].map(normalize_gender)
    frame["diagnosis_numeric"] = frame["visit_dx_3class"].map(normalize_diagnosis)
    frame["left_stem"] = frame["scan_id"].map(lambda value: group_stem(value, "left"))
    frame["right_stem"] = frame["scan_id"].map(lambda value: group_stem(value, "right"))
    frame["combined_stem"] = frame["scan_id"].map(lambda value: group_stem(value, "combined"))

    required_volume_cols = [
        "left_mask_volume_mm3",
        "right_mask_volume_mm3",
        "left_mesh_volume_mm3",
        "right_mesh_volume_mm3",
        "total_mask_volume_mm3",
        "total_mesh_volume_mm3",
    ]
    missing_cols = [column for column in required_volume_cols if column not in frame.columns]
    if missing_cols:
        raise ValueError(f"Clinical CSV is missing required columns: {missing_cols}")

    eligible_mask = (
        frame["diagnosis_numeric"].notna()
        & frame["age_numeric"].notna()
        & frame["gender_numeric"].notna()
    )
    if "left" in groups or "combined" in groups:
        eligible_mask &= frame["has_left_ply"]
    if "right" in groups or "combined" in groups:
        eligible_mask &= frame["has_right_ply"]
    eligible = frame[eligible_mask].copy()

    eligible = apply_diagnosis_filter(eligible, args.diagnosis_filter)
    if args.require_paired or "combined" in groups:
        eligible = eligible[eligible["paired_available"]].copy()

    eligible = eligible.sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable").reset_index(drop=True)
    mesh_volume_fill_counts = fill_missing_mesh_volumes(eligible, groups)
    save_dataframe(eligible, eligible_manifest_path(output_root))

    if eligible.empty:
        raise RuntimeError("No eligible scans found after filtering.")

    selected = eligible
    selection_mode = "all_eligible_scans"
    if args.sample_count > 0:
        earliest = eligible.groupby("RID", sort=False, as_index=False).first()
        if args.sample_count > len(earliest):
            raise RuntimeError(
                f"Requested {args.sample_count} pilot scans but only {len(earliest)} unique subjects are eligible."
            )
        rng = random.Random(args.seed)
        sampled_indices = rng.sample(list(earliest.index), args.sample_count)
        selected = earliest.loc[sampled_indices].copy()
        selected = selected.sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable").reset_index(drop=True)
        selection_mode = "subject_unique_pilot"

    save_dataframe(selected, selected_manifest_path(output_root))

    summary = {
        "source_ply_dir": str(source_ply_dir),
        "clinical_csv": str(clinical_csv),
        "eligible_scans": int(len(eligible)),
        "selected_scans": int(len(selected)),
        "unique_eligible_subjects": int(eligible["RID"].nunique()),
        "unique_selected_subjects": int(selected["RID"].nunique()),
        "selection_mode": selection_mode,
        "sample_count": int(args.sample_count),
        "seed": int(args.seed),
        "require_paired": bool(args.require_paired),
        "diagnosis_filter": args.diagnosis_filter,
        "diagnosis_counts": selected["visit_dx_3class"].value_counts(dropna=False).to_dict(),
        "baseline_diagnosis_counts": selected["baseline_dx_3class"].value_counts(dropna=False).to_dict()
        if "baseline_dx_3class" in selected.columns
        else {},
        "mesh_volume_filled_from_source_ply": mesh_volume_fill_counts,
    }
    write_json(stage_summary_path(output_root, "manifest"), summary)
    print(json.dumps(summary, indent=2))


def combined_mesh_for_row(row) -> Any:
    import trimesh

    left_mesh = load_mesh(Path(row["left_source_ply"]))
    right_mesh = load_mesh(Path(row["right_source_ply"]))
    return trimesh.util.concatenate([left_mesh, right_mesh])


def source_mesh_for_row(row, group: str):
    if group == "combined":
        return combined_mesh_for_row(row)
    return load_mesh(Path(row[GROUP_CONFIGS[group]["source_column"]]))


def finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def mesh_volume_mm3_and_source(row, group: str) -> tuple[float, str]:
    existing = finite_float_or_none(row.get(GROUP_CONFIGS[group]["mesh_volume_col"]))
    if existing is not None:
        return existing, "clinical_csv"
    mesh = source_mesh_for_row(row, group)
    return abs(float(mesh.volume)), "computed_from_source_ply"


def mask_volume_mm3_for_row(row, group: str) -> float:
    value = finite_float_or_none(row.get(GROUP_CONFIGS[group]["mask_volume_col"]))
    if value is None:
        stem = row.get(f"{group}_stem", row.get("scan_id", "<unknown>"))
        raise ValueError(f"Missing finite mask volume for {group} scan: {stem}")
    return value


def label_values_for_row(row, group: str) -> list[float]:
    mesh_volume, _ = mesh_volume_mm3_and_source(row, group)
    return [
        normalize_diagnosis(row["visit_dx_3class"]),
        float(row["age_numeric"]),
        float(row["gender_numeric"]),
        mask_volume_mm3_for_row(row, group),
        mesh_volume,
    ]


def fill_missing_mesh_volumes(frame, groups: Sequence[str]) -> dict[str, int]:
    filled_counts: dict[str, int] = {}
    for group in groups:
        column = GROUP_CONFIGS[group]["mesh_volume_col"]
        filled = 0
        for index, value in frame[column].items():
            if finite_float_or_none(value) is not None:
                continue
            mesh_volume, _ = mesh_volume_mm3_and_source(frame.loc[index].to_dict(), group)
            frame.at[index, column] = mesh_volume
            filled += 1
        filled_counts[column] = filled
    return filled_counts


def build_group_prepare_metadata(
    group: str,
    manifest,
    dimension_values: list[float],
    global_scale: float,
    output_dir: Path,
):
    import numpy as np
    import pandas as pd

    max_dimension = float(max(dimension_values))
    min_dimension = float(min(dimension_values))
    mean_dimension = float(np.mean(dimension_values))
    std_dimension = float(np.std(dimension_values))
    buffered = max_dimension * GLOBAL_SCALE_BUFFER
    distance_unscale = 1.0 / global_scale
    volume_unscale = distance_unscale ** 3

    metadata = pd.DataFrame(
        [
            {
                "group": group,
                "total_files": int(len(manifest)),
                "minimal_success": int(len(manifest)),
                "minimal_failed": 0,
                "dimension_min_mm": min_dimension,
                "dimension_max_mm": max_dimension,
                "dimension_mean_mm": mean_dimension,
                "dimension_std_mm": std_dimension,
                "dimension_max_buffered_mm": buffered,
                "global_scale_factor": global_scale,
                "distance_unscale_factor": distance_unscale,
                "volume_unscale_factor": volume_unscale,
                "source_dir": str(output_dir),
            }
        ]
    )
    return metadata


def run_stage_prepare(args: argparse.Namespace) -> None:
    print_header("Stage: Prepare")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    manifest = read_manifest(output_root)

    summary: dict[str, Any] = {"groups": {}}

    for group in groups:
        print_header(f"Prepare Group: {group}")
        prepare_base = group_prepare_base(output_root, group)
        prepare_dir = group_prepare_dir(output_root, group)
        prepare_dir.mkdir(parents=True, exist_ok=True)

        dimension_values: list[float] = []
        scan_rows: list[dict[str, Any]] = []

        for row in manifest.to_dict("records"):
            mesh = source_mesh_for_row(row, group)
            dimension_values.append(mesh_max_dimension(mesh))

        if not dimension_values:
            raise RuntimeError(f"No meshes available for group {group}")

        max_dimension = max(dimension_values)
        buffered_max = max_dimension * GLOBAL_SCALE_BUFFER
        global_scale = 1.0 / buffered_max

        for row in manifest.to_dict("records"):
            mesh = source_mesh_for_row(row, group)
            center_mesh_in_place(mesh)
            mesh.apply_scale(global_scale)
            stem = row[f"{group}_stem"]
            output_path = prepare_dir / filename_from_stem(stem, "ply")
            mesh.export(output_path)
            scan_rows.append(
                {
                    "scan_id": row["scan_id"],
                    "RID": row["RID"],
                    "VISCODE": row["VISCODE"],
                    "prepared_stem": stem,
                    "prepared_ply": str(output_path),
                    "mask_volume_mm3": mask_volume_mm3_for_row(row, group),
                    "mesh_volume_mm3": mesh_volume_mm3_and_source(row, group)[0],
                }
            )

        metadata = build_group_prepare_metadata(group, manifest, dimension_values, global_scale, prepare_dir)
        metadata_path = prepare_base / "metadata.csv"
        save_dataframe(metadata, metadata_path)

        import pandas as pd

        scan_manifest = pd.DataFrame(scan_rows)
        save_dataframe(scan_manifest, prepare_base / "scan_manifest.csv")

        write_group_readme(
            prepare_base / "README.txt",
            [
                f"{group.upper()} HIPPOCAMPUS PREPARED PLY FILES",
                "=" * 88,
                "",
                f"Total files processed: {len(scan_rows)}",
                f"Dimension range: {min(dimension_values):.4f} - {max(dimension_values):.4f} mm",
                f"Max dimension with buffer: {buffered_max:.4f} mm",
                f"Global scale factor: {global_scale:.10f}",
                f"Distance unscale factor: {1.0 / global_scale:.10f}",
                f"Volume unscale factor: {(1.0 / global_scale) ** 3:.10f}",
                "",
                "Output structure:",
                f"  minimal/: centered and globally scaled meshes",
                "  metadata.csv: group-level scaling information",
                "  scan_manifest.csv: scan-level mapping and volumes",
            ],
        )

        summary["groups"][group] = {
            "prepared_dir": str(prepare_dir),
            "count": len(scan_rows),
            "global_scale_factor": global_scale,
            "distance_unscale_factor": 1.0 / global_scale,
            "volume_unscale_factor": (1.0 / global_scale) ** 3,
        }

    write_json(stage_summary_path(output_root, "prepare"), summary)
    print(json.dumps(summary, indent=2))


def run_stage_rigid(args: argparse.Namespace) -> None:
    import pandas as pd
    import shapeworks as sw

    print_header("Stage: Rigid Registration")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    summary: dict[str, Any] = {"groups": {}}

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        for group in groups:
            print_header(f"Rigid Group: {group}")
            input_dir = group_prepare_dir(output_root, group)
            output_dir = group_rigid_dir(output_root, group)
            output_dir.mkdir(parents=True, exist_ok=True)
            ply_paths = sorted(input_dir.glob("*.ply"))
            if not ply_paths:
                raise RuntimeError(f"No prepared PLY files found for group {group}: {input_dir}")

            mesh_items: list[tuple[str, Any]] = []
            for ply_path in ply_paths:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    mesh_items.append((ply_path.stem, sw.Mesh(str(ply_path))))

            ref_index = sw.find_reference_mesh_index([mesh for _, mesh in mesh_items])
            ref_name, ref_mesh = mesh_items[ref_index]

            registered = 0
            for name, mesh in mesh_items:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    transform = mesh.createTransform(ref_mesh, sw.Mesh.AlignmentType.Rigid, int(args.shapeworks_iterations))
                    mesh.applyTransform(transform)
                    mesh.write(str(output_dir / f"{name}.ply"))
                registered += 1

            original_metadata = load_dataframe(group_prepare_base(output_root, group) / "metadata.csv")
            original_metadata["rigid_registered"] = True
            original_metadata["reference_medoid"] = ref_name
            original_metadata["registered"] = registered
            original_metadata["failed_register"] = 0
            save_dataframe(original_metadata, group_rigid_base(output_root, group) / "metadata.csv")

            write_group_readme(
                group_rigid_base(output_root, group) / "reference_medoid.txt",
                [
                    f"{group.upper()} HIPPOCAMPUS - RIGID REGISTRATION",
                    "=" * 88,
                    "",
                    f"Reference medoid: {ref_name}",
                    f"Registered meshes: {registered}",
                    f"Iterations per pair: {int(args.shapeworks_iterations)}",
                ],
            )

            summary["groups"][group] = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "reference_medoid": ref_name,
                "registered": registered,
            }

    write_json(stage_summary_path(output_root, "rigid"), summary)
    print(json.dumps(summary, indent=2))


def run_deformetrica_atlas(
    vtk_dir: Path,
    template_file: Path,
    output_dir: Path,
    max_iterations: int,
    gpu_mode: str,
) -> list[str]:
    import deformetrica as dfca

    vtk_files = sorted(vtk_dir.glob("*.vtk"))
    if not vtk_files:
        raise RuntimeError(f"No VTK files found for atlas estimation: {vtk_dir}")

    subject_ids = [vtk_file.stem for vtk_file in vtk_files]
    dataset_filenames = [[{"hippo": str(vtk_file)}] for vtk_file in vtk_files]
    template_specifications = {
        "hippo": {
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
        "max_iterations": max_iterations,
        "initial_step_size": 0.5,
        "convergence_tolerance": 1e-6,
        "save_every_n_iters": max(10, max_iterations // 5),
    }
    model_options = {
        "deformation_kernel_type": "keops",
        "deformation_kernel_width": 0.05,
        "number_of_timepoints": 25,
    }

    deformetrica = dfca.Deformetrica(output_dir=str(output_dir), verbosity="ERROR")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        deformetrica.estimate_deterministic_atlas(
            template_specifications,
            {"dataset_filenames": dataset_filenames, "subject_ids": subject_ids},
            estimator_options=estimator_options,
            model_options=model_options,
        )
    return subject_ids


def run_stage_correspond(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import trimesh

    print_header("Stage: Dense Correspondence")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    summary: dict[str, Any] = {"groups": {}}

    for group in groups:
        print_header(f"Correspondence Group: {group}")
        rigid_dir = group_rigid_dir(output_root, group)
        rigid_metadata = load_dataframe(group_rigid_base(output_root, group) / "metadata.csv")
        output_base = group_correspondence_base(output_root, group)
        vtk_dir = output_base / "minimal_vtk"
        deform_dir = output_base / "minimal_deformetrica"
        final_vtk_dir = output_base / "minimal_final_vtk"
        final_ply_dir = output_base / "minimal_final_ply"
        vtk_dir.mkdir(parents=True, exist_ok=True)
        deform_dir.mkdir(parents=True, exist_ok=True)
        final_vtk_dir.mkdir(parents=True, exist_ok=True)
        final_ply_dir.mkdir(parents=True, exist_ok=True)

        original_volumes: dict[str, float] = {}
        original_filenames: dict[str, str] = {}
        rigid_ply_paths = sorted(rigid_dir.glob("*.ply"))
        if not rigid_ply_paths:
            raise RuntimeError(f"No rigidly registered meshes found for group {group}: {rigid_dir}")

        for ply_path in rigid_ply_paths:
            mesh = trimesh.load(ply_path, force="mesh", process=False)
            original_volumes[ply_path.stem] = abs(float(mesh.volume))
            original_filenames[ply_path.stem] = ply_path.name
            write_legacy_vtk(mesh.vertices, mesh.faces, vtk_dir / f"{ply_path.stem}.vtk")

        ref_name = rigid_metadata["reference_medoid"].iloc[0]
        template_file = vtk_dir / f"{ref_name}.vtk"
        if not template_file.is_file():
            raise RuntimeError(f"Template medoid VTK not found for group {group}: {template_file}")

        reconstruction_paths = sorted(deform_dir.glob("DeterministicAtlas__Reconstruction__*.vtk"))
        template_output = deform_dir / "DeterministicAtlas__EstimatedParameters__Template_hippo.vtk"
        if not reconstruction_paths or not template_output.is_file():
            run_deformetrica_atlas(
                vtk_dir=vtk_dir,
                template_file=template_file,
                output_dir=deform_dir,
                max_iterations=int(args.iterations),
                gpu_mode=args.deformetrica_gpu_mode,
            )
            reconstruction_paths = sorted(deform_dir.glob("DeterministicAtlas__Reconstruction__*.vtk"))
            template_output = deform_dir / "DeterministicAtlas__EstimatedParameters__Template_hippo.vtk"

        if not reconstruction_paths:
            raise RuntimeError(f"No reconstruction VTK files were created for group {group}.")

        if template_output.is_file():
            template_vertices, template_faces = read_legacy_vtk(template_output)
            write_legacy_vtk(template_vertices, template_faces, final_vtk_dir / "template_minimal.vtk")
            template_mesh = trimesh.Trimesh(vertices=template_vertices, faces=template_faces, process=False)
            template_mesh.export(final_ply_dir / "template_minimal.ply")

        rescaled_rows: list[dict[str, Any]] = []
        for recon_path in reconstruction_paths:
            subject_id = subject_id_from_reconstruction(recon_path)
            if subject_id not in original_volumes:
                raise RuntimeError(f"Reconstruction subject id not found in original volume map: {subject_id}")
            vertices, faces = read_legacy_vtk(recon_path)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            current_volume = abs(float(mesh.volume))
            scale_factor = (original_volumes[subject_id] / current_volume) ** (1.0 / 3.0)
            mesh.vertices *= scale_factor
            final_vtk_path = final_vtk_dir / original_filenames[subject_id].replace(".ply", ".vtk")
            final_ply_path = final_ply_dir / original_filenames[subject_id]
            write_legacy_vtk(mesh.vertices, mesh.faces, final_vtk_path)
            mesh.export(final_ply_path)
            rescaled_rows.append(
                {
                    "subject_id": subject_id,
                    "input_volume_scaled": original_volumes[subject_id],
                    "deformetrica_volume_scaled": current_volume,
                    "scale_factor": scale_factor,
                    "final_volume_scaled": abs(float(mesh.volume)),
                }
            )

        subject_final_plys = list_mesh_paths(final_ply_dir, "ply")
        summary_row = pd.DataFrame(
            [
                {
                    "group": group,
                    "version": "minimal",
                    "iterations": int(args.iterations),
                    "template_medoid": ref_name,
                    "template_vtk": "template_minimal.vtk" if template_output.is_file() else None,
                    "template_ply": "template_minimal.ply" if template_output.is_file() else None,
                    "total_input": len(rigid_ply_paths),
                    "successful": len(subject_final_plys),
                    "final_vtk": len(list_mesh_paths(final_vtk_dir, "vtk")),
                    "final_ply": len(subject_final_plys),
                    "volume_unscale_factor": float(rigid_metadata["volume_unscale_factor"].iloc[0]),
                }
            ]
        )
        save_dataframe(summary_row, output_base / "minimal_summary.csv")
        save_dataframe(pd.DataFrame(rescaled_rows), output_base / "minimal_rescale_details.csv")
        write_group_readme(
            output_base / "minimal_README.txt",
            [
                f"{group.upper()} MINIMAL - CORRESPONDENCE WITH VOLUME PRESERVATION",
                "=" * 88,
                "",
                f"Input meshes: {len(rigid_ply_paths)}",
                f"Initial template (medoid): {ref_name}",
                f"Deformetrica iterations: {int(args.iterations)}",
                f"Final subject PLY files: {len(subject_final_plys)}",
            ],
        )

        summary["groups"][group] = {
            "input_count": len(rigid_ply_paths),
            "final_subject_count": len(subject_final_plys),
            "reference_medoid": ref_name,
            "output_base": str(output_base),
        }

    write_json(stage_summary_path(output_root, "correspond"), summary)
    print(json.dumps(summary, indent=2))


def compute_global_min_max(file_paths: Sequence[Path]) -> tuple[float, float]:
    import numpy as np

    global_min = math.inf
    global_max = -math.inf
    for file_path in file_paths:
        mesh = load_mesh(file_path)
        vertices = np.asarray(mesh.vertices)
        global_min = min(global_min, float(vertices.min()))
        global_max = max(global_max, float(vertices.max()))
    if not math.isfinite(global_min) or not math.isfinite(global_max):
        raise RuntimeError("Could not compute global min/max for OBJ scaling.")
    return global_min, global_max


def scale_mesh_to_uniform_range(mesh, global_min: float, global_max: float, target_min: float, target_max: float):
    scaling_factor = (target_max - target_min) / (global_max - global_min)
    mesh.vertices = (mesh.vertices - global_min) * scaling_factor + target_min
    return scaling_factor


def run_stage_obj(args: argparse.Namespace) -> None:
    print_header("Stage: OBJ Conversion and Scaling")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    summary: dict[str, Any] = {"groups": {}}

    for group in groups:
        print_header(f"OBJ Group: {group}")
        output_base = group_correspondence_base(output_root, group)
        final_ply_dir = output_base / "minimal_final_ply"
        final_obj_dir = output_base / "minimal_final_obj"
        scaled_obj_dir = output_base / "minimal_scaled_obj_files"
        final_obj_dir.mkdir(parents=True, exist_ok=True)
        scaled_obj_dir.mkdir(parents=True, exist_ok=True)

        ply_paths = list_mesh_paths(final_ply_dir, "ply", include_templates=True)
        if not ply_paths:
            raise RuntimeError(f"No final PLY files available for OBJ conversion: {final_ply_dir}")

        obj_paths: list[Path] = []
        for ply_path in ply_paths:
            mesh = load_mesh(ply_path)
            obj_path = final_obj_dir / ply_path.with_suffix(".obj").name
            mesh.export(obj_path)
            obj_paths.append(obj_path)

        global_min, global_max = compute_global_min_max(obj_paths)
        scaling_factor = (TARGET_OBJ_MAX - TARGET_OBJ_MIN) / (global_max - global_min)

        for obj_path in obj_paths:
            mesh = load_mesh(obj_path)
            scale_mesh_to_uniform_range(mesh, global_min, global_max, TARGET_OBJ_MIN, TARGET_OBJ_MAX)
            mesh.export(scaled_obj_dir / obj_path.name)

        write_json(
            scaled_obj_dir / "scale_info.json",
            {
                "global_min": float(global_min),
                "global_max": float(global_max),
                "target_min": float(TARGET_OBJ_MIN),
                "target_max": float(TARGET_OBJ_MAX),
                "scaling_factor": float(scaling_factor),
            },
        )

        summary["groups"][group] = {
            "final_obj_dir": str(final_obj_dir),
            "scaled_obj_dir": str(scaled_obj_dir),
            "subject_obj_count": len(list_mesh_paths(scaled_obj_dir, "obj")),
            "all_obj_count": len(list_mesh_paths(scaled_obj_dir, "obj", include_templates=True)),
            "global_min": float(global_min),
            "global_max": float(global_max),
        }

    write_json(stage_summary_path(output_root, "obj"), summary)
    print(json.dumps(summary, indent=2))


def run_stage_labels(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch

    print_header("Stage: Labels")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    manifest = read_manifest(output_root)
    summary: dict[str, Any] = {"groups": {}}

    for group in groups:
        print_header(f"Labels Group: {group}")
        scaled_obj_dir = group_correspondence_base(output_root, group) / "minimal_scaled_obj_files"
        subject_obj_stems = set(list_mesh_stems(scaled_obj_dir, "obj"))
        labels: dict[str, torch.Tensor] = {}
        csv_rows: list[dict[str, Any]] = []

        for row in manifest.to_dict("records"):
            stem = row[f"{group}_stem"]
            if stem not in subject_obj_stems:
                raise RuntimeError(f"Missing scaled OBJ for label creation: {stem}")
            diagnosis, age, sex, mask_volume, mesh_volume = label_values_for_row(row, group)
            labels[stem] = torch.tensor([diagnosis, age, sex, mask_volume, mesh_volume], dtype=torch.float32)
            csv_rows.append(
                {
                    "stem": stem,
                    "RID": row["RID"],
                    "VISCODE": row["VISCODE"],
                    "diagnosis": diagnosis,
                    "age": age,
                    "sex": sex,
                    "mask_volume_mm3": mask_volume,
                    "mesh_volume_mm3": mesh_volume,
                }
            )

        torch.save(labels, scaled_obj_dir / "labels.pt")
        save_dataframe(pd.DataFrame(csv_rows), scaled_obj_dir / "labels.csv")
        write_json(
            scaled_obj_dir / "label_schema.json",
            {
                "columns": [
                    "diagnosis",
                    "age",
                    "sex",
                    "mask_volume_mm3",
                    "mesh_volume_mm3",
                ],
                "diagnosis_encoding": DIAGNOSIS_MAP,
                "sex_encoding": {"male": 0.0, "female": 1.0},
            },
        )
        summary["groups"][group] = {"labels": len(labels), "dir": str(scaled_obj_dir)}

    write_json(stage_summary_path(output_root, "labels"), summary)
    print(json.dumps(summary, indent=2))


def compute_split_sizes(total_subjects: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total_subjects < 3:
        raise RuntimeError("At least 3 subjects are required to build train/val/test splits.")
    n_test = max(1, int(round(total_subjects * test_ratio)))
    n_val = max(1, int(round(total_subjects * val_ratio)))
    n_train = total_subjects - n_test - n_val
    if n_train < 1:
        deficit = 1 - n_train
        while deficit > 0 and n_val > 1:
            n_val -= 1
            deficit -= 1
        while deficit > 0 and n_test > 1:
            n_test -= 1
            deficit -= 1
        n_train = total_subjects - n_test - n_val
    if n_train < 1:
        raise RuntimeError("Could not construct non-empty train split.")
    return n_train, n_val, n_test


def run_stage_splits(args: argparse.Namespace) -> None:
    print_header("Stage: Splits")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    manifest = read_manifest(output_root)
    rng = random.Random(args.seed)
    summary: dict[str, Any] = {"groups": {}}

    for group in groups:
        print_header(f"Split Group: {group}")
        split_dir = group_split_dir(output_root, group)
        split_dir.mkdir(parents=True, exist_ok=True)

        subject_to_files: dict[str, list[str]] = {}
        for row in manifest.to_dict("records"):
            subject_to_files.setdefault(row["RID"], []).append(filename_from_stem(row[f"{group}_stem"], "obj"))

        subjects = sorted(subject_to_files)
        rng.shuffle(subjects)
        n_train, n_val, n_test = compute_split_sizes(
            len(subjects), args.train_ratio, args.val_ratio, args.test_ratio
        )
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train : n_train + n_val]
        test_subjects = subjects[n_train + n_val :]

        def collect_files(subject_ids: Sequence[str]) -> list[str]:
            files: list[str] = []
            for subject_id in subject_ids:
                files.extend(sorted(subject_to_files[subject_id]))
            return sorted(files)

        train_files = collect_files(train_subjects)
        val_files = collect_files(val_subjects)
        test_files = collect_files(test_subjects)

        train_path = split_dir / f"train_split_{group}_hippocampus_adni_large.json"
        val_path = split_dir / f"val_split_{group}_hippocampus_adni_large.json"
        test_path = split_dir / f"test_split_{group}_hippocampus_adni_large.json"
        write_json(train_path, train_files)
        write_json(val_path, val_files)
        write_json(test_path, test_files)

        summary["groups"][group] = {
            "subjects": len(subjects),
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "test_subjects": len(test_subjects),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
        }

    write_json(stage_summary_path(output_root, "splits"), summary)
    print(json.dumps(summary, indent=2))


def split_files_for_group(output_root: Path, group: str) -> dict[str, Path]:
    split_dir = group_split_dir(output_root, group)
    return {
        "train": split_dir / f"train_split_{group}_hippocampus_adni_large.json",
        "val": split_dir / f"val_split_{group}_hippocampus_adni_large.json",
        "test": split_dir / f"test_split_{group}_hippocampus_adni_large.json",
    }


def expected_stems_from_split(split_path: Path) -> set[str]:
    return {Path(item).stem for item in json.loads(split_path.read_text(encoding="utf-8"))}


def run_stage_sdf(args: argparse.Namespace) -> None:
    print_header("Stage: SDF")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    preprocess_script = Path(args.preprocess_script).resolve()
    pangolin_window_uri = pangolin_window_uri_for_current_env(args.pangolin_window_uri)
    summary: dict[str, Any] = {"groups": {}}

    for group in groups:
        print_header(f"SDF Group: {group}")
        correspondence_base = group_correspondence_base(output_root, group)
        source_dir = correspondence_base / "minimal_scaled_obj_files"
        data_dir = correspondence_base / "sdf_data"
        npz_dir = data_dir / "SdfSamples" / "minimal_scaled_obj_files"
        data_dir.mkdir(parents=True, exist_ok=True)
        split_paths = split_files_for_group(output_root, group)
        missing_by_split: dict[str, list[str]] = {}
        for split_name, split_path in split_paths.items():
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing split file for SDF generation: {split_path}")
            command = [
                args.inr_python,
                str(preprocess_script),
                "--data_dir",
                str(data_dir),
                "--source",
                str(source_dir),
                "--split",
                str(split_path),
                "--threads",
                str(args.sdf_threads),
                "--skip",
            ]
            print(f"Pangolin window URI: {pangolin_window_uri}")
            env = pangolin_env_for_preprocess(pangolin_window_uri)
            print(f"\n$ {' '.join(command)}")
            subprocess.run(command, cwd=str(preprocess_script.parent), env=env, check=True)
            existing_stems = set(path.stem for path in npz_dir.glob("*.npz")) if npz_dir.is_dir() else set()
            missing_stems = sorted(expected_stems_from_split(split_path) - existing_stems)
            if missing_stems:
                missing_by_split[split_name] = missing_stems

        if missing_by_split:
            raise RuntimeError(
                "SDF preprocessing did not produce the expected .npz files. "
                "This usually means PreprocessMesh failed in the current environment "
                "(for example Pangolin/X11 headless issues). "
                f"Missing outputs: {missing_by_split}"
            )

        summary["groups"][group] = {
            "data_dir": str(data_dir),
            "npz_dir": str(npz_dir),
            "npz_count": len(list(npz_dir.glob("*.npz"))) if npz_dir.is_dir() else 0,
        }

    write_json(stage_summary_path(output_root, "sdf"), summary)
    print(json.dumps(summary, indent=2))


def validate_topology(final_ply_paths: list[Path]) -> tuple[int, int, bool]:
    import numpy as np

    reference_faces = None
    reference_vertices = None
    identical = True
    face_count = 0
    vertex_count = 0
    for path in final_ply_paths:
        mesh = load_mesh(path)
        vertices = len(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if reference_faces is None:
            reference_faces = faces
            reference_vertices = vertices
            face_count = len(faces)
            vertex_count = vertices
            continue
        if vertices != reference_vertices or len(faces) != face_count or not np.array_equal(faces, reference_faces):
            identical = False
            break
    return vertex_count, face_count, identical


def run_stage_validate(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import torch

    print_header("Stage: Validate")
    output_root = Path(args.output_root).resolve()
    groups = parse_groups(args.groups)
    check_sdf = not bool(getattr(args, "skip_sdf_check", False))
    manifest = read_manifest(output_root)
    summary: dict[str, Any] = {"passed": True, "groups": {}}
    detail_rows: list[dict[str, Any]] = []

    for group in groups:
        print_header(f"Validate Group: {group}")
        expected_stems = sorted(manifest[f"{group}_stem"].tolist())
        expected_subjects = set(manifest["RID"].tolist())
        prepare_dir = group_prepare_dir(output_root, group)
        rigid_dir = group_rigid_dir(output_root, group)
        correspondence_base = group_correspondence_base(output_root, group)
        final_ply_dir = correspondence_base / "minimal_final_ply"
        final_obj_dir = correspondence_base / "minimal_final_obj"
        scaled_obj_dir = correspondence_base / "minimal_scaled_obj_files"
        npz_dir = correspondence_base / "sdf_data" / "SdfSamples" / "minimal_scaled_obj_files"

        prepared_stems = set(list_mesh_stems(prepare_dir, "ply"))
        rigid_stems = set(list_mesh_stems(rigid_dir, "ply"))
        final_ply_stems = set(list_mesh_stems(final_ply_dir, "ply"))
        final_obj_stems = set(list_mesh_stems(final_obj_dir, "obj"))
        scaled_obj_stems = set(list_mesh_stems(scaled_obj_dir, "obj"))
        npz_stems = set(path.stem for path in npz_dir.glob("*.npz")) if npz_dir.is_dir() else set()

        failures: list[str] = []
        if prepared_stems != set(expected_stems):
            failures.append("prepared_stem_set_mismatch")
        if rigid_stems != set(expected_stems):
            failures.append("rigid_stem_set_mismatch")
        if final_ply_stems != set(expected_stems):
            failures.append("final_ply_stem_set_mismatch")
        if final_obj_stems != set(expected_stems):
            failures.append("final_obj_stem_set_mismatch")
        if scaled_obj_stems != set(expected_stems):
            failures.append("scaled_obj_stem_set_mismatch")
        if check_sdf and npz_stems != set(expected_stems):
            failures.append("sdf_stem_set_mismatch")

        final_ply_paths = [final_ply_dir / filename_from_stem(stem, "ply") for stem in expected_stems]
        vertex_count, face_count, topology_identical = validate_topology(final_ply_paths)
        if not topology_identical:
            failures.append("topology_mismatch")

        max_volume_error_pct = 0.0
        for row in manifest.to_dict("records"):
            stem = row[f"{group}_stem"]
            prepared_mesh = load_mesh(prepare_dir / filename_from_stem(stem, "ply"))
            final_mesh = load_mesh(final_ply_dir / filename_from_stem(stem, "ply"))
            prepared_volume = abs(float(prepared_mesh.volume))
            final_volume = abs(float(final_mesh.volume))
            error_pct = abs(final_volume - prepared_volume) / prepared_volume * 100.0
            max_volume_error_pct = max(max_volume_error_pct, error_pct)
            detail_rows.append(
                {
                    "group": group,
                    "stem": stem,
                    "prepared_volume_scaled": prepared_volume,
                    "final_volume_scaled": final_volume,
                    "volume_error_pct": error_pct,
                }
            )
        if max_volume_error_pct > args.volume_error_threshold_pct:
            failures.append("volume_error_threshold_exceeded")

        scaled_paths = list_mesh_paths(scaled_obj_dir, "obj", include_templates=True)
        global_min, global_max = compute_global_min_max(scaled_paths)
        if global_min < TARGET_OBJ_MIN - args.range_tolerance or global_max > TARGET_OBJ_MAX + args.range_tolerance:
            failures.append("scaled_obj_range_out_of_bounds")

        labels_path = scaled_obj_dir / "labels.pt"
        labels = torch.load(labels_path, map_location="cpu")
        if set(labels.keys()) != set(expected_stems):
            failures.append("label_key_mismatch")
        for row in manifest.to_dict("records"):
            stem = row[f"{group}_stem"]
            label = labels[stem]
            expected = np.asarray(
                label_values_for_row(row, group),
                dtype=np.float32,
            )
            if label.shape[0] != 5 or not np.allclose(label.numpy(), expected, atol=1e-5, rtol=1e-5):
                failures.append("label_value_mismatch")
                break

        split_paths = split_files_for_group(output_root, group)
        split_union: set[str] = set()
        subject_sets: dict[str, set[str]] = {}
        stem_to_subject = {row[f"{group}_stem"]: row["RID"] for row in manifest.to_dict("records")}
        for split_name, split_path in split_paths.items():
            split_items = set(json.loads(split_path.read_text(encoding="utf-8")))
            split_stems = {Path(item).stem for item in split_items}
            split_union |= split_stems
            subject_sets[split_name] = {stem_to_subject[stem] for stem in split_stems}
        if split_union != set(expected_stems):
            failures.append("split_union_mismatch")
        if subject_sets["train"] & subject_sets["val"] or subject_sets["train"] & subject_sets["test"] or subject_sets["val"] & subject_sets["test"]:
            failures.append("split_subject_leakage")

        passed = not failures
        if not passed:
            summary["passed"] = False
        summary["groups"][group] = {
            "passed": passed,
            "failures": failures,
            "expected_shapes": len(expected_stems),
            "unique_subjects": len(expected_subjects),
            "vertex_count": vertex_count,
            "face_count": face_count,
            "topology_identical": topology_identical,
            "max_volume_error_pct": max_volume_error_pct,
            "scaled_obj_min": global_min,
            "scaled_obj_max": global_max,
            "sdf_checked": check_sdf,
            "sdf_count": len(npz_stems),
        }

    save_dataframe(pd.DataFrame(detail_rows), reports_dir(output_root) / "validation_details.csv")
    write_json(stage_summary_path(output_root, "validation"), summary)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


def add_common_stage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))


def run_pipeline(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    script_path = Path(__file__).resolve()
    groups = parse_groups(args.groups)
    shared = ["--output-root", str(output_root), "--groups", *groups]
    manifest_args = [
        args.inr_python,
        str(script_path),
        "manifest",
        *shared,
        "--source-ply-dir",
        str(Path(args.source_ply_dir).resolve()),
        "--clinical-csv",
        str(Path(args.clinical_csv).resolve()),
        "--sample-count",
        str(args.sample_count),
        "--seed",
        str(args.seed),
        "--diagnosis-filter",
        args.diagnosis_filter,
    ]
    if args.require_paired:
        manifest_args.append("--require-paired")

    stages = [
        manifest_args,
        [args.inr_python, str(script_path), "prepare", *shared],
        [
            args.shapeworks_python,
            str(script_path),
            "rigid",
            *shared,
            "--shapeworks-iterations",
            str(args.shapeworks_iterations),
        ],
        [
            args.deformetrica_python,
            str(script_path),
            "correspond",
            *shared,
            "--iterations",
            str(args.iterations),
            "--deformetrica-gpu-mode",
            args.deformetrica_gpu_mode,
        ],
        [args.inr_python, str(script_path), "obj", *shared],
        [args.inr_python, str(script_path), "labels", *shared],
        [
            args.inr_python,
            str(script_path),
            "splits",
            *shared,
            "--seed",
            str(args.seed),
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
            "--test-ratio",
            str(args.test_ratio),
        ],
        [
            args.inr_python,
            str(script_path),
            "sdf",
            *shared,
            "--inr-python",
            args.inr_python,
            "--preprocess-script",
            args.preprocess_script,
            "--sdf-threads",
            str(args.sdf_threads),
            "--pangolin-window-uri",
            args.pangolin_window_uri,
        ],
        [
            args.inr_python,
            str(script_path),
            "validate",
            *shared,
            "--volume-error-threshold-pct",
            str(args.volume_error_threshold_pct),
            "--range-tolerance",
            str(args.range_tolerance),
        ],
    ]

    print_header("Run Pipeline")
    for command in stages:
        run_subprocess(command, cwd=script_path.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline using multiple interpreters.")
    add_common_stage_args(run_parser)
    run_parser.add_argument("--source-ply-dir", required=True)
    run_parser.add_argument("--clinical-csv", required=True)
    run_parser.add_argument("--sample-count", type=int, default=10)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--require-paired", action="store_true")
    run_parser.add_argument("--diagnosis-filter", choices=DIAGNOSIS_FILTER_CHOICES, default="all")
    run_parser.add_argument("--iterations", type=int, default=30)
    run_parser.add_argument("--shapeworks-iterations", type=int, default=100)
    run_parser.add_argument("--deformetrica-gpu-mode", choices=["auto", "cpu"], default="auto")
    run_parser.add_argument("--sdf-threads", type=int, default=16)
    run_parser.add_argument("--train-ratio", type=float, default=0.8)
    run_parser.add_argument("--val-ratio", type=float, default=0.1)
    run_parser.add_argument("--test-ratio", type=float, default=0.1)
    run_parser.add_argument("--volume-error-threshold-pct", type=float, default=1.0)
    run_parser.add_argument("--range-tolerance", type=float, default=1e-5)
    run_parser.add_argument("--shapeworks-python", default=DEFAULT_SHAPEWORKS_PYTHON)
    run_parser.add_argument("--deformetrica-python", default=DEFAULT_DEFORMETRICA_PYTHON)
    run_parser.add_argument("--inr-python", default=DEFAULT_INR_PYTHON)
    run_parser.add_argument("--preprocess-script", default=DEFAULT_PREPROCESS_SCRIPT)
    run_parser.add_argument("--pangolin-window-uri", default="auto")

    manifest_parser = subparsers.add_parser("manifest", help="Select eligible scans and write manifest files.")
    add_common_stage_args(manifest_parser)
    manifest_parser.add_argument("--source-ply-dir", required=True)
    manifest_parser.add_argument("--clinical-csv", required=True)
    manifest_parser.add_argument("--sample-count", type=int, default=10)
    manifest_parser.add_argument("--seed", type=int, default=42)
    manifest_parser.add_argument("--require-paired", action="store_true")
    manifest_parser.add_argument("--diagnosis-filter", choices=DIAGNOSIS_FILTER_CHOICES, default="all")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare centered and globally scaled minimal PLYs.")
    add_common_stage_args(prepare_parser)

    rigid_parser = subparsers.add_parser("rigid", help="Rigidly register prepared PLYs with ShapeWorks.")
    add_common_stage_args(rigid_parser)
    rigid_parser.add_argument("--shapeworks-iterations", type=int, default=100)

    correspond_parser = subparsers.add_parser("correspond", help="Run Deformetrica and preserve prepared volumes.")
    add_common_stage_args(correspond_parser)
    correspond_parser.add_argument("--iterations", type=int, default=30)
    correspond_parser.add_argument("--deformetrica-gpu-mode", choices=["auto", "cpu"], default="auto")

    obj_parser = subparsers.add_parser("obj", help="Convert final PLYs to OBJ and scale to [-0.9, 0.9].")
    add_common_stage_args(obj_parser)

    labels_parser = subparsers.add_parser("labels", help="Create labels.pt and labels.csv with both volume types.")
    add_common_stage_args(labels_parser)

    splits_parser = subparsers.add_parser("splits", help="Create subject-level train/val/test splits.")
    add_common_stage_args(splits_parser)
    splits_parser.add_argument("--seed", type=int, default=42)
    splits_parser.add_argument("--train-ratio", type=float, default=0.8)
    splits_parser.add_argument("--val-ratio", type=float, default=0.1)
    splits_parser.add_argument("--test-ratio", type=float, default=0.1)

    sdf_parser = subparsers.add_parser("sdf", help="Create DeepSDF samples from scaled OBJ folders.")
    add_common_stage_args(sdf_parser)
    sdf_parser.add_argument("--inr-python", default=DEFAULT_INR_PYTHON)
    sdf_parser.add_argument("--preprocess-script", default=DEFAULT_PREPROCESS_SCRIPT)
    sdf_parser.add_argument("--sdf-threads", type=int, default=16)
    sdf_parser.add_argument("--pangolin-window-uri", default="auto")

    validate_parser = subparsers.add_parser("validate", help="Validate counts, topology, labels, splits, and SDFs.")
    add_common_stage_args(validate_parser)
    validate_parser.add_argument("--volume-error-threshold-pct", type=float, default=1.0)
    validate_parser.add_argument("--range-tolerance", type=float, default=1e-5)
    validate_parser.add_argument("--skip-sdf-check", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            run_pipeline(args)
        elif args.command == "manifest":
            run_stage_manifest(args)
        elif args.command == "prepare":
            run_stage_prepare(args)
        elif args.command == "rigid":
            run_stage_rigid(args)
        elif args.command == "correspond":
            run_stage_correspond(args)
        elif args.command == "obj":
            run_stage_obj(args)
        elif args.command == "labels":
            run_stage_labels(args)
        elif args.command == "splits":
            run_stage_splits(args)
        elif args.command == "sdf":
            run_stage_sdf(args)
        elif args.command == "validate":
            run_stage_validate(args)
        else:
            parser.error(f"Unhandled command: {args.command}")
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
