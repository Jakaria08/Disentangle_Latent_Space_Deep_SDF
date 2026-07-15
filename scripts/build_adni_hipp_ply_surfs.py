#!/usr/bin/env python3
"""Build hippocampal PLY surfaces, volume tables, and plots for ADNI masks.

The primary scientific volume is computed directly from the binary mask voxels.
PLY meshes are reconstructed from the mask volume for visualization and shape
features, using the same volume-first approach as the notebook in
test_hipp_1001_bl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import trimesh
from scipy import ndimage
from skimage import measure


DEFAULT_BASE_DIR = Path("/home/jakaria/ADNI/ADNI_1_GO_Large")
MASK_RE = re.compile(
    r"^(?P<RID>\d+)_(?P<VISCODE>bl|m\d+)\.(?P<hemi>[LR])\.hipp\.mgz$"
)

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

BASELINE_DX_MAP = {
    "CN": "CN",
    "NL": "CN",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "MCI": "MCI",
    "AD": "AD",
    "Dementia": "AD",
}

VISIT_CONVERSION_EVENT_MAP = {
    "NL to MCI": "CN_to_MCI",
    "NL to Dementia": "CN_to_AD",
    "MCI to Dementia": "MCI_to_AD",
    "MCI to NL": "reverter",
    "Dementia to MCI": "reverter",
}

CLINICAL_COLUMNS = [
    "RID",
    "VISCODE",
    "EXAMDATE",
    "Month.bl",
    "month_from_viscode",
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "DX.bl",
    "baseline_dx_3class",
    "DX",
    "visit_dx_3class",
    "conversion_event_at_visit",
    "subject_conversion_type",
    "conversion_month",
    "cn_to_mci",
    "cn_to_ad",
    "mci_to_ad",
    "reverter",
    "MCI.convert.month",
    "MCI.convert.targvisit",
    "MMSE",
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "APOE4",
    "AUX.DIAGNOSIS",
    "AUX.GROUP",
    "has_left_mask",
    "has_right_mask",
    "has_any_mask",
]


@dataclass
class MaskTask:
    rid: str
    viscode: str
    month_from_viscode: int
    hemi: str
    mask_path: str
    ply_path: str
    sigma: float
    closing_iterations: int
    padding: int
    overwrite: bool
    skip_existing: bool


def parse_args() -> argparse.Namespace:
    default_workers = max(1, min(16, (os.cpu_count() or 2) - 1))
    parser = argparse.ArgumentParser(
        description="Create ADNI hippocampus PLY surfaces, volume CSVs, and plots."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--mask-dir-name", default="adni_hipp_masks")
    parser.add_argument("--clinical-csv", default="ClinicalInfo.csv")
    parser.add_argument("--output-dir-name", default="adni_hipp_ply_surfs")
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N labeled mask files. Use 0 for all masks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate PLY files even when the output already exists.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Recompute existing PLY files instead of loading QC from them.",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument(
        "--spaghetti-max-subjects",
        type=int,
        default=250,
        help="Maximum subjects per diagnosis in spaghetti plots.",
    )
    return parser.parse_args()


def month_from_viscode(viscode: str) -> float:
    if pd.isna(viscode):
        return np.nan
    viscode = str(viscode)
    if viscode == "bl":
        return 0
    if viscode.startswith("m") and viscode[1:].isdigit():
        return int(viscode[1:])
    return np.nan


def rid_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (10**12, text)


def read_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"RID": "string", "VISCODE": "string"}, keep_default_na=False)
    df["RID"] = df["RID"].astype(str)
    df["VISCODE"] = df["VISCODE"].astype(str)
    df["month_from_viscode"] = df["VISCODE"].map(month_from_viscode)
    df["baseline_dx_3class"] = df["DX.bl"].map(BASELINE_DX_MAP)
    df["visit_dx_3class"] = df["DX"].map(VISIT_DX_MAP)
    df["conversion_event_at_visit"] = df["DX"].map(VISIT_CONVERSION_EVENT_MAP).fillna("")

    for col in ["Month.bl", "AGE", "PTEDUCAT", "MMSE", "CDRSB", "ADAS11", "ADAS13", "APOE4"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({"": np.nan, "NA": np.nan}), errors="coerce")

    return df


def discover_masks(mask_dir: Path) -> pd.DataFrame:
    records = []
    bad_names = []
    for path in sorted(mask_dir.glob("*.mgz")):
        match = MASK_RE.match(path.name)
        if not match:
            bad_names.append(path.name)
            continue
        row = match.groupdict()
        row["month_from_viscode"] = month_from_viscode(row["VISCODE"])
        row["mask_path"] = str(path)
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"No mask files matching {MASK_RE.pattern} found in {mask_dir}")

    df["RID"] = df["RID"].astype(str)
    df["VISCODE"] = df["VISCODE"].astype(str)
    df["hemi"] = df["hemi"].astype(str)
    df["bad_mask_name_count"] = len(bad_names)
    return df.sort_values(["RID", "month_from_viscode", "hemi"], kind="stable").reset_index(drop=True)


def classify_subjects(clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = clinical.dropna(subset=["visit_dx_3class"]).copy()
    labeled = labeled.sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable")

    subject_rows = []
    for rid, group in labeled.groupby("RID", sort=False):
        visits = group[["VISCODE", "month_from_viscode", "visit_dx_3class"]].dropna()
        visits = visits.sort_values(["month_from_viscode", "VISCODE"], kind="stable")
        seq = list(visits["visit_dx_3class"])
        months = list(visits["month_from_viscode"])
        viscodes = list(visits["VISCODE"])
        if not seq:
            continue

        first_dx = seq[0]
        last_dx = seq[-1]
        unique_dx = list(dict.fromkeys(seq))

        cn_to_mci_month = first_transition_month(seq, months, "CN", "MCI")
        cn_to_ad_month = first_transition_month(seq, months, "CN", "AD")
        mci_to_ad_month = first_transition_month(seq, months, "MCI", "AD")
        reverter_month = first_reverter_month(seq, months)

        cn_to_mci = not math.isnan(cn_to_mci_month)
        cn_to_ad = not math.isnan(cn_to_ad_month)
        mci_to_ad = not math.isnan(mci_to_ad_month)
        reverter = not math.isnan(reverter_month)

        if reverter:
            conversion_type = "reverter_or_other"
            conversion_month = reverter_month
        elif first_dx == "CN" and cn_to_mci and mci_to_ad:
            conversion_type = "CN_to_MCI_to_AD"
            conversion_month = cn_to_mci_month
        elif first_dx == "CN" and cn_to_ad:
            conversion_type = "CN_to_AD"
            conversion_month = cn_to_ad_month
        elif first_dx == "CN" and cn_to_mci:
            conversion_type = "CN_to_MCI"
            conversion_month = cn_to_mci_month
        elif first_dx == "MCI" and mci_to_ad:
            conversion_type = "MCI_to_AD"
            conversion_month = mci_to_ad_month
        elif len(set(seq)) == 1:
            conversion_type = f"stable_{first_dx}"
            conversion_month = np.nan
        else:
            conversion_type = "other_or_mixed"
            conversion_month = np.nan

        subject_rows.append(
            {
                "RID": rid,
                "n_labeled_visits": len(seq),
                "first_visit": viscodes[0],
                "last_visit": viscodes[-1],
                "first_month": months[0],
                "last_month": months[-1],
                "first_dx": first_dx,
                "last_dx": last_dx,
                "dx_sequence": " -> ".join(unique_dx),
                "subject_conversion_type": conversion_type,
                "conversion_month": conversion_month,
                "cn_to_mci": cn_to_mci,
                "cn_to_ad": cn_to_ad,
                "mci_to_ad": mci_to_ad,
                "reverter": reverter,
                "cn_to_mci_month": cn_to_mci_month,
                "cn_to_ad_month": cn_to_ad_month,
                "mci_to_ad_month": mci_to_ad_month,
                "reverter_month": reverter_month,
            }
        )

    subject_summary = pd.DataFrame(subject_rows)
    enriched = clinical.merge(
        subject_summary[
            [
                "RID",
                "subject_conversion_type",
                "conversion_month",
                "cn_to_mci",
                "cn_to_ad",
                "mci_to_ad",
                "reverter",
            ]
        ],
        on="RID",
        how="left",
    )
    return enriched, subject_summary


def first_transition_month(
    seq: list[str], months: list[float], source: str, target: str
) -> float:
    seen_source = False
    for dx, month in zip(seq, months):
        if dx == source:
            seen_source = True
        elif seen_source and dx == target:
            return float(month)
    return np.nan


def first_reverter_month(seq: list[str], months: list[float]) -> float:
    order = {"CN": 0, "MCI": 1, "AD": 2}
    previous = order.get(seq[0], 0)
    for dx, month in zip(seq[1:], months[1:]):
        current = order.get(dx, previous)
        if current < previous:
            return float(month)
        previous = current
    return np.nan


def attach_mask_availability(clinical: pd.DataFrame, mask_manifest: pd.DataFrame) -> pd.DataFrame:
    availability = (
        mask_manifest.assign(value=True)
        .pivot_table(
            index=["RID", "VISCODE"],
            columns="hemi",
            values="value",
            aggfunc="any",
            fill_value=False,
        )
        .reset_index()
    )
    if "L" not in availability.columns:
        availability["L"] = False
    if "R" not in availability.columns:
        availability["R"] = False
    availability = availability.rename(columns={"L": "has_left_mask", "R": "has_right_mask"})
    availability["has_any_mask"] = availability["has_left_mask"] | availability["has_right_mask"]

    merged = clinical.merge(availability, on=["RID", "VISCODE"], how="left")
    for col in ["has_left_mask", "has_right_mask", "has_any_mask"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    return merged


def make_tasks(
    mask_manifest: pd.DataFrame,
    clean_clinical: pd.DataFrame,
    output_dir: Path,
    sigma: float,
    closing_iterations: int,
    padding: int,
    overwrite: bool,
    skip_existing: bool,
    dry_run: int,
) -> list[MaskTask]:
    labeled_keys = set(
        clean_clinical.loc[
            clean_clinical["visit_dx_3class"].isin(["AD", "CN", "MCI"]), ["RID", "VISCODE"]
        ].itertuples(index=False, name=None)
    )
    labeled_masks = mask_manifest[
        mask_manifest[["RID", "VISCODE"]].apply(tuple, axis=1).isin(labeled_keys)
    ].copy()
    labeled_masks = labeled_masks.sort_values(
        ["RID", "month_from_viscode", "hemi"], kind="stable"
    )
    if dry_run > 0:
        labeled_masks = labeled_masks.head(dry_run)

    tasks = []
    for row in labeled_masks.itertuples(index=False):
        name = f"{row.RID}_{row.VISCODE}.{row.hemi}.hipp.ply"
        tasks.append(
            MaskTask(
                rid=row.RID,
                viscode=row.VISCODE,
                month_from_viscode=int(row.month_from_viscode),
                hemi=row.hemi,
                mask_path=row.mask_path,
                ply_path=str(output_dir / name),
                sigma=sigma,
                closing_iterations=closing_iterations,
                padding=padding,
                overwrite=overwrite,
                skip_existing=skip_existing,
            )
        )
    return tasks


def mesh_qc(mesh: trimesh.Trimesh) -> dict[str, float | int | bool]:
    volume = abs(float(mesh.volume)) if mesh.is_watertight else np.nan
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "surface_area_mm2": float(mesh.area),
        "mesh_volume_mm3": volume,
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "mesh_components": int(len(mesh.split(only_watertight=False))),
    }


def keep_largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    labels, component_count = ndimage.label(mask)
    if component_count == 0:
        return mask, 0, []
    sizes = ndimage.sum(mask, labels, range(1, component_count + 1))
    sizes = [int(size) for size in sizes]
    keep_label = int(np.argmax(sizes)) + 1
    return labels == keep_label, int(component_count), sizes


def process_mask(task: MaskTask) -> dict[str, object]:
    out = {
        "RID": task.rid,
        "VISCODE": task.viscode,
        "month_from_viscode": task.month_from_viscode,
        "hemi": task.hemi,
        "mask_path": task.mask_path,
        "ply_path": task.ply_path,
        "sigma": task.sigma,
        "closing_iterations": task.closing_iterations,
        "padding": task.padding,
        "status": "started",
        "error": "",
    }
    try:
        mask_path = Path(task.mask_path)
        ply_path = Path(task.ply_path)
        image = nib.load(str(mask_path))
        mask = np.asanyarray(image.dataobj) > 0
        voxel_sizes = tuple(float(x) for x in image.header.get_zooms()[:3])
        voxel_volume = float(np.prod(voxel_sizes))
        voxel_count = int(mask.sum())
        out.update(
            {
                "voxel_size_x_mm": voxel_sizes[0],
                "voxel_size_y_mm": voxel_sizes[1],
                "voxel_size_z_mm": voxel_sizes[2],
                "voxel_volume_mm3": voxel_volume,
                "mask_voxels": voxel_count,
                "mask_volume_mm3": voxel_count * voxel_volume,
            }
        )

        if voxel_count == 0:
            out["status"] = "failed"
            out["error"] = "empty_mask"
            return out

        if ply_path.exists() and task.skip_existing and not task.overwrite:
            mesh = trimesh.load_mesh(ply_path, process=False)
            out.update(mesh_qc(mesh))
            out["status"] = "skipped_existing"
            return out

        coords = np.argwhere(mask)
        min_idx = np.maximum(coords.min(axis=0) - task.padding, 0)
        max_idx = np.minimum(coords.max(axis=0) + task.padding + 1, mask.shape)
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(min_idx, max_idx))
        cropped = mask[slices]

        cropped, source_components, source_sizes = keep_largest_component(cropped)
        if source_components == 0:
            out["status"] = "failed"
            out["error"] = "empty_crop_after_component_filter"
            return out

        repaired = ndimage.binary_closing(cropped, iterations=task.closing_iterations)
        repaired = ndimage.binary_fill_holes(repaired)
        repaired, repaired_components, repaired_sizes = keep_largest_component(repaired)

        smooth = ndimage.gaussian_filter(repaired.astype(np.float32), sigma=task.sigma)
        if smooth.min() > 0.5 or smooth.max() < 0.5:
            out["status"] = "failed"
            out["error"] = "marching_cubes_level_outside_volume_range"
            return out

        vertices, faces, _, _ = measure.marching_cubes(
            smooth,
            level=0.5,
            spacing=voxel_sizes,
        )
        vertices = vertices + (min_idx.astype(np.float64) * np.asarray(voxel_sizes))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        components = sorted(
            mesh.split(only_watertight=False), key=lambda item: len(item.faces), reverse=True
        )
        mesh = components[0].copy()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()

        ply_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(ply_path)

        out.update(mesh_qc(mesh))
        out.update(
            {
                "source_voxel_components": source_components,
                "source_component_sizes": "|".join(str(size) for size in source_sizes[:10]),
                "repaired_voxel_components": repaired_components,
                "repaired_component_sizes": "|".join(str(size) for size in repaired_sizes[:10]),
                "crop_x0": int(min_idx[0]),
                "crop_y0": int(min_idx[1]),
                "crop_z0": int(min_idx[2]),
                "crop_x1": int(max_idx[0]),
                "crop_y1": int(max_idx[1]),
                "crop_z1": int(max_idx[2]),
            }
        )
        out["status"] = "ok"
        return out
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        return out


def run_tasks(tasks: list[MaskTask], workers: int) -> pd.DataFrame:
    if not tasks:
        return pd.DataFrame()

    print(f"Processing {len(tasks)} masks with {workers} worker(s)")
    records = []
    if workers <= 1:
        for i, task in enumerate(tasks, start=1):
            records.append(process_mask(task))
            if i % 50 == 0 or i == len(tasks):
                print(f"  processed {i}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_mask, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if i % 100 == 0 or i == len(tasks):
                    print(f"  processed {i}/{len(tasks)}")

    df = pd.DataFrame(records)
    sort_cols = [col for col in ["RID", "month_from_viscode", "VISCODE", "hemi"] if col in df]
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def build_wide_volumes(volume_long: pd.DataFrame) -> pd.DataFrame:
    if volume_long.empty:
        return pd.DataFrame()

    successful = volume_long[volume_long["status"].isin(["ok", "skipped_existing"])].copy()
    if successful.empty:
        return pd.DataFrame()

    value_cols = [
        "mask_volume_mm3",
        "mesh_volume_mm3",
        "surface_area_mm2",
        "vertices",
        "triangles",
        "watertight",
        "euler_number",
        "mesh_components",
        "mask_voxels",
    ]
    wide_parts = []
    for value in value_cols:
        if value not in successful.columns:
            continue
        pivot = successful.pivot_table(
            index=["RID", "VISCODE", "month_from_viscode"],
            columns="hemi",
            values=value,
            aggfunc="first",
        ).reset_index()
        rename = {}
        if "L" in pivot.columns:
            rename["L"] = f"left_{value}"
        if "R" in pivot.columns:
            rename["R"] = f"right_{value}"
        pivot = pivot.rename(columns=rename)
        wide_parts.append(pivot)

    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on=["RID", "VISCODE", "month_from_viscode"], how="outer")

    for prefix in ["mask_volume_mm3", "mesh_volume_mm3", "surface_area_mm2", "mask_voxels"]:
        left = f"left_{prefix}"
        right = f"right_{prefix}"
        total = f"total_{prefix}"
        if left in wide.columns and right in wide.columns:
            wide[total] = np.where(
                wide[left].notna() & wide[right].notna(),
                wide[left] + wide[right],
                np.nan,
            )

    if {"left_mask_volume_mm3", "right_mask_volume_mm3"}.issubset(wide.columns):
        denom = (wide["left_mask_volume_mm3"] + wide["right_mask_volume_mm3"]) / 2.0
        wide["mask_asymmetry_index"] = (
            wide["left_mask_volume_mm3"] - wide["right_mask_volume_mm3"]
        ) / denom.replace(0, np.nan)

    return wide.sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable")


def write_csvs(
    output_dir: Path,
    clinical: pd.DataFrame,
    subject_summary: pd.DataFrame,
    mask_manifest: pd.DataFrame,
    volume_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical_cols = [col for col in CLINICAL_COLUMNS if col in clinical.columns]
    clean_clinical = clinical.loc[
        clinical["visit_dx_3class"].isin(["AD", "CN", "MCI"]), clinical_cols
    ].copy()
    clean_clinical.to_csv(output_dir / "clean_clinical_longitudinal.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_conversion_summary.csv", index=False)

    manifest = mask_manifest.merge(
        clean_clinical[["RID", "VISCODE", "visit_dx_3class"]],
        on=["RID", "VISCODE"],
        how="left",
    )
    manifest["has_clinical_label"] = manifest["visit_dx_3class"].notna()
    manifest.to_csv(output_dir / "mask_manifest.csv", index=False)
    manifest.loc[~manifest["has_clinical_label"]].to_csv(
        output_dir / "unmatched_masks.csv", index=False
    )

    clinical_mask_keys = set(mask_manifest[["RID", "VISCODE"]].itertuples(index=False, name=None))
    clean_clinical.loc[
        ~clean_clinical[["RID", "VISCODE"]].apply(tuple, axis=1).isin(clinical_mask_keys)
    ].to_csv(output_dir / "unmatched_clinical.csv", index=False)

    volume_long.to_csv(output_dir / "hipp_volumes_long.csv", index=False)
    failed = volume_long.loc[~volume_long["status"].isin(["ok", "skipped_existing"])].copy()
    failed.to_csv(output_dir / "failed_conversions.csv", index=False)

    volume_wide = build_wide_volumes(volume_long)
    volume_wide.to_csv(output_dir / "hipp_volumes_wide.csv", index=False)

    merged = clean_clinical.merge(
        volume_wide,
        on=["RID", "VISCODE", "month_from_viscode"],
        how="inner",
    )
    merged.to_csv(output_dir / "clinical_volume_merged.csv", index=False)
    return clean_clinical, merged


def save_plot(fig: plt.Figure, plots_dir: Path, name: str, plot_format: str) -> None:
    fig.tight_layout()
    fig.savefig(plots_dir / f"{name}.{plot_format}", dpi=180)
    plt.close(fig)


def make_plots(
    merged: pd.DataFrame,
    volume_long: pd.DataFrame,
    subject_summary: pd.DataFrame,
    plots_dir: Path,
    plot_format: str,
    spaghetti_max_subjects: int,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    plot_df = merged.copy()
    if plot_df.empty:
        print("No merged clinical-volume rows available for plots")
        return

    dx_order = ["CN", "MCI", "AD"]
    palette = {"CN": "#2E86AB", "MCI": "#F18F01", "AD": "#C73E1D"}

    for volume_col, title in [
        ("left_mask_volume_mm3", "Left Hippocampus Mask Volume"),
        ("right_mask_volume_mm3", "Right Hippocampus Mask Volume"),
        ("total_mask_volume_mm3", "Total Hippocampus Mask Volume"),
    ]:
        if volume_col not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=plot_df,
            x="AGE",
            y=volume_col,
            hue="visit_dx_3class",
            hue_order=dx_order,
            palette=palette,
            alpha=0.45,
            s=24,
            ax=ax,
        )
        sns.regplot(
            data=plot_df,
            x="AGE",
            y=volume_col,
            scatter=False,
            lowess=True,
            color="black",
            line_kws={"linewidth": 2},
            ax=ax,
        )
        ax.set_title(f"{title} vs Age")
        ax.set_xlabel("Age")
        ax.set_ylabel("Volume (mm3)")
        save_plot(fig, plots_dir, f"{volume_col}_vs_age_by_dx", plot_format)

    long_cols = [
        ("left_mask_volume_mm3", "Left"),
        ("right_mask_volume_mm3", "Right"),
        ("total_mask_volume_mm3", "Total"),
    ]
    for volume_col, label in long_cols:
        if volume_col not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=plot_df,
            x="month_from_viscode",
            y=volume_col,
            hue="visit_dx_3class",
            hue_order=dx_order,
            palette=palette,
            errorbar="se",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"Mean Longitudinal {label} Hippocampus Volume")
        ax.set_xlabel("Month from baseline")
        ax.set_ylabel("Volume (mm3)")
        save_plot(fig, plots_dir, f"mean_longitudinal_{volume_col}_by_dx", plot_format)

    baseline = plot_df[plot_df["VISCODE"] == "bl"].copy()
    if not baseline.empty and "total_mask_volume_mm3" in baseline.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            data=baseline,
            x="visit_dx_3class",
            y="total_mask_volume_mm3",
            order=dx_order,
            palette=palette,
            hue="visit_dx_3class",
            hue_order=dx_order,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=baseline,
            x="visit_dx_3class",
            y="total_mask_volume_mm3",
            order=dx_order,
            color="black",
            alpha=0.25,
            size=2,
            ax=ax,
        )
        ax.set_title("Baseline Total Hippocampus Volume by Diagnosis")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Volume (mm3)")
        save_plot(fig, plots_dir, "baseline_total_volume_boxplot_by_dx", plot_format)

    if "total_mask_volume_mm3" in plot_df.columns:
        spaghetti = sample_subjects_for_spaghetti(plot_df, spaghetti_max_subjects)
        if not spaghetti.empty:
            grid = sns.relplot(
                data=spaghetti,
                x="month_from_viscode",
                y="total_mask_volume_mm3",
                hue="visit_dx_3class",
                col="visit_dx_3class",
                col_order=dx_order,
                kind="line",
                units="RID",
                estimator=None,
                palette=palette,
                alpha=0.25,
                linewidth=0.8,
                height=4,
                aspect=1.1,
            )
            grid.set_axis_labels("Month from baseline", "Total volume (mm3)")
            grid.fig.suptitle("Subject-Level Total Hippocampus Volume Trajectories", y=1.04)
            grid.fig.savefig(plots_dir / f"spaghetti_total_volume_by_dx.{plot_format}", dpi=180)
            plt.close(grid.fig)

    if {"mask_asymmetry_index", "visit_dx_3class"}.issubset(plot_df.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(
            data=plot_df,
            x="visit_dx_3class",
            y="mask_asymmetry_index",
            order=dx_order,
            palette=palette,
            hue="visit_dx_3class",
            hue_order=dx_order,
            legend=False,
            inner="quartile",
            cut=0,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title("Left-Right Hippocampus Volume Asymmetry")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("(Left - Right) / Mean")
        save_plot(fig, plots_dir, "mask_asymmetry_by_dx", plot_format)

    atrophy = compute_annualized_atrophy(plot_df)
    if not atrophy.empty:
        atrophy.to_csv(plots_dir.parent / "annualized_atrophy_rates.csv", index=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(
            data=atrophy,
            x="subject_conversion_type",
            y="annualized_total_mask_volume_change_mm3",
            color="#8FB339",
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title("Annualized Total Hippocampus Volume Change")
        ax.set_xlabel("Subject conversion type")
        ax.set_ylabel("mm3/year")
        ax.tick_params(axis="x", rotation=35)
        save_plot(fig, plots_dir, "annualized_total_volume_change_by_conversion", plot_format)

    aligned = plot_df.loc[
        plot_df["subject_conversion_type"].isin(["CN_to_MCI", "CN_to_AD", "MCI_to_AD", "CN_to_MCI_to_AD"])
        & plot_df["conversion_month"].notna()
    ].copy()
    if not aligned.empty and "total_mask_volume_mm3" in aligned.columns:
        aligned["month_relative_to_conversion"] = (
            aligned["month_from_viscode"] - aligned["conversion_month"]
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=aligned,
            x="month_relative_to_conversion",
            y="total_mask_volume_mm3",
            hue="subject_conversion_type",
            errorbar="se",
            marker="o",
            ax=ax,
        )
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title("Total Volume Aligned to Conversion Month")
        ax.set_xlabel("Months relative to conversion")
        ax.set_ylabel("Total volume (mm3)")
        save_plot(fig, plots_dir, "total_volume_aligned_to_conversion", plot_format)

    if {"watertight", "status"}.issubset(volume_long.columns):
        qc = volume_long.copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(data=qc, x="status", ax=axes[0], color="#4C78A8")
        axes[0].set_title("Conversion Status")
        axes[0].tick_params(axis="x", rotation=25)
        sns.countplot(data=qc, x="watertight", ax=axes[1], color="#54A24B")
        axes[1].set_title("PLY Watertight QC")
        save_plot(fig, plots_dir, "ply_conversion_qc", plot_format)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=plot_df,
        x="VISCODE",
        hue="visit_dx_3class",
        hue_order=dx_order,
        palette=palette,
        order=sorted(plot_df["VISCODE"].unique(), key=lambda item: month_from_viscode(item)),
        ax=ax,
    )
    ax.set_title("Matched Visit Counts by Diagnosis")
    ax.set_xlabel("Visit")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, plots_dir, "matched_visit_counts_by_dx", plot_format)


def sample_subjects_for_spaghetti(df: pd.DataFrame, max_subjects_per_dx: int) -> pd.DataFrame:
    parts = []
    for dx, group in df.groupby("visit_dx_3class", sort=False):
        rids = sorted(group["RID"].dropna().unique(), key=rid_sort_key)
        if len(rids) > max_subjects_per_dx:
            rids = rids[:max_subjects_per_dx]
        parts.append(group[group["RID"].isin(rids)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compute_annualized_atrophy(df: pd.DataFrame) -> pd.DataFrame:
    if "total_mask_volume_mm3" not in df.columns:
        return pd.DataFrame()
    rows = []
    for rid, group in df.dropna(subset=["total_mask_volume_mm3"]).groupby("RID", sort=False):
        group = group.sort_values(["month_from_viscode", "VISCODE"], kind="stable")
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        delta_months = float(last["month_from_viscode"] - first["month_from_viscode"])
        if delta_months <= 0:
            continue
        rows.append(
            {
                "RID": rid,
                "first_month": first["month_from_viscode"],
                "last_month": last["month_from_viscode"],
                "baseline_dx": first["visit_dx_3class"],
                "last_dx": last["visit_dx_3class"],
                "subject_conversion_type": first.get("subject_conversion_type", ""),
                "baseline_total_mask_volume_mm3": first["total_mask_volume_mm3"],
                "last_total_mask_volume_mm3": last["total_mask_volume_mm3"],
                "annualized_total_mask_volume_change_mm3": (
                    (last["total_mask_volume_mm3"] - first["total_mask_volume_mm3"])
                    / (delta_months / 12.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    mask_dir = base_dir / args.mask_dir_name
    clinical_path = base_dir / args.clinical_csv
    output_dir = base_dir / args.output_dir_name
    plots_dir = output_dir / "plots"

    if not mask_dir.is_dir():
        raise FileNotFoundError(mask_dir)
    if not clinical_path.is_file():
        raise FileNotFoundError(clinical_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base directory: {base_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Clinical CSV: {clinical_path}")
    print(f"Output directory: {output_dir}")

    clinical = read_clinical(clinical_path)
    clinical, subject_summary = classify_subjects(clinical)
    mask_manifest = discover_masks(mask_dir)
    clinical = attach_mask_availability(clinical, mask_manifest)

    tasks = make_tasks(
        mask_manifest=mask_manifest,
        clean_clinical=clinical,
        output_dir=output_dir,
        sigma=args.sigma,
        closing_iterations=args.closing_iterations,
        padding=args.padding,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

    print(f"Clinical rows: {len(clinical)}")
    print(f"Mask files: {len(mask_manifest)}")
    print(f"Labeled mask files queued: {len(tasks)}")
    if args.dry_run:
        print(f"Dry run enabled: processing first {args.dry_run} labeled mask files")

    volume_long = run_tasks(tasks, max(1, args.workers))
    clean_clinical, merged = write_csvs(
        output_dir=output_dir,
        clinical=clinical,
        subject_summary=subject_summary,
        mask_manifest=mask_manifest,
        volume_long=volume_long,
    )

    if not args.no_plots:
        make_plots(
            merged=merged,
            volume_long=volume_long,
            subject_summary=subject_summary,
            plots_dir=plots_dir,
            plot_format=args.plot_format,
            spaghetti_max_subjects=args.spaghetti_max_subjects,
        )

    summary = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "clinical_rows": int(len(clinical)),
        "clean_clinical_rows": int(len(clean_clinical)),
        "mask_files": int(len(mask_manifest)),
        "queued_mask_files": int(len(tasks)),
        "volume_rows": int(len(volume_long)),
        "merged_rows": int(len(merged)),
        "status_counts": volume_long["status"].value_counts(dropna=False).to_dict()
        if not volume_long.empty
        else {},
        "sigma": args.sigma,
        "closing_iterations": args.closing_iterations,
        "padding": args.padding,
        "workers": args.workers,
        "dry_run": args.dry_run,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
