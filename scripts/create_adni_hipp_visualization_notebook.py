#!/usr/bin/env python3
"""Create the ADNI hippocampus visualization notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT_DIR = Path("/home/jakaria/ADNI/ADNI_1_GO_Large/adni_hipp_ply_surfs")
NOTEBOOK_PATH = OUTPUT_DIR / "adni_hipp_volume_shape_visualization.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "inr_sdf",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }

    nb["cells"] = [
        markdown(
            """# ADNI Hippocampus Volume and Surface Visualization

This notebook reads the outputs from `build_adni_hipp_ply_surfs.py` and builds analysis plots for AD/CN/MCI volume trends, longitudinal change, conversion groups, mesh QC, and optional 3D PLY inspection.

Run it after the conversion job finishes for final results. It can also be run while processing is ongoing; it will analyze whatever CSV rows are currently available."""
        ),
        code(
            """from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import trimesh
import plotly.graph_objects as go
from IPython.display import display

try:
    import statsmodels.formula.api as smf
except Exception as exc:
    smf = None
    print(f"statsmodels unavailable: {exc}")

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

sns.set_theme(style="whitegrid", context="notebook")

BASE_DIR = Path("/home/jakaria/ADNI/ADNI_1_GO_Large")
OUT_DIR = BASE_DIR / "adni_hipp_ply_surfs"
PLOTS_DIR = OUT_DIR / "notebook_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DX_ORDER = ["CN", "MCI", "AD"]
DX_PALETTE = {"CN": "#2E86AB", "MCI": "#F18F01", "AD": "#C73E1D"}
CONVERSION_PALETTE = {
    "stable_CN": "#2E86AB",
    "stable_MCI": "#F18F01",
    "stable_AD": "#C73E1D",
    "CN_to_MCI": "#73A942",
    "CN_to_AD": "#7B2CBF",
    "CN_to_MCI_to_AD": "#5A189A",
    "MCI_to_AD": "#D1495B",
    "reverter_or_other": "#6C757D",
    "other_or_mixed": "#6C757D",
}

def savefig(name):
    path = PLOTS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(path)

def month_from_viscode(value):
    if pd.isna(value):
        return np.nan
    value = str(value)
    if value == "bl":
        return 0
    if value.startswith("m") and value[1:].isdigit():
        return int(value[1:])
    return np.nan

print(f"Reading outputs from: {OUT_DIR}")"""
        ),
        markdown("## Load Data"),
        code(
            """csv_paths = {
    "clinical": OUT_DIR / "clean_clinical_longitudinal.csv",
    "merged": OUT_DIR / "clinical_volume_merged.csv",
    "long": OUT_DIR / "hipp_volumes_long.csv",
    "wide": OUT_DIR / "hipp_volumes_wide.csv",
    "subjects": OUT_DIR / "subject_conversion_summary.csv",
    "manifest": OUT_DIR / "mask_manifest.csv",
    "failed": OUT_DIR / "failed_conversions.csv",
    "run_summary": OUT_DIR / "run_summary.json",
}

missing = [name for name, path in csv_paths.items() if name != "run_summary" and not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required output files: {missing}")

clinical = pd.read_csv(csv_paths["clinical"])
merged = pd.read_csv(csv_paths["merged"])
vol_long = pd.read_csv(csv_paths["long"])
vol_wide = pd.read_csv(csv_paths["wide"])
subjects = pd.read_csv(csv_paths["subjects"])
manifest = pd.read_csv(csv_paths["manifest"])
failed = pd.read_csv(csv_paths["failed"]) if csv_paths["failed"].exists() else pd.DataFrame()

for df in [clinical, merged, vol_long, vol_wide, subjects, manifest, failed]:
    if "RID" in df.columns:
        df["RID"] = df["RID"].astype(str)
    if "VISCODE" in df.columns:
        df["VISCODE"] = df["VISCODE"].astype(str)
    if "month_from_viscode" in df.columns:
        df["month_from_viscode"] = pd.to_numeric(df["month_from_viscode"], errors="coerce")

for col in ["AGE", "MMSE", "CDRSB", "ADAS11", "ADAS13", "APOE4", "Month.bl"]:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

if csv_paths["run_summary"].exists():
    run_summary = json.loads(csv_paths["run_summary"].read_text())
else:
    run_summary = {}

summary_rows = [
    ("clinical labeled rows", len(clinical)),
    ("merged visit rows with volumes", len(merged)),
    ("long hemisphere volume rows", len(vol_long)),
    ("wide visit volume rows", len(vol_wide)),
    ("subject conversion rows", len(subjects)),
    ("mask manifest rows", len(manifest)),
    ("failed conversion rows", len(failed)),
]
display(pd.DataFrame(summary_rows, columns=["item", "count"]))
display(pd.Series(run_summary, name="run_summary").to_frame())"""
        ),
        markdown("## Data Readiness and Matching QC"),
        code(
            """fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if "has_clinical_label" in manifest.columns:
    sns.countplot(data=manifest, x="has_clinical_label", ax=axes[0, 0], color="#4C78A8")
    axes[0, 0].set_title("Mask files with clinical labels")
else:
    axes[0, 0].axis("off")

if "status" in vol_long.columns and not vol_long.empty:
    sns.countplot(data=vol_long, x="status", ax=axes[0, 1], color="#54A24B")
    axes[0, 1].set_title("Conversion status")
    axes[0, 1].tick_params(axis="x", rotation=25)
else:
    axes[0, 1].axis("off")

if "visit_dx_3class" in clinical.columns:
    sns.countplot(data=clinical, x="visit_dx_3class", order=DX_ORDER, palette=DX_PALETTE, hue="visit_dx_3class", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("Clinical visit diagnosis distribution")
else:
    axes[1, 0].axis("off")

if "subject_conversion_type" in subjects.columns:
    order = subjects["subject_conversion_type"].value_counts().index
    sns.countplot(data=subjects, y="subject_conversion_type", order=order, color="#6A994E", ax=axes[1, 1])
    axes[1, 1].set_title("Subject conversion groups")
else:
    axes[1, 1].axis("off")

savefig("01_data_readiness_qc")
plt.show()

tables = {}
if "visit_dx_3class" in merged.columns:
    tables["visits_by_dx"] = merged["visit_dx_3class"].value_counts(dropna=False)
if "subject_conversion_type" in subjects.columns:
    tables["subjects_by_conversion"] = subjects["subject_conversion_type"].value_counts(dropna=False)
if "VISCODE" in merged.columns:
    tables["visits_by_viscode"] = merged["VISCODE"].value_counts().sort_index(key=lambda idx: idx.map(month_from_viscode))

for name, table in tables.items():
    print(f"\\n{name}")
    display(table.to_frame("count"))"""
        ),
        markdown("## Baseline Volume Distributions"),
        code(
            """baseline = merged[merged["VISCODE"].eq("bl")].copy()
mask_volume_cols = ["left_mask_volume_mm3", "right_mask_volume_mm3", "total_mask_volume_mm3"]
mesh_volume_cols = ["left_mesh_volume_mm3", "right_mesh_volume_mm3", "total_mesh_volume_mm3"]
volume_cols = mask_volume_cols
available_mesh_volume_cols = [col for col in mesh_volume_cols if col in merged.columns]
all_volume_cols = mask_volume_cols + available_mesh_volume_cols

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
for ax, col in zip(axes, volume_cols):
    if col not in baseline.columns:
        ax.axis("off")
        continue
    sns.violinplot(
        data=baseline,
        x="visit_dx_3class",
        y=col,
        order=DX_ORDER,
        palette=DX_PALETTE,
        hue="visit_dx_3class",
        legend=False,
        inner="quartile",
        cut=0,
        ax=ax,
    )
    sns.stripplot(
        data=baseline,
        x="visit_dx_3class",
        y=col,
        order=DX_ORDER,
        color="black",
        alpha=0.22,
        size=2,
        ax=ax,
    )
    ax.set_title(col.replace("_", " "))
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Volume (mm3)")
savefig("02_baseline_volume_distributions")
plt.show()

baseline_summary = (
    baseline.groupby("visit_dx_3class")[volume_cols]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reindex(DX_ORDER)
)
display(baseline_summary)"""
        ),
        markdown("## Mesh PLY Volume Distributions"),
        code(
            """if available_mesh_volume_cols:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, col in zip(axes, mesh_volume_cols):
        if col not in baseline.columns:
            ax.axis("off")
            continue
        sns.violinplot(
            data=baseline,
            x="visit_dx_3class",
            y=col,
            order=DX_ORDER,
            palette=DX_PALETTE,
            hue="visit_dx_3class",
            legend=False,
            inner="quartile",
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=baseline,
            x="visit_dx_3class",
            y=col,
            order=DX_ORDER,
            color="black",
            alpha=0.22,
            size=2,
            ax=ax,
        )
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("PLY mesh volume (mm3)")
    savefig("03_baseline_mesh_volume_distributions")
    plt.show()

    mesh_baseline_summary = (
        baseline.groupby("visit_dx_3class")[available_mesh_volume_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reindex(DX_ORDER)
    )
    display(mesh_baseline_summary)
else:
    print("No mesh volume columns available yet.")"""
        ),
        markdown("## Mask Volume Versus Mesh PLY Volume"),
        code(
            """comparison_pairs = [
    ("left_mask_volume_mm3", "left_mesh_volume_mm3", "left"),
    ("right_mask_volume_mm3", "right_mesh_volume_mm3", "right"),
    ("total_mask_volume_mm3", "total_mesh_volume_mm3", "total"),
]
comparison_pairs = [(mask_col, mesh_col, label) for mask_col, mesh_col, label in comparison_pairs if mask_col in merged.columns and mesh_col in merged.columns]

mesh_bias = merged.copy()
for mask_col, mesh_col, label in comparison_pairs:
    mesh_bias[f"{label}_mesh_minus_mask_mm3"] = mesh_bias[mesh_col] - mesh_bias[mask_col]
    mesh_bias[f"{label}_mesh_minus_mask_pct"] = 100.0 * mesh_bias[f"{label}_mesh_minus_mask_mm3"] / mesh_bias[mask_col].replace(0, np.nan)

if comparison_pairs:
    fig, axes = plt.subplots(2, len(comparison_pairs), figsize=(6 * len(comparison_pairs), 10), squeeze=False)
    for col_idx, (mask_col, mesh_col, label) in enumerate(comparison_pairs):
        ax = axes[0, col_idx]
        sns.scatterplot(
            data=mesh_bias,
            x=mask_col,
            y=mesh_col,
            hue="visit_dx_3class",
            hue_order=DX_ORDER,
            palette=DX_PALETTE,
            alpha=0.45,
            s=24,
            ax=ax,
        )
        low = np.nanmin(mesh_bias[[mask_col, mesh_col]].values)
        high = np.nanmax(mesh_bias[[mask_col, mesh_col]].values)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{label.title()}: mesh vs mask volume")
        ax.set_xlabel("Mask volume (mm3)")
        ax.set_ylabel("PLY mesh volume (mm3)")

        ax = axes[1, col_idx]
        pct_col = f"{label}_mesh_minus_mask_pct"
        sns.boxplot(
            data=mesh_bias,
            x="visit_dx_3class",
            y=pct_col,
            order=DX_ORDER,
            palette=DX_PALETTE,
            hue="visit_dx_3class",
            legend=False,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{label.title()}: mesh minus mask (%)")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Mesh bias (%)")
    savefig("04_mask_vs_mesh_volume_bias")
    plt.show()

    bias_cols = [f"{label}_mesh_minus_mask_mm3" for _, _, label in comparison_pairs] + [f"{label}_mesh_minus_mask_pct" for _, _, label in comparison_pairs]
    display(mesh_bias.groupby("visit_dx_3class")[bias_cols].agg(["count", "mean", "std", "median"]).reindex(DX_ORDER))
else:
    print("No matching mask and mesh volume columns available.")"""
        ),
        markdown("## Volume Versus Age"),
        code(
            """fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, col in zip(axes, volume_cols):
    if col not in merged.columns:
        ax.axis("off")
        continue
    sns.scatterplot(
        data=merged,
        x="AGE",
        y=col,
        hue="visit_dx_3class",
        hue_order=DX_ORDER,
        palette=DX_PALETTE,
        alpha=0.40,
        s=22,
        ax=ax,
    )
    for dx in DX_ORDER:
        sub = merged[merged["visit_dx_3class"].eq(dx)]
        if len(sub) > 20:
            sns.regplot(
                data=sub,
                x="AGE",
                y=col,
                scatter=False,
                lowess=True,
                color=DX_PALETTE[dx],
                line_kws={"linewidth": 2},
                ax=ax,
            )
    ax.set_title(col.replace("_", " "))
    ax.set_xlabel("Age")
    ax.set_ylabel("Volume (mm3)")
savefig("03_volume_vs_age_by_diagnosis")
plt.show()"""
        ),
        markdown("## Mesh PLY Volume Versus Age"),
        code(
            """if available_mesh_volume_cols:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, col in zip(axes, mesh_volume_cols):
        if col not in merged.columns:
            ax.axis("off")
            continue
        sns.scatterplot(
            data=merged,
            x="AGE",
            y=col,
            hue="visit_dx_3class",
            hue_order=DX_ORDER,
            palette=DX_PALETTE,
            alpha=0.40,
            s=22,
            ax=ax,
        )
        for dx in DX_ORDER:
            sub = merged[merged["visit_dx_3class"].eq(dx)]
            if len(sub) > 20:
                sns.regplot(
                    data=sub,
                    x="AGE",
                    y=col,
                    scatter=False,
                    lowess=True,
                    color=DX_PALETTE[dx],
                    line_kws={"linewidth": 2},
                    ax=ax,
                )
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("Age")
        ax.set_ylabel("PLY mesh volume (mm3)")
    savefig("05_mesh_volume_vs_age_by_diagnosis")
    plt.show()
else:
    print("No mesh volume columns available yet.")"""
        ),
        markdown("## Longitudinal Mean Trends"),
        code(
            """fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
for ax, col in zip(axes, volume_cols):
    if col not in merged.columns:
        ax.axis("off")
        continue
    sns.lineplot(
        data=merged,
        x="month_from_viscode",
        y=col,
        hue="visit_dx_3class",
        hue_order=DX_ORDER,
        palette=DX_PALETTE,
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        ax=ax,
    )
    ax.set_title(col.replace("_", " "))
    ax.set_xlabel("Month from baseline")
    ax.set_ylabel("Volume (mm3)")
savefig("04_mean_longitudinal_volume_by_diagnosis")
plt.show()

longitudinal_summary = (
    merged.groupby(["visit_dx_3class", "month_from_viscode"])[volume_cols]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
)
display(longitudinal_summary.head(30))"""
        ),
        markdown("## Mesh PLY Longitudinal Mean Trends"),
        code(
            """if available_mesh_volume_cols:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
    for ax, col in zip(axes, mesh_volume_cols):
        if col not in merged.columns:
            ax.axis("off")
            continue
        sns.lineplot(
            data=merged,
            x="month_from_viscode",
            y=col,
            hue="visit_dx_3class",
            hue_order=DX_ORDER,
            palette=DX_PALETTE,
            estimator="mean",
            errorbar=("ci", 95),
            marker="o",
            ax=ax,
        )
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("Month from baseline")
        ax.set_ylabel("PLY mesh volume (mm3)")
    savefig("06_mean_longitudinal_mesh_volume_by_diagnosis")
    plt.show()

    mesh_longitudinal_summary = (
        merged.groupby(["visit_dx_3class", "month_from_viscode"])[available_mesh_volume_cols]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    display(mesh_longitudinal_summary.head(30))
else:
    print("No mesh volume columns available yet.")"""
        ),
        markdown("## Subject Trajectories"),
        code(
            """def sample_subjects(df, max_per_group=120, group_col="visit_dx_3class"):
    parts = []
    for group_name, group in df.groupby(group_col, sort=False):
        rids = sorted(group["RID"].dropna().unique(), key=lambda x: int(x) if str(x).isdigit() else 10**12)
        rids = rids[:max_per_group]
        parts.append(group[group["RID"].isin(rids)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

spaghetti = sample_subjects(merged.dropna(subset=["total_mask_volume_mm3"]), max_per_group=120)
if not spaghetti.empty:
    grid = sns.relplot(
        data=spaghetti,
        x="month_from_viscode",
        y="total_mask_volume_mm3",
        col="visit_dx_3class",
        col_order=DX_ORDER,
        hue="visit_dx_3class",
        hue_order=DX_ORDER,
        units="RID",
        estimator=None,
        kind="line",
        palette=DX_PALETTE,
        alpha=0.22,
        linewidth=0.9,
        height=4,
        aspect=1.2,
    )
    grid.set_axis_labels("Month from baseline", "Total hippocampus volume (mm3)")
    grid.fig.suptitle("Subject-Level Total Hippocampus Volume Trajectories", y=1.05)
    grid.fig.savefig(PLOTS_DIR / "05_subject_spaghetti_by_diagnosis.png", dpi=180, bbox_inches="tight")
    plt.show()
else:
    print("No trajectory data available.")"""
        ),
        markdown("## Mesh PLY Subject Trajectories"),
        code(
            """if "total_mesh_volume_mm3" in merged.columns:
    mesh_spaghetti = sample_subjects(merged.dropna(subset=["total_mesh_volume_mm3"]), max_per_group=120)
    if not mesh_spaghetti.empty:
        grid = sns.relplot(
            data=mesh_spaghetti,
            x="month_from_viscode",
            y="total_mesh_volume_mm3",
            col="visit_dx_3class",
            col_order=DX_ORDER,
            hue="visit_dx_3class",
            hue_order=DX_ORDER,
            units="RID",
            estimator=None,
            kind="line",
            palette=DX_PALETTE,
            alpha=0.22,
            linewidth=0.9,
            height=4,
            aspect=1.2,
        )
        grid.set_axis_labels("Month from baseline", "Total PLY mesh volume (mm3)")
        grid.fig.suptitle("Subject-Level Total PLY Mesh Volume Trajectories", y=1.05)
        grid.fig.savefig(PLOTS_DIR / "mesh_subject_spaghetti_by_diagnosis.png", dpi=180, bbox_inches="tight")
        plt.show()
    else:
        print("No mesh trajectory data available.")
else:
    print("No total_mesh_volume_mm3 column available.")"""
        ),
        markdown("## Baseline-Normalized Change"),
        code(
            """base_values = (
    merged.sort_values(["RID", "month_from_viscode"])
    .groupby("RID", as_index=False)
    .first()[["RID", "total_mask_volume_mm3", "left_mask_volume_mm3", "right_mask_volume_mm3"]]
    .rename(
        columns={
            "total_mask_volume_mm3": "baseline_total_mask_volume_mm3",
            "left_mask_volume_mm3": "baseline_left_mask_volume_mm3",
            "right_mask_volume_mm3": "baseline_right_mask_volume_mm3",
        }
    )
)
change = merged.merge(base_values, on="RID", how="left")
for side in ["total", "left", "right"]:
    cur = f"{side}_mask_volume_mm3"
    base = f"baseline_{side}_mask_volume_mm3"
    if cur in change.columns and base in change.columns:
        change[f"{side}_delta_mm3"] = change[cur] - change[base]
        change[f"{side}_delta_pct"] = 100.0 * change[f"{side}_delta_mm3"] / change[base].replace(0, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(
    data=change,
    x="month_from_viscode",
    y="total_delta_mm3",
    hue="visit_dx_3class",
    hue_order=DX_ORDER,
    palette=DX_PALETTE,
    estimator="mean",
    errorbar=("ci", 95),
    marker="o",
    ax=axes[0],
)
axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
axes[0].set_title("Mean total volume change from subject baseline")
axes[0].set_xlabel("Month from baseline")
axes[0].set_ylabel("Delta volume (mm3)")

sns.lineplot(
    data=change,
    x="month_from_viscode",
    y="total_delta_pct",
    hue="visit_dx_3class",
    hue_order=DX_ORDER,
    palette=DX_PALETTE,
    estimator="mean",
    errorbar=("ci", 95),
    marker="o",
    ax=axes[1],
)
axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
axes[1].set_title("Mean percent total volume change from subject baseline")
axes[1].set_xlabel("Month from baseline")
axes[1].set_ylabel("Delta volume (%)")
savefig("06_baseline_normalized_change")
plt.show()

display(change[["RID", "VISCODE", "visit_dx_3class", "subject_conversion_type", "total_mask_volume_mm3", "total_delta_mm3", "total_delta_pct"]].head())"""
        ),
        markdown("## Mesh PLY Baseline-Normalized Change"),
        code(
            """if {"total_mesh_volume_mm3", "left_mesh_volume_mm3", "right_mesh_volume_mm3"}.issubset(merged.columns):
    mesh_base_values = (
        merged.sort_values(["RID", "month_from_viscode"])
        .groupby("RID", as_index=False)
        .first()[["RID", "total_mesh_volume_mm3", "left_mesh_volume_mm3", "right_mesh_volume_mm3"]]
        .rename(
            columns={
                "total_mesh_volume_mm3": "baseline_total_mesh_volume_mm3",
                "left_mesh_volume_mm3": "baseline_left_mesh_volume_mm3",
                "right_mesh_volume_mm3": "baseline_right_mesh_volume_mm3",
            }
        )
    )
    mesh_change = merged.merge(mesh_base_values, on="RID", how="left")
    for side in ["total", "left", "right"]:
        cur = f"{side}_mesh_volume_mm3"
        base = f"baseline_{side}_mesh_volume_mm3"
        mesh_change[f"{side}_mesh_delta_mm3"] = mesh_change[cur] - mesh_change[base]
        mesh_change[f"{side}_mesh_delta_pct"] = 100.0 * mesh_change[f"{side}_mesh_delta_mm3"] / mesh_change[base].replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(
        data=mesh_change,
        x="month_from_viscode",
        y="total_mesh_delta_mm3",
        hue="visit_dx_3class",
        hue_order=DX_ORDER,
        palette=DX_PALETTE,
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Mean total PLY mesh volume change from subject baseline")
    axes[0].set_xlabel("Month from baseline")
    axes[0].set_ylabel("Delta PLY mesh volume (mm3)")

    sns.lineplot(
        data=mesh_change,
        x="month_from_viscode",
        y="total_mesh_delta_pct",
        hue="visit_dx_3class",
        hue_order=DX_ORDER,
        palette=DX_PALETTE,
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Mean percent total PLY mesh volume change from subject baseline")
    axes[1].set_xlabel("Month from baseline")
    axes[1].set_ylabel("Delta PLY mesh volume (%)")
    savefig("07_mesh_baseline_normalized_change")
    plt.show()

    display(mesh_change[["RID", "VISCODE", "visit_dx_3class", "subject_conversion_type", "total_mesh_volume_mm3", "total_mesh_delta_mm3", "total_mesh_delta_pct"]].head())
else:
    print("Mesh volume columns are not available.")"""
        ),
        markdown("## Annualized Atrophy Rates"),
        code(
            """def annualized_change_table(df):
    rows = []
    for rid, group in df.dropna(subset=["total_mask_volume_mm3"]).groupby("RID", sort=False):
        group = group.sort_values(["month_from_viscode", "VISCODE"])
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        months = last["month_from_viscode"] - first["month_from_viscode"]
        if not np.isfinite(months) or months <= 0:
            continue
        rows.append(
            {
                "RID": rid,
                "first_dx": first["visit_dx_3class"],
                "last_dx": last["visit_dx_3class"],
                "subject_conversion_type": first.get("subject_conversion_type", ""),
                "first_month": first["month_from_viscode"],
                "last_month": last["month_from_viscode"],
                "baseline_total_mask_volume_mm3": first["total_mask_volume_mm3"],
                "last_total_mask_volume_mm3": last["total_mask_volume_mm3"],
                "annualized_change_mm3": (last["total_mask_volume_mm3"] - first["total_mask_volume_mm3"]) / (months / 12.0),
                "annualized_change_pct": 100.0 * (last["total_mask_volume_mm3"] - first["total_mask_volume_mm3"]) / first["total_mask_volume_mm3"] / (months / 12.0),
            }
        )
    return pd.DataFrame(rows)

atrophy = annualized_change_table(merged)
display(atrophy.head())
atrophy.to_csv(OUT_DIR / "notebook_annualized_atrophy_rates.csv", index=False)

if not atrophy.empty:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.boxplot(data=atrophy, x="first_dx", y="annualized_change_mm3", order=DX_ORDER, palette=DX_PALETTE, hue="first_dx", legend=False, ax=axes[0])
    sns.stripplot(data=atrophy, x="first_dx", y="annualized_change_mm3", order=DX_ORDER, color="black", alpha=0.25, size=2, ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Annualized volume change by baseline diagnosis")
    axes[0].set_xlabel("Baseline diagnosis")
    axes[0].set_ylabel("mm3/year")

    order = atrophy["subject_conversion_type"].value_counts().index
    sns.boxplot(data=atrophy, x="subject_conversion_type", y="annualized_change_mm3", order=order, color="#8FB339", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Annualized volume change by conversion group")
    axes[1].set_xlabel("Conversion group")
    axes[1].set_ylabel("mm3/year")
    savefig("07_annualized_atrophy_rates")
    plt.show()"""
        ),
        markdown("## Mesh PLY Annualized Volume Change"),
        code(
            """def annualized_mesh_change_table(df):
    if "total_mesh_volume_mm3" not in df.columns:
        return pd.DataFrame()
    rows = []
    for rid, group in df.dropna(subset=["total_mesh_volume_mm3"]).groupby("RID", sort=False):
        group = group.sort_values(["month_from_viscode", "VISCODE"])
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        months = last["month_from_viscode"] - first["month_from_viscode"]
        if not np.isfinite(months) or months <= 0:
            continue
        rows.append(
            {
                "RID": rid,
                "first_dx": first["visit_dx_3class"],
                "last_dx": last["visit_dx_3class"],
                "subject_conversion_type": first.get("subject_conversion_type", ""),
                "first_month": first["month_from_viscode"],
                "last_month": last["month_from_viscode"],
                "baseline_total_mesh_volume_mm3": first["total_mesh_volume_mm3"],
                "last_total_mesh_volume_mm3": last["total_mesh_volume_mm3"],
                "annualized_mesh_change_mm3": (last["total_mesh_volume_mm3"] - first["total_mesh_volume_mm3"]) / (months / 12.0),
                "annualized_mesh_change_pct": 100.0 * (last["total_mesh_volume_mm3"] - first["total_mesh_volume_mm3"]) / first["total_mesh_volume_mm3"] / (months / 12.0),
            }
        )
    return pd.DataFrame(rows)

mesh_atrophy = annualized_mesh_change_table(merged)
display(mesh_atrophy.head())
mesh_atrophy.to_csv(OUT_DIR / "notebook_mesh_annualized_atrophy_rates.csv", index=False)

if not mesh_atrophy.empty:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.boxplot(data=mesh_atrophy, x="first_dx", y="annualized_mesh_change_mm3", order=DX_ORDER, palette=DX_PALETTE, hue="first_dx", legend=False, ax=axes[0])
    sns.stripplot(data=mesh_atrophy, x="first_dx", y="annualized_mesh_change_mm3", order=DX_ORDER, color="black", alpha=0.25, size=2, ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Annualized PLY mesh volume change by baseline diagnosis")
    axes[0].set_xlabel("Baseline diagnosis")
    axes[0].set_ylabel("mm3/year")

    order = mesh_atrophy["subject_conversion_type"].value_counts().index
    sns.boxplot(data=mesh_atrophy, x="subject_conversion_type", y="annualized_mesh_change_mm3", order=order, color="#8FB339", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Annualized PLY mesh volume change by conversion group")
    axes[1].set_xlabel("Conversion group")
    axes[1].set_ylabel("mm3/year")
    savefig("08_mesh_annualized_atrophy_rates")
    plt.show()
else:
    print("No mesh annualized atrophy rows available yet.")"""
        ),
        markdown("## Conversion-Aligned Trends"),
        code(
            """aligned = merged[
    merged["subject_conversion_type"].isin(["CN_to_MCI", "CN_to_AD", "CN_to_MCI_to_AD", "MCI_to_AD"])
    & merged["conversion_month"].notna()
].copy()

if not aligned.empty:
    aligned["month_relative_to_conversion"] = aligned["month_from_viscode"] - aligned["conversion_month"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    sns.lineplot(
        data=aligned,
        x="month_relative_to_conversion",
        y="total_mask_volume_mm3",
        hue="subject_conversion_type",
        palette=CONVERSION_PALETTE,
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        ax=axes[0],
    )
    axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Total volume aligned to conversion month")
    axes[0].set_xlabel("Months relative to conversion")
    axes[0].set_ylabel("Total volume (mm3)")

    aligned_change = aligned.merge(base_values[["RID", "baseline_total_mask_volume_mm3"]], on="RID", how="left")
    aligned_change["total_delta_pct"] = 100.0 * (
        aligned_change["total_mask_volume_mm3"] - aligned_change["baseline_total_mask_volume_mm3"]
    ) / aligned_change["baseline_total_mask_volume_mm3"].replace(0, np.nan)
    sns.lineplot(
        data=aligned_change,
        x="month_relative_to_conversion",
        y="total_delta_pct",
        hue="subject_conversion_type",
        palette=CONVERSION_PALETTE,
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        ax=axes[1],
    )
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=1, linestyle=":")
    axes[1].set_title("Percent change aligned to conversion month")
    axes[1].set_xlabel("Months relative to conversion")
    axes[1].set_ylabel("Delta from baseline (%)")
    savefig("08_conversion_aligned_trends")
    plt.show()
else:
    print("No conversion-aligned rows available yet. This is expected during a small dry run.")"""
        ),
        markdown("## Left-Right Asymmetry"),
        code(
            """fig, axes = plt.subplots(1, 3, figsize=(19, 5))

if {"left_mask_volume_mm3", "right_mask_volume_mm3"}.issubset(merged.columns):
    sns.scatterplot(
        data=merged,
        x="left_mask_volume_mm3",
        y="right_mask_volume_mm3",
        hue="visit_dx_3class",
        hue_order=DX_ORDER,
        palette=DX_PALETTE,
        alpha=0.45,
        s=24,
        ax=axes[0],
    )
    low = np.nanmin(merged[["left_mask_volume_mm3", "right_mask_volume_mm3"]].values)
    high = np.nanmax(merged[["left_mask_volume_mm3", "right_mask_volume_mm3"]].values)
    axes[0].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Left vs right volume")

if "mask_asymmetry_index" in merged.columns:
    sns.violinplot(
        data=merged,
        x="visit_dx_3class",
        y="mask_asymmetry_index",
        order=DX_ORDER,
        palette=DX_PALETTE,
        hue="visit_dx_3class",
        legend=False,
        inner="quartile",
        cut=0,
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Asymmetry distribution")
    axes[1].set_xlabel("Diagnosis")
    axes[1].set_ylabel("(Left - Right) / mean")

    sns.scatterplot(
        data=merged,
        x="AGE",
        y="mask_asymmetry_index",
        hue="visit_dx_3class",
        hue_order=DX_ORDER,
        palette=DX_PALETTE,
        alpha=0.45,
        s=24,
        ax=axes[2],
    )
    axes[2].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[2].set_title("Asymmetry vs age")
    axes[2].set_xlabel("Age")
savefig("09_left_right_asymmetry")
plt.show()"""
        ),
        markdown("## Cognitive Associations"),
        code(
            """cognitive_cols = [col for col in ["MMSE", "CDRSB", "ADAS11", "ADAS13"] if col in merged.columns]
if cognitive_cols:
    fig, axes = plt.subplots(1, len(cognitive_cols), figsize=(5.8 * len(cognitive_cols), 5), squeeze=False)
    for ax, col in zip(axes.ravel(), cognitive_cols):
        sns.scatterplot(
            data=merged,
            x="total_mask_volume_mm3",
            y=col,
            hue="visit_dx_3class",
            hue_order=DX_ORDER,
            palette=DX_PALETTE,
            alpha=0.45,
            s=24,
            ax=ax,
        )
        sns.regplot(
            data=merged,
            x="total_mask_volume_mm3",
            y=col,
            scatter=False,
            color="black",
            line_kws={"linewidth": 1.5},
            ax=ax,
        )
        ax.set_title(f"{col} vs total volume")
        ax.set_xlabel("Total hippocampus volume (mm3)")
    savefig("10_cognitive_associations_total_volume")
    plt.show()

    if "total_mesh_volume_mm3" in merged.columns:
        fig, axes = plt.subplots(1, len(cognitive_cols), figsize=(5.8 * len(cognitive_cols), 5), squeeze=False)
        for ax, col in zip(axes.ravel(), cognitive_cols):
            sns.scatterplot(
                data=merged,
                x="total_mesh_volume_mm3",
                y=col,
                hue="visit_dx_3class",
                hue_order=DX_ORDER,
                palette=DX_PALETTE,
                alpha=0.45,
                s=24,
                ax=ax,
            )
            sns.regplot(
                data=merged,
                x="total_mesh_volume_mm3",
                y=col,
                scatter=False,
                color="black",
                line_kws={"linewidth": 1.5},
                ax=ax,
            )
            ax.set_title(f"{col} vs PLY mesh volume")
            ax.set_xlabel("Total PLY mesh volume (mm3)")
        savefig("11_cognitive_associations_total_mesh_volume")
        plt.show()

    corr_cols = [
        "total_mask_volume_mm3",
        "left_mask_volume_mm3",
        "right_mask_volume_mm3",
        "total_mesh_volume_mm3",
        "left_mesh_volume_mm3",
        "right_mesh_volume_mm3",
    ] + cognitive_cols + ["AGE", "APOE4"]
    corr_cols = [col for col in corr_cols if col in merged.columns]
    corr = merged[corr_cols].corr(numeric_only=True, method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Spearman correlation matrix")
    savefig("11_spearman_correlation_matrix")
    plt.show()
else:
    print("No cognitive columns found.")"""
        ),
        markdown("## Mesh and Reconstruction QC"),
        code(
            """qc = vol_long.copy()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

if "watertight" in qc.columns:
    sns.countplot(data=qc, x="watertight", color="#4C78A8", ax=axes[0, 0])
    axes[0, 0].set_title("Watertight meshes")

if "euler_number" in qc.columns:
    sns.histplot(data=qc, x="euler_number", bins=30, color="#F18F01", ax=axes[0, 1])
    axes[0, 1].set_title("Euler number")

if "mesh_components" in qc.columns:
    sns.countplot(data=qc, x="mesh_components", color="#54A24B", ax=axes[0, 2])
    axes[0, 2].set_title("Mesh components")

if {"mask_volume_mm3", "mesh_volume_mm3"}.issubset(qc.columns):
    sns.scatterplot(data=qc, x="mask_volume_mm3", y="mesh_volume_mm3", hue="hemi", alpha=0.55, s=24, ax=axes[1, 0])
    axes[1, 0].set_title("Mesh volume vs mask volume")
    low = np.nanmin(qc[["mask_volume_mm3", "mesh_volume_mm3"]].values)
    high = np.nanmax(qc[["mask_volume_mm3", "mesh_volume_mm3"]].values)
    axes[1, 0].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)

if "vertices" in qc.columns:
    sns.histplot(data=qc, x="vertices", hue="hemi", bins=40, element="step", ax=axes[1, 1])
    axes[1, 1].set_title("Vertex counts")

if "source_voxel_components" in qc.columns:
    sns.countplot(data=qc, x="source_voxel_components", hue="hemi", ax=axes[1, 2])
    axes[1, 2].set_title("Source voxel components")

savefig("12_mesh_reconstruction_qc")
plt.show()

qc_flags = qc[
    (qc.get("status", "ok") != "ok")
    | (qc.get("watertight", True) != True)
    | (qc.get("mesh_components", 1) != 1)
].copy()
print(f"QC flagged rows: {len(qc_flags)}")
display(qc_flags.head(30))"""
        ),
        markdown("## Missingness and Visit Coverage"),
        code(
            """visit_order = sorted(clinical["VISCODE"].dropna().unique(), key=month_from_viscode)

coverage = (
    clinical.assign(has_volume=clinical[["RID", "VISCODE"]].apply(tuple, axis=1).isin(set(merged[["RID", "VISCODE"]].apply(tuple, axis=1))))
    .pivot_table(index="visit_dx_3class", columns="VISCODE", values="has_volume", aggfunc="mean")
    .reindex(index=DX_ORDER, columns=visit_order)
)

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(coverage, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
ax.set_title("Fraction of clinical visits with both-volume rows")
ax.set_xlabel("Visit")
ax.set_ylabel("Diagnosis")
savefig("13_visit_coverage_heatmap")
plt.show()

counts = merged.pivot_table(index="visit_dx_3class", columns="VISCODE", values="RID", aggfunc="count").reindex(index=DX_ORDER, columns=visit_order)
display(counts.fillna(0).astype(int))"""
        ),
        markdown("## Optional Statistical Models"),
        code(
            """if smf is None:
    print("statsmodels is unavailable.")
elif len(merged) < 30:
    print("Too few merged rows for meaningful models. Re-run after full processing finishes.")
else:
    model_df = merged.dropna(subset=["total_mask_volume_mm3", "AGE", "visit_dx_3class", "PTGENDER"]).copy()
    model_df["visit_dx_3class"] = pd.Categorical(model_df["visit_dx_3class"], categories=DX_ORDER)
    print(f"Model rows: {len(model_df)}, subjects: {model_df['RID'].nunique()}")

    ols = smf.ols("total_mask_volume_mm3 ~ AGE + C(PTGENDER) + C(visit_dx_3class)", data=model_df).fit()
    print("OLS: total volume ~ age + sex + diagnosis")
    display(ols.summary().tables[1])

    if "total_mesh_volume_mm3" in model_df.columns:
        mesh_model_df = model_df.dropna(subset=["total_mesh_volume_mm3"]).copy()
        mesh_ols = smf.ols("total_mesh_volume_mm3 ~ AGE + C(PTGENDER) + C(visit_dx_3class)", data=mesh_model_df).fit()
        print("OLS: total PLY mesh volume ~ age + sex + diagnosis")
        display(mesh_ols.summary().tables[1])

    if model_df["RID"].nunique() > 20 and len(model_df) > model_df["RID"].nunique():
        mixed = smf.mixedlm(
            "total_mask_volume_mm3 ~ AGE + C(PTGENDER) + C(visit_dx_3class)",
            data=model_df,
            groups=model_df["RID"],
        ).fit(reml=False, method="lbfgs")
        print("Mixed model with subject random intercept")
        display(mixed.summary().tables[1])"""
        ),
        markdown("## 3D PLY Inspection"),
        code(
            """def add_mesh_trace(fig, mesh, name, color):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=1.0,
            flatshading=False,
            lighting=dict(ambient=0.40, diffuse=0.75, specular=0.18, roughness=0.78),
            lightposition=dict(x=100, y=160, z=180),
        )
    )

def show_subject_surface(rid=None, viscode=None):
    available = vol_long[vol_long["status"].isin(["ok", "skipped_existing"])].copy()
    if available.empty:
        raise ValueError("No successful PLY rows available.")
    if rid is None:
        rid = str(available.iloc[0]["RID"])
    if viscode is None:
        viscode = str(available[available["RID"].eq(str(rid))].iloc[0]["VISCODE"])
    rows = available[(available["RID"].eq(str(rid))) & (available["VISCODE"].eq(str(viscode)))]
    if rows.empty:
        raise ValueError(f"No PLY rows found for RID={rid}, VISCODE={viscode}")

    fig = go.Figure()
    for _, row in rows.sort_values("hemi").iterrows():
        mesh = trimesh.load_mesh(row["ply_path"], process=False)
        color = "#2E86DE" if row["hemi"] == "L" else "#E67E22"
        add_mesh_trace(fig, mesh, f"{rid}_{viscode}_{row['hemi']}", color)
    title = f"Hippocampus PLY surfaces: RID={rid}, VISCODE={viscode}"
    fig.update_layout(
        title=title,
        width=900,
        height=700,
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.show()

# Change these values to inspect another subject.
example_rid = str(merged.iloc[0]["RID"]) if not merged.empty else None
example_viscode = str(merged.iloc[0]["VISCODE"]) if not merged.empty else None
show_subject_surface(example_rid, example_viscode)"""
        ),
        markdown("## Export Analysis Tables"),
        code(
            """analysis_dir = OUT_DIR / "notebook_analysis_tables"
analysis_dir.mkdir(parents=True, exist_ok=True)

baseline.to_csv(analysis_dir / "baseline_rows.csv", index=False)
change.to_csv(analysis_dir / "baseline_normalized_change.csv", index=False)
atrophy.to_csv(analysis_dir / "annualized_atrophy_rates.csv", index=False)
if "mesh_atrophy" in globals():
    mesh_atrophy.to_csv(analysis_dir / "mesh_annualized_atrophy_rates.csv", index=False)
if "mesh_bias" in globals():
    mesh_bias.to_csv(analysis_dir / "mask_vs_mesh_volume_bias.csv", index=False)
if "mesh_change" in globals():
    mesh_change.to_csv(analysis_dir / "mesh_baseline_normalized_change.csv", index=False)

if "baseline_summary" in globals():
    baseline_summary.to_csv(analysis_dir / "baseline_volume_summary.csv")
if "mesh_baseline_summary" in globals():
    mesh_baseline_summary.to_csv(analysis_dir / "baseline_mesh_volume_summary.csv")
if "longitudinal_summary" in globals():
    longitudinal_summary.to_csv(analysis_dir / "longitudinal_volume_summary.csv", index=False)
if "mesh_longitudinal_summary" in globals():
    mesh_longitudinal_summary.to_csv(analysis_dir / "longitudinal_mesh_volume_summary.csv", index=False)

print(f"Saved analysis tables to {analysis_dir}")
print(f"Saved notebook figures to {PLOTS_DIR}")"""
        ),
    ]

    for index, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{index:02d}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
